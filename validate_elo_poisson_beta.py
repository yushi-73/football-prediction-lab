#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
J1/J2統合Eloを、J1リーグ順位シミュレーション用のポアソンλ補正として検証するコード。

目的：
- beta_elo = 0.00 を「Elo補正なし」の基準にする
- beta_elo = 0.05, 0.10, ... を横並び比較する
- 各シーズンの途中時点から残り試合をポアソン分布でシミュレーションし、最終順位MAEを評価する

前提入力：
1. j1_j2_elo_input_1993_2025.csv
   - date, year, division, home, away, home_goal, away_goal を含む
2. elo_outputs/j1_j2_elo_match_history.csv
   - build_j1_j2_elo.py の出力
   - date, year, home, away, home_rating_post, away_rating_post などを含む
3. elo_outputs/j1_j2_elo_season_start_ratings.csv
   - build_j1_j2_elo.py の出力

基本式：
    lambda_home_base = league_home_avg * home_attack_home * away_defense_away
    lambda_away_base = league_away_avg * away_attack_away * home_defense_home

    elo_factor = exp(beta_elo * (home_elo - away_elo) / 400)
    lambda_home = lambda_home_base * elo_factor
    lambda_away = lambda_away_base / elo_factor

注意：
- 既にホーム/アウェイ別の得点・失点でλを作るため、Elo側に home_adv は足さない設計。
- 未来シミュレーション中はEloを固定する設計。
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


# ==============================
# 設定
# ==============================

@dataclass
class Config:
    results_path: Path
    elo_history_path: Path
    season_start_path: Path
    outdir: Path
    eval_start_year: int
    eval_end_year: int | None
    cutoff_matches_list: List[int]
    betas: List[float]
    n_sims: int
    seed: int
    prior_matches: float
    lambda_cap: float
    min_lambda: float
    strength_low: float
    strength_high: float
    use_j1_only: bool


# ==============================
# 汎用関数
# ==============================

def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def ensure_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} に必要列がありません: {missing}")


def result_points(hg: int, ag: int) -> Tuple[int, int, int, int, int, int]:
    """1試合の勝点・勝敗数を返す。戻り値: hp, ap, hw, hd, hl, aw, ad, al ではなく簡略用。"""
    if hg > ag:
        return 3, 0, 1, 0, 0, 0, 0, 1
    if hg < ag:
        return 0, 3, 0, 0, 1, 1, 0, 0
    return 1, 1, 0, 1, 0, 0, 1, 0


# ==============================
# 順位表作成
# ==============================

def init_table(teams: List[str]) -> Dict[str, Dict[str, int]]:
    return {
        t: {"P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "GD": 0, "Pts": 0}
        for t in teams
    }


def add_match_to_table(table: Dict[str, Dict[str, int]], home: str, away: str, hg: int, ag: int) -> None:
    hp, ap, hw, hd, hl, aw, ad, al = result_points(hg, ag)

    table[home]["P"] += 1
    table[home]["W"] += hw
    table[home]["D"] += hd
    table[home]["L"] += hl
    table[home]["GF"] += int(hg)
    table[home]["GA"] += int(ag)
    table[home]["GD"] = table[home]["GF"] - table[home]["GA"]
    table[home]["Pts"] += hp

    table[away]["P"] += 1
    table[away]["W"] += aw
    table[away]["D"] += ad
    table[away]["L"] += al
    table[away]["GF"] += int(ag)
    table[away]["GA"] += int(hg)
    table[away]["GD"] = table[away]["GF"] - table[away]["GA"]
    table[away]["Pts"] += ap


def build_table(matches: pd.DataFrame, teams: List[str]) -> Dict[str, Dict[str, int]]:
    table = init_table(teams)
    for r in matches.itertuples(index=False):
        add_match_to_table(table, r.home, r.away, int(r.home_goal), int(r.away_goal))
    return table


def positions_from_table(table: Dict[str, Dict[str, int]]) -> Dict[str, int]:
    """Jリーグ風に 勝点→得失点差→総得点→勝利数→チーム名 で順位化。"""
    order = sorted(
        table.keys(),
        key=lambda t: (
            -table[t]["Pts"],
            -table[t]["GD"],
            -table[t]["GF"],
            -table[t]["W"],
            t,
        ),
    )
    return {team: i + 1 for i, team in enumerate(order)}


# ==============================
# カットオフ処理
# ==============================

def find_cutoff_date(season_matches: pd.DataFrame, teams: List[str], cutoff_matches: int) -> pd.Timestamp | None:
    """
    全チームが最低 cutoff_matches 試合を終えた最初の日付を返す。
    日付単位でまとめて処理するため、同日試合の途中で切らない。
    """
    played = {t: 0 for t in teams}
    for date, g in season_matches.sort_values(["date", "match_id" if "match_id" in season_matches.columns else "home"]).groupby("date"):
        for r in g.itertuples(index=False):
            played[r.home] += 1
            played[r.away] += 1
        if min(played.values()) >= cutoff_matches:
            return pd.Timestamp(date)
    return None


# ==============================
# λ推定
# ==============================

def safe_rate(goals: float, matches: float, league_avg: float, prior_matches: float) -> float:
    """
    少試合の暴れを抑えるため、リーグ平均を prior_matches 試合分だけ足して平均回帰する。
    """
    if league_avg <= 0:
        league_avg = 1.0
    rate = (goals + prior_matches * league_avg) / max(matches + prior_matches, 1e-9)
    return rate / league_avg


def estimate_strengths(
    known_matches: pd.DataFrame,
    teams: List[str],
    prior_matches: float,
    strength_low: float,
    strength_high: float,
) -> Tuple[float, float, Dict[str, Dict[str, float]]]:
    """
    ホーム/アウェイ別の攻撃力・守備力を推定する。

    home_att: ホームでの得点力
    home_def: ホームでの失点しやすさ。1より大きいほど守備が弱い
    away_att: アウェイでの得点力
    away_def: アウェイでの失点しやすさ。1より大きいほど守備が弱い
    """
    if len(known_matches) == 0:
        raise ValueError("known_matches が空です。cutoff_matches を小さくしてください。")

    league_home_avg = known_matches["home_goal"].mean()
    league_away_avg = known_matches["away_goal"].mean()

    # 異常に小さい場合の保険
    league_home_avg = float(max(league_home_avg, 0.2))
    league_away_avg = float(max(league_away_avg, 0.2))

    raw = {
        t: {
            "hm": 0, "hgf": 0, "hga": 0,
            "am": 0, "agf": 0, "aga": 0,
        }
        for t in teams
    }

    for r in known_matches.itertuples(index=False):
        raw[r.home]["hm"] += 1
        raw[r.home]["hgf"] += int(r.home_goal)
        raw[r.home]["hga"] += int(r.away_goal)

        raw[r.away]["am"] += 1
        raw[r.away]["agf"] += int(r.away_goal)
        raw[r.away]["aga"] += int(r.home_goal)

    strengths: Dict[str, Dict[str, float]] = {}
    for t in teams:
        x = raw[t]
        home_att = safe_rate(x["hgf"], x["hm"], league_home_avg, prior_matches)
        home_def = safe_rate(x["hga"], x["hm"], league_away_avg, prior_matches)
        away_att = safe_rate(x["agf"], x["am"], league_away_avg, prior_matches)
        away_def = safe_rate(x["aga"], x["am"], league_home_avg, prior_matches)

        strengths[t] = {
            "home_att": float(np.clip(home_att, strength_low, strength_high)),
            "home_def": float(np.clip(home_def, strength_low, strength_high)),
            "away_att": float(np.clip(away_att, strength_low, strength_high)),
            "away_def": float(np.clip(away_def, strength_low, strength_high)),
        }

    return league_home_avg, league_away_avg, strengths


def make_lambdas_for_remaining(
    remaining_matches: pd.DataFrame,
    league_home_avg: float,
    league_away_avg: float,
    strengths: Dict[str, Dict[str, float]],
    elos: Dict[str, float],
    beta_elo: float,
    lambda_cap: float,
    min_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    h_lams = []
    a_lams = []

    for r in remaining_matches.itertuples(index=False):
        h = r.home
        a = r.away

        # 基本λ：ホーム/アウェイ別攻守
        lam_h_base = league_home_avg * strengths[h]["home_att"] * strengths[a]["away_def"]
        lam_a_base = league_away_avg * strengths[a]["away_att"] * strengths[h]["home_def"]

        # Elo補正：ホーム補正は入れない。H/A別λが既にホーム性を含むため。
        elo_diff = float(elos.get(h, 1500.0) - elos.get(a, 1500.0))
        elo_factor = math.exp(beta_elo * elo_diff / 400.0)

        lam_h = lam_h_base * elo_factor
        lam_a = lam_a_base / elo_factor

        h_lams.append(float(np.clip(lam_h, min_lambda, lambda_cap)))
        a_lams.append(float(np.clip(lam_a, min_lambda, lambda_cap)))

    return np.array(h_lams, dtype=float), np.array(a_lams, dtype=float)


# ==============================
# Elo取得
# ==============================

def get_cutoff_elos(
    elo_history: pd.DataFrame,
    season_start: pd.DataFrame,
    target_year: int,
    season_start_date: pd.Timestamp,
    cutoff_date: pd.Timestamp,
    teams: List[str],
) -> Dict[str, float]:
    """
    対象年の開幕時点Eloを初期値にし、cutoff_dateまでの実試合で更新されたEloを返す。
    build_j1_j2_elo.py の season_start は年初回帰後の値なので、必ずこちらを初期値にする。
    """
    elos: Dict[str, float] = {}

    ss = season_start[season_start["year"] == target_year]
    for r in ss.itertuples(index=False):
        elos[str(r.team)] = float(r.rating_start)

    # 対象年の公式戦データ内、cutoffまでの更新だけを反映する
    hist = elo_history[
        (elo_history["date"] >= season_start_date)
        & (elo_history["date"] <= cutoff_date)
    ].sort_values(["date", "match_id" if "match_id" in elo_history.columns else "home"])

    for r in hist.itertuples(index=False):
        elos[str(r.home)] = float(r.home_rating_post)
        elos[str(r.away)] = float(r.away_rating_post)

    return {t: float(elos.get(t, 1500.0)) for t in teams}


# ==============================
# 1シーズン検証
# ==============================

def simulate_one_season(
    season_matches: pd.DataFrame,
    known_matches: pd.DataFrame,
    remaining_matches: pd.DataFrame,
    teams: List[str],
    elos: Dict[str, float],
    beta_elo: float,
    cfg: Config,
    rng: np.random.Generator,
) -> Dict[str, float]:
    actual_table = build_table(season_matches, teams)
    actual_pos = positions_from_table(actual_table)

    known_table = build_table(known_matches, teams)

    league_home_avg, league_away_avg, strengths = estimate_strengths(
        known_matches=known_matches,
        teams=teams,
        prior_matches=cfg.prior_matches,
        strength_low=cfg.strength_low,
        strength_high=cfg.strength_high,
    )

    lam_h, lam_a = make_lambdas_for_remaining(
        remaining_matches=remaining_matches,
        league_home_avg=league_home_avg,
        league_away_avg=league_away_avg,
        strengths=strengths,
        elos=elos,
        beta_elo=beta_elo,
        lambda_cap=cfg.lambda_cap,
        min_lambda=cfg.min_lambda,
    )

    team_idx = {t: i for i, t in enumerate(teams)}
    actual_pos_arr = np.array([actual_pos[t] for t in teams], dtype=int)

    # 既知試合終了時点の成績を配列にする
    base_pts = np.array([known_table[t]["Pts"] for t in teams], dtype=int)
    base_gf = np.array([known_table[t]["GF"] for t in teams], dtype=int)
    base_ga = np.array([known_table[t]["GA"] for t in teams], dtype=int)
    base_w = np.array([known_table[t]["W"] for t in teams], dtype=int)

    home_indices = np.array([team_idx[r.home] for r in remaining_matches.itertuples(index=False)], dtype=int)
    away_indices = np.array([team_idx[r.away] for r in remaining_matches.itertuples(index=False)], dtype=int)

    pos_sum = np.zeros(len(teams), dtype=float)
    exact_count = np.zeros(len(teams), dtype=int)
    sim_draws = 0
    sim_goals = 0

    for _ in range(cfg.n_sims):
        pts = base_pts.copy()
        gf = base_gf.copy()
        ga = base_ga.copy()
        wins = base_w.copy()

        hg_arr = rng.poisson(lam_h)
        ag_arr = rng.poisson(lam_a)

        sim_draws += int(np.sum(hg_arr == ag_arr))
        sim_goals += int(np.sum(hg_arr + ag_arr))

        for j, (hi, ai, hg, ag) in enumerate(zip(home_indices, away_indices, hg_arr, ag_arr)):
            gf[hi] += int(hg)
            ga[hi] += int(ag)
            gf[ai] += int(ag)
            ga[ai] += int(hg)

            if hg > ag:
                pts[hi] += 3
                wins[hi] += 1
            elif hg < ag:
                pts[ai] += 3
                wins[ai] += 1
            else:
                pts[hi] += 1
                pts[ai] += 1

        gd = gf - ga
        # 勝点→得失点差→総得点→勝利数→チーム名
        order = sorted(
            range(len(teams)),
            key=lambda i: (-pts[i], -gd[i], -gf[i], -wins[i], teams[i]),
        )
        pos = np.empty(len(teams), dtype=int)
        for rank, i in enumerate(order, start=1):
            pos[i] = rank

        pos_sum += pos
        exact_count += (pos == actual_pos_arr)

    avg_pos = pos_sum / cfg.n_sims
    mae = float(np.mean(np.abs(avg_pos - actual_pos_arr)))
    prob_actual_position = float(np.mean(exact_count / cfg.n_sims))

    actual_remaining_draw_rate = float((remaining_matches["home_goal"] == remaining_matches["away_goal"]).mean())
    actual_remaining_goals_pm = float((remaining_matches["home_goal"] + remaining_matches["away_goal"]).mean())

    sim_total_matches = cfg.n_sims * len(remaining_matches)
    sim_draw_rate = float(sim_draws / sim_total_matches) if sim_total_matches else float("nan")
    sim_goals_pm = float(sim_goals / sim_total_matches) if sim_total_matches else float("nan")

    return {
        "mae": mae,
        "prob_actual_position": prob_actual_position,
        "sim_draw_rate": sim_draw_rate,
        "actual_remaining_draw_rate": actual_remaining_draw_rate,
        "sim_goals_per_match": sim_goals_pm,
        "actual_remaining_goals_per_match": actual_remaining_goals_pm,
        "league_home_avg_known": float(league_home_avg),
        "league_away_avg_known": float(league_away_avg),
        "mean_lambda_home_remaining": float(np.mean(lam_h)),
        "mean_lambda_away_remaining": float(np.mean(lam_a)),
    }


# ==============================
# メイン検証
# ==============================

def run_validation(cfg: Config) -> None:
    cfg.outdir.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(cfg.results_path)
    elo_history = pd.read_csv(cfg.elo_history_path)
    season_start = pd.read_csv(cfg.season_start_path)

    ensure_columns(results, ["year", "date", "division", "home", "away", "home_goal", "away_goal"], "results")
    ensure_columns(elo_history, ["year", "date", "home", "away", "home_rating_post", "away_rating_post"], "elo_history")
    ensure_columns(season_start, ["year", "team", "rating_start"], "season_start")

    results["date"] = pd.to_datetime(results["date"])
    elo_history["date"] = pd.to_datetime(elo_history["date"])

    # 予測対象は基本J1のみ。EloはJ1/J2統合履歴を使う。
    target = results.copy()
    if cfg.use_j1_only:
        target = target[target["division"] == "J1"].copy()

    if cfg.eval_end_year is None:
        cfg.eval_end_year = int(target["year"].max())

    target = target[(target["year"] >= cfg.eval_start_year) & (target["year"] <= cfg.eval_end_year)].copy()

    detail_rows = []

    for cutoff_matches in cfg.cutoff_matches_list:
        for beta in cfg.betas:
            for year in sorted(target["year"].unique()):
                season_matches = target[target["year"] == year].copy().sort_values(["date", "match_id" if "match_id" in target.columns else "home"])
                teams = sorted(set(season_matches["home"]).union(season_matches["away"]))

                if len(teams) == 0:
                    continue

                cutoff_date = find_cutoff_date(season_matches, teams, cutoff_matches)
                if cutoff_date is None:
                    print(f"[SKIP] {year}: 全チーム{cutoff_matches}試合到達日がありません")
                    continue

                known_matches = season_matches[season_matches["date"] <= cutoff_date].copy()
                remaining_matches = season_matches[season_matches["date"] > cutoff_date].copy()

                if len(remaining_matches) == 0:
                    print(f"[SKIP] {year}: cutoff={cutoff_matches} 以降の残り試合がありません")
                    continue

                season_start_date = season_matches["date"].min()
                elos = get_cutoff_elos(
                    elo_history=elo_history,
                    season_start=season_start,
                    target_year=int(year),
                    season_start_date=season_start_date,
                    cutoff_date=cutoff_date,
                    teams=teams,
                )

                # beta・year・cutoffごとに乱数を変えつつ再現可能にする
                local_seed = int(cfg.seed + year * 100000 + cutoff_matches * 1000 + round(beta * 1000))
                rng = np.random.default_rng(local_seed)

                metrics = simulate_one_season(
                    season_matches=season_matches,
                    known_matches=known_matches,
                    remaining_matches=remaining_matches,
                    teams=teams,
                    elos=elos,
                    beta_elo=float(beta),
                    cfg=cfg,
                    rng=rng,
                )

                detail_rows.append({
                    "cutoff_matches": cutoff_matches,
                    "beta_elo": beta,
                    "year": int(year),
                    "cutoff_date": cutoff_date.date().isoformat(),
                    "teams": len(teams),
                    "known_matches": len(known_matches),
                    "remaining_matches": len(remaining_matches),
                    **metrics,
                })

                print(
                    f"year={year} cutoff={cutoff_matches} beta={beta:.3f} "
                    f"MAE={metrics['mae']:.3f} "
                    f"prob={metrics['prob_actual_position']:.3f} "
                    f"draw_sim={metrics['sim_draw_rate']:.3f}"
                )

    detail = pd.DataFrame(detail_rows)
    if detail.empty:
        raise RuntimeError("検証結果が空です。eval期間やcutoff_matchesを確認してください。")

    summary = (
        detail
        .groupby(["cutoff_matches", "beta_elo"], as_index=False)
        .agg(
            seasons=("year", "nunique"),
            mean_mae=("mae", "mean"),
            std_mae=("mae", "std"),
            mean_prob_actual_position=("prob_actual_position", "mean"),
            mean_sim_draw_rate=("sim_draw_rate", "mean"),
            mean_actual_remaining_draw_rate=("actual_remaining_draw_rate", "mean"),
            mean_sim_goals_per_match=("sim_goals_per_match", "mean"),
            mean_actual_remaining_goals_per_match=("actual_remaining_goals_per_match", "mean"),
            mean_lambda_home_remaining=("mean_lambda_home_remaining", "mean"),
            mean_lambda_away_remaining=("mean_lambda_away_remaining", "mean"),
        )
        .sort_values(["cutoff_matches", "mean_mae", "beta_elo"])
    )

    detail_path = cfg.outdir / "elo_poisson_beta_validation_detail.csv"
    summary_path = cfg.outdir / "elo_poisson_beta_validation_summary.csv"
    config_path = cfg.outdir / "elo_poisson_beta_validation_config.csv"

    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    pd.DataFrame([{
        "results_path": str(cfg.results_path),
        "elo_history_path": str(cfg.elo_history_path),
        "season_start_path": str(cfg.season_start_path),
        "eval_start_year": cfg.eval_start_year,
        "eval_end_year": cfg.eval_end_year,
        "cutoff_matches_list": ",".join(map(str, cfg.cutoff_matches_list)),
        "betas": ",".join(map(str, cfg.betas)),
        "n_sims": cfg.n_sims,
        "seed": cfg.seed,
        "prior_matches": cfg.prior_matches,
        "lambda_cap": cfg.lambda_cap,
        "min_lambda": cfg.min_lambda,
        "strength_low": cfg.strength_low,
        "strength_high": cfg.strength_high,
        "use_j1_only": cfg.use_j1_only,
    }]).to_csv(config_path, index=False, encoding="utf-8-sig")

    print("\n==============================")
    print("=== Elo補正ポアソン検証まとめ ===")
    print("==============================")
    print(summary.to_string(index=False))
    print("\nSaved:")
    print(f"- {summary_path}")
    print(f"- {detail_path}")
    print(f"- {config_path}")


# ==============================
# CLI
# ==============================

def main() -> None:
    parser = argparse.ArgumentParser(description="J1/J2統合Eloをポアソンλ補正として使う検証コード")
    parser.add_argument("--results", default="j1_j2_elo_input_1993_2025.csv", help="J1/J2統合試合結果CSV")
    parser.add_argument("--elo-history", default="elo_outputs/j1_j2_elo_match_history.csv", help="Elo試合履歴CSV")
    parser.add_argument("--season-start", default="elo_outputs/j1_j2_elo_season_start_ratings.csv", help="シーズン開始時Elo CSV")
    parser.add_argument("--outdir", default="elo_poisson_validation_outputs", help="出力フォルダ")
    parser.add_argument("--eval-start-year", type=int, default=2012, help="検証開始年。古い特殊ルールを避けるなら2012以降推奨")
    parser.add_argument("--eval-end-year", type=int, default=None, help="検証終了年。省略時はデータ内の最終年")
    parser.add_argument("--cutoff-matches-list", default="20", help="全チームが何試合以上終えた時点で予測するか。例: 5,10,15,20")
    parser.add_argument("--betas", default="0,0.05,0.10,0.15,0.20,0.30", help="検証する beta_elo。カンマ区切り")
    parser.add_argument("--n-sims", type=int, default=1000, help="各season/betaのシミュレーション回数")
    parser.add_argument("--seed", type=int, default=42, help="乱数seed")
    parser.add_argument("--prior-matches", type=float, default=3.0, help="攻守係数の平均回帰の強さ。3ならリーグ平均3試合分を足す")
    parser.add_argument("--lambda-cap", type=float, default=3.5, help="λ上限")
    parser.add_argument("--min-lambda", type=float, default=0.05, help="λ下限")
    parser.add_argument("--strength-low", type=float, default=0.35, help="攻守係数の下限")
    parser.add_argument("--strength-high", type=float, default=2.50, help="攻守係数の上限")
    parser.add_argument("--include-j2-target", action="store_true", help="指定すると予測対象にもJ2を含める。通常は使わない")

    args = parser.parse_args()

    cfg = Config(
        results_path=Path(args.results),
        elo_history_path=Path(args.elo_history),
        season_start_path=Path(args.season_start),
        outdir=Path(args.outdir),
        eval_start_year=args.eval_start_year,
        eval_end_year=args.eval_end_year,
        cutoff_matches_list=parse_int_list(args.cutoff_matches_list),
        betas=parse_float_list(args.betas),
        n_sims=args.n_sims,
        seed=args.seed,
        prior_matches=args.prior_matches,
        lambda_cap=args.lambda_cap,
        min_lambda=args.min_lambda,
        strength_low=args.strength_low,
        strength_high=args.strength_high,
        use_j1_only=not args.include_j2_target,
    )

    run_validation(cfg)


if __name__ == "__main__":
    main()
