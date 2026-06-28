#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eloなし・前半戦→後半戦 ポアソン分布リーグシミュレーション

想定入力:
  j1_j2_elo_input_1993_2025.csv
  または同等の列を持つCSV

必須列:
  year, date, division, home, away, home_goal, away_goal

デフォルト設定:
  DECAY = 1.0
  PREV_DECAY = 0.995
  PREV_WEIGHT = 0.20
  GOAL_CAP_FOR_STRENGTH = 4
  LAMBDA_CAP = 3.5

目的:
  指定リーグ・指定年について、シーズンを時系列で前半/後半に分割し、
  前半戦の実績 + 過年度priorから攻撃/守備係数を作り、後半戦をポアソン分布でシミュレーションする。

例:
  python simulate_no_elo_halfseason_poisson.py --years 2023,2024,2025 --n-sims 10000
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


# =========================
# 設定
# =========================

@dataclass
class Config:
    input_csv: str = "j1_j2_elo_input_1993_2025.csv"
    outdir: str = "no_elo_halfseason_poisson_outputs"
    target_division: str = "J1"
    years: str = "2023,2024,2025"
    n_sims: int = 10000
    seed: int = 42

    # 検証した設定
    decay: float = 1.0
    prev_decay: float = 0.995
    prev_weight: float = 0.20
    goal_cap_for_strength: float = 4.0
    lambda_cap: float = 3.5

    # priorに使う過去データ
    # all: J1/J2両方を使用。昇格組のJ2成績もpriorに入る。
    # same: target_divisionのみ使用。
    prior_division_mode: str = "all"

    # 攻守係数が極端になりすぎるのを防ぐクリップ
    strength_low: float = 0.35
    strength_high: float = 2.50

    # 前半戦の切り方
    # chronological_half: 対象年・対象divisionの試合を日付順に並べて前半半分をknownにする。
    split_mode: str = "chronological_half"


# =========================
# 読み込み・前処理
# =========================

def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Eloなし・前半戦→後半戦 ポアソン分布シミュレーション")
    p.add_argument("--input", dest="input_csv", default=Config.input_csv)
    p.add_argument("--outdir", default=Config.outdir)
    p.add_argument("--target-division", default=Config.target_division)
    p.add_argument("--years", default=Config.years, help="例: 2023,2024,2025")
    p.add_argument("--n-sims", type=int, default=Config.n_sims)
    p.add_argument("--seed", type=int, default=Config.seed)

    p.add_argument("--decay", type=float, default=Config.decay)
    p.add_argument("--prev-decay", type=float, default=Config.prev_decay)
    p.add_argument("--prev-weight", type=float, default=Config.prev_weight)
    p.add_argument("--goal-cap", dest="goal_cap_for_strength", type=float, default=Config.goal_cap_for_strength)
    p.add_argument("--lambda-cap", type=float, default=Config.lambda_cap)

    p.add_argument("--prior-division-mode", choices=["all", "same"], default=Config.prior_division_mode)
    p.add_argument("--strength-low", type=float, default=Config.strength_low)
    p.add_argument("--strength-high", type=float, default=Config.strength_high)
    p.add_argument("--split-mode", choices=["chronological_half"], default=Config.split_mode)

    ns = p.parse_args()
    return Config(**vars(ns))


def load_matches(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"year", "date", "division", "home", "away", "home_goal", "away_goal"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"入力CSVに必要列がありません: {sorted(missing)}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["home_goal"] = pd.to_numeric(df["home_goal"], errors="coerce")
    df["away_goal"] = pd.to_numeric(df["away_goal"], errors="coerce")

    df = df.dropna(subset=["year", "date", "division", "home", "away", "home_goal", "away_goal"])
    df["year"] = df["year"].astype(int)
    df["home_goal"] = df["home_goal"].astype(int)
    df["away_goal"] = df["away_goal"].astype(int)

    if "match_id" not in df.columns:
        df["match_id"] = [f"M_{i+1:06d}" for i in range(len(df))]

    df = df.sort_values(["date", "match_id"]).reset_index(drop=True)
    return df


def parse_years(years: str) -> List[int]:
    years = years.strip()
    if "-" in years and "," not in years:
        a, b = years.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x.strip()) for x in years.split(",") if x.strip()]


# =========================
# 順位表
# =========================

def init_table(teams: Iterable[str]) -> Dict[str, Dict[str, float]]:
    return {
        t: {"team": t, "P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "GD": 0, "Pts": 0}
        for t in teams
    }


def apply_result(table: Dict[str, Dict[str, float]], home: str, away: str, hg: int, ag: int) -> None:
    h = table[home]
    a = table[away]
    h["P"] += 1
    a["P"] += 1
    h["GF"] += hg
    h["GA"] += ag
    a["GF"] += ag
    a["GA"] += hg
    h["GD"] = h["GF"] - h["GA"]
    a["GD"] = a["GF"] - a["GA"]

    if hg > ag:
        h["W"] += 1
        a["L"] += 1
        h["Pts"] += 3
    elif hg < ag:
        a["W"] += 1
        h["L"] += 1
        a["Pts"] += 3
    else:
        h["D"] += 1
        a["D"] += 1
        h["Pts"] += 1
        a["Pts"] += 1


def build_table(matches: pd.DataFrame, teams: Iterable[str]) -> pd.DataFrame:
    table = init_table(teams)
    for r in matches.itertuples(index=False):
        apply_result(table, r.home, r.away, int(r.home_goal), int(r.away_goal))
    out = pd.DataFrame(table.values())
    # Jリーグの完全な順位決定は直接対戦等もあるが、ここでは通常の主要指標で統一する。
    out = out.sort_values(["Pts", "GD", "GF", "W", "team"], ascending=[False, False, False, False, True])
    out["position"] = np.arange(1, len(out) + 1)
    return out.reset_index(drop=True)


# =========================
# 攻撃・守備係数
# =========================

def _weighted_stats(
    matches: pd.DataFrame,
    teams: List[str],
    goal_cap: float,
    decay: float,
    most_recent_first: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """ホーム/アウェイ別の加重得点・失点・試合数を作る。"""
    stats = {
        t: {
            "team": t,
            "home_games": 0.0,
            "away_games": 0.0,
            "home_gf": 0.0,
            "home_ga": 0.0,
            "away_gf": 0.0,
            "away_ga": 0.0,
        }
        for t in teams
    }

    if len(matches) == 0:
        league = {"home_avg": 1.0, "away_avg": 1.0}
        return pd.DataFrame(stats.values()), league

    ordered = matches.sort_values(["date", "match_id"]).reset_index(drop=True)
    n = len(ordered)

    # current dataでは古い方がage大、recentがage 0。
    # decay=1なら全試合同じ重み。
    for idx, r in enumerate(ordered.itertuples(index=False)):
        if most_recent_first:
            # すでに新しい順に入れる場合用。今回は使わないが残しておく。
            age = idx
        else:
            age = n - 1 - idx
        w = decay ** age

        hg = min(float(r.home_goal), goal_cap)
        ag = min(float(r.away_goal), goal_cap)

        if r.home in stats:
            s = stats[r.home]
            s["home_games"] += w
            s["home_gf"] += w * hg
            s["home_ga"] += w * ag
        if r.away in stats:
            s = stats[r.away]
            s["away_games"] += w
            s["away_gf"] += w * ag
            s["away_ga"] += w * hg

    total_home_games = sum(s["home_games"] for s in stats.values())
    total_away_games = sum(s["away_games"] for s in stats.values())
    total_home_gf = sum(s["home_gf"] for s in stats.values())
    total_away_gf = sum(s["away_gf"] for s in stats.values())

    league = {
        "home_avg": total_home_gf / total_home_games if total_home_games > 0 else 1.0,
        "away_avg": total_away_gf / total_away_games if total_away_games > 0 else 1.0,
    }
    return pd.DataFrame(stats.values()), league


def _strength_from_stats(stats: pd.DataFrame, league: Dict[str, float]) -> pd.DataFrame:
    rows = []
    home_avg = max(league["home_avg"], 1e-9)
    away_avg = max(league["away_avg"], 1e-9)

    for r in stats.itertuples(index=False):
        # データがない場合はリーグ平均=係数1に戻す。
        home_gf_pg = r.home_gf / r.home_games if r.home_games > 0 else home_avg
        home_ga_pg = r.home_ga / r.home_games if r.home_games > 0 else away_avg
        away_gf_pg = r.away_gf / r.away_games if r.away_games > 0 else away_avg
        away_ga_pg = r.away_ga / r.away_games if r.away_games > 0 else home_avg

        rows.append(
            {
                "team": r.team,
                "home_attack": home_gf_pg / home_avg,
                "home_defense": home_ga_pg / away_avg,
                "away_attack": away_gf_pg / away_avg,
                "away_defense": away_ga_pg / home_avg,
            }
        )
    return pd.DataFrame(rows)


def estimate_strengths(
    current_matches: pd.DataFrame,
    prior_matches: pd.DataFrame,
    teams: List[str],
    cfg: Config,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """前半戦実績と過年度priorを混ぜて攻守係数を返す。"""
    cur_stats, cur_league = _weighted_stats(
        current_matches, teams, cfg.goal_cap_for_strength, cfg.decay
    )
    cur_strength = _strength_from_stats(cur_stats, cur_league)

    prev_stats, prev_league = _weighted_stats(
        prior_matches, teams, cfg.goal_cap_for_strength, cfg.prev_decay
    )
    prev_strength = _strength_from_stats(prev_stats, prev_league)

    merged = cur_strength.merge(prev_strength, on="team", suffixes=("_cur", "_prev"), how="left")

    out_rows = []
    for r in merged.itertuples(index=False):
        row = {"team": r.team}
        for key in ["home_attack", "home_defense", "away_attack", "away_defense"]:
            cur = getattr(r, f"{key}_cur")
            prev = getattr(r, f"{key}_prev")
            if pd.isna(prev):
                prev = 1.0
            val = (1.0 - cfg.prev_weight) * cur + cfg.prev_weight * prev
            val = float(np.clip(val, cfg.strength_low, cfg.strength_high))
            row[key] = val
        out_rows.append(row)

    strengths = pd.DataFrame(out_rows)

    # λの土台となるリーグ得点環境は、予測対象年の前半戦から取る。
    # これにより、当年の得点環境を反映する。
    base_avgs = {
        "league_home_avg": float(cur_league["home_avg"]),
        "league_away_avg": float(cur_league["away_avg"]),
    }
    return strengths, base_avgs


def make_lambdas(
    future_matches: pd.DataFrame,
    strengths: pd.DataFrame,
    base_avgs: Dict[str, float],
    cfg: Config,
) -> pd.DataFrame:
    st = strengths.set_index("team").to_dict("index")
    rows = []
    for r in future_matches.itertuples(index=False):
        hs = st[r.home]
        aw = st[r.away]
        lam_h = base_avgs["league_home_avg"] * hs["home_attack"] * aw["away_defense"]
        lam_a = base_avgs["league_away_avg"] * aw["away_attack"] * hs["home_defense"]
        lam_h = float(np.clip(lam_h, 0.01, cfg.lambda_cap))
        lam_a = float(np.clip(lam_a, 0.01, cfg.lambda_cap))
        rows.append(
            {
                "match_id": r.match_id,
                "year": int(r.year),
                "date": r.date,
                "home": r.home,
                "away": r.away,
                "actual_home_goal": int(r.home_goal),
                "actual_away_goal": int(r.away_goal),
                "lambda_home": lam_h,
                "lambda_away": lam_a,
            }
        )
    return pd.DataFrame(rows)


# =========================
# シミュレーション
# =========================

def simulate_season(
    year: int,
    all_matches: pd.DataFrame,
    cfg: Config,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    season = all_matches[(all_matches["year"] == year) & (all_matches["division"] == cfg.target_division)].copy()
    season = season.sort_values(["date", "match_id"]).reset_index(drop=True)
    if len(season) == 0:
        raise ValueError(f"{year}年 {cfg.target_division} の試合がありません。")

    teams = sorted(set(season["home"]) | set(season["away"]))
    n_matches = len(season)
    split_idx = n_matches // 2
    known = season.iloc[:split_idx].copy().reset_index(drop=True)
    future = season.iloc[split_idx:].copy().reset_index(drop=True)

    # priorは対象年より前のデータ。デフォルトではJ1/J2両方を使う。
    prior = all_matches[all_matches["year"] < year].copy()
    if cfg.prior_division_mode == "same":
        prior = prior[prior["division"] == cfg.target_division].copy()
    # 対象年に参加しているチームに関係する試合だけで十分。
    prior = prior[(prior["home"].isin(teams)) | (prior["away"].isin(teams))]
    prior = prior.sort_values(["date", "match_id"]).reset_index(drop=True)

    strengths, base_avgs = estimate_strengths(known, prior, teams, cfg)
    lambdas = make_lambdas(future, strengths, base_avgs, cfg)

    known_table = build_table(known, teams)
    actual_table = build_table(season, teams)
    actual_pos = actual_table.set_index("team")["position"].to_dict()

    # シミュレーション用の初期値
    team_to_idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)
    pts0 = np.zeros(n_teams, dtype=np.int16)
    gf0 = np.zeros(n_teams, dtype=np.int16)
    ga0 = np.zeros(n_teams, dtype=np.int16)
    w0 = np.zeros(n_teams, dtype=np.int16)

    for r in known.itertuples(index=False):
        hi = team_to_idx[r.home]
        ai = team_to_idx[r.away]
        hg = int(r.home_goal)
        ag = int(r.away_goal)
        gf0[hi] += hg
        ga0[hi] += ag
        gf0[ai] += ag
        ga0[ai] += hg
        if hg > ag:
            pts0[hi] += 3
            w0[hi] += 1
        elif hg < ag:
            pts0[ai] += 3
            w0[ai] += 1
        else:
            pts0[hi] += 1
            pts0[ai] += 1

    positions_count = np.zeros((n_teams, n_teams), dtype=np.int32)
    positions_sum = np.zeros(n_teams, dtype=np.float64)
    total_draws = 0
    total_goals = 0

    # future arrays
    home_idx = np.array([team_to_idx[x] for x in lambdas["home"]], dtype=np.int16)
    away_idx = np.array([team_to_idx[x] for x in lambdas["away"]], dtype=np.int16)
    lam_h_arr = lambdas["lambda_home"].to_numpy(dtype=float)
    lam_a_arr = lambdas["lambda_away"].to_numpy(dtype=float)

    for sim in range(cfg.n_sims):
        pts = pts0.astype(np.int16).copy()
        gf = gf0.astype(np.int16).copy()
        ga = ga0.astype(np.int16).copy()
        wins = w0.astype(np.int16).copy()

        h_goals = rng.poisson(lam_h_arr)
        a_goals = rng.poisson(lam_a_arr)
        total_draws += int(np.sum(h_goals == a_goals))
        total_goals += int(np.sum(h_goals + a_goals))

        # 試合結果反映
        for i in range(len(lambdas)):
            hi = home_idx[i]
            ai = away_idx[i]
            hg = int(h_goals[i])
            ag = int(a_goals[i])
            gf[hi] += hg
            ga[hi] += ag
            gf[ai] += ag
            ga[ai] += hg
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
        # sort: Pts, GD, GF, W descending, team name ascending
        # np.lexsortは最後のキーが最優先。昇順なのでマイナスを使う。
        order = np.lexsort((np.array(teams), -wins, -gf, -gd, -pts))
        pos = np.empty(n_teams, dtype=np.int16)
        pos[order] = np.arange(1, n_teams + 1)
        positions_sum += pos
        positions_count[np.arange(n_teams), pos - 1] += 1

    # team detail
    rows = []
    for t, idx in team_to_idx.items():
        probs = positions_count[idx] / cfg.n_sims
        act = int(actual_pos[t])
        rows.append(
            {
                "year": year,
                "team": t,
                "actual_position": act,
                "avg_pred_position": positions_sum[idx] / cfg.n_sims,
                "prob_actual_position": probs[act - 1],
                "prob_top1": probs[0],
                "prob_top3": probs[:3].sum(),
                "prob_bottom3": probs[-3:].sum(),
            }
        )
    detail = pd.DataFrame(rows).sort_values(["year", "avg_pred_position", "team"]).reset_index(drop=True)

    # distribution long
    dist_rows = []
    for t, idx in team_to_idx.items():
        for p in range(1, n_teams + 1):
            dist_rows.append({"year": year, "team": t, "position": p, "probability": positions_count[idx, p - 1] / cfg.n_sims})
    distribution = pd.DataFrame(dist_rows)

    mae = float(np.mean(np.abs(detail["avg_pred_position"] - detail["actual_position"])))
    prob_actual = float(detail["prob_actual_position"].mean())

    actual_future_draw_rate = float(np.mean(future["home_goal"].to_numpy() == future["away_goal"].to_numpy())) if len(future) else np.nan
    actual_future_gpm = float(np.mean(future["home_goal"].to_numpy() + future["away_goal"].to_numpy())) if len(future) else np.nan
    sim_draw_rate = total_draws / (cfg.n_sims * len(future)) if len(future) else np.nan
    sim_gpm = total_goals / (cfg.n_sims * len(future)) if len(future) else np.nan

    summary = {
        "year": year,
        "division": cfg.target_division,
        "n_teams": n_teams,
        "total_matches": n_matches,
        "known_matches_first_half": len(known),
        "simulated_matches_second_half": len(future),
        "first_match_date": season["date"].min().date().isoformat(),
        "split_date_last_known": known["date"].max().date().isoformat(),
        "split_date_first_future": future["date"].min().date().isoformat() if len(future) else None,
        "league_home_avg_for_lambda": base_avgs["league_home_avg"],
        "league_away_avg_for_lambda": base_avgs["league_away_avg"],
        "mae": mae,
        "mean_prob_actual_position": prob_actual,
        "sim_draw_rate": sim_draw_rate,
        "actual_second_half_draw_rate": actual_future_draw_rate,
        "sim_goals_per_match": sim_gpm,
        "actual_second_half_goals_per_match": actual_future_gpm,
    }

    lambdas_out = lambdas.copy()
    lambdas_out["split"] = "second_half"
    return detail, distribution, lambdas_out, summary


# =========================
# main
# =========================

def main() -> None:
    cfg = parse_args()
    input_path = Path(cfg.input_csv)
    if not input_path.exists():
        alt = Path.cwd() / cfg.input_csv
        if alt.exists():
            input_path = alt
        else:
            raise FileNotFoundError(f"入力CSVが見つかりません: {cfg.input_csv}")

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    matches = load_matches(input_path)
    years = parse_years(cfg.years)
    rng = np.random.default_rng(cfg.seed)

    all_detail = []
    all_dist = []
    all_lambdas = []
    summaries = []

    print("=== Eloなし・前半戦→後半戦 Poisson simulation ===")
    print(f"input: {input_path}")
    print(f"target_division: {cfg.target_division}")
    print(f"years: {years}")
    print(f"n_sims: {cfg.n_sims}")
    print(f"DECAY={cfg.decay}, PREV_DECAY={cfg.prev_decay}, PREV_WEIGHT={cfg.prev_weight}")
    print(f"GOAL_CAP={cfg.goal_cap_for_strength}, LAMBDA_CAP={cfg.lambda_cap}")

    for y in years:
        print(f"\n--- {y} {cfg.target_division} ---")
        detail, dist, lambdas, summary = simulate_season(y, matches, cfg, rng)
        all_detail.append(detail)
        all_dist.append(dist)
        all_lambdas.append(lambdas)
        summaries.append(summary)
        print(
            f"MAE={summary['mae']:.4f}, "
            f"P(actual pos)={summary['mean_prob_actual_position']:.4f}, "
            f"draw={summary['sim_draw_rate']:.3f}/{summary['actual_second_half_draw_rate']:.3f}, "
            f"goals={summary['sim_goals_per_match']:.3f}/{summary['actual_second_half_goals_per_match']:.3f}"
        )

    summary_df = pd.DataFrame(summaries).sort_values("year")
    detail_df = pd.concat(all_detail, ignore_index=True) if all_detail else pd.DataFrame()
    dist_df = pd.concat(all_dist, ignore_index=True) if all_dist else pd.DataFrame()
    lambdas_df = pd.concat(all_lambdas, ignore_index=True) if all_lambdas else pd.DataFrame()

    # 全年平均行を追加
    if len(summary_df):
        avg_row = {
            "year": "ALL_MEAN",
            "division": cfg.target_division,
            "n_teams": summary_df["n_teams"].mean(),
            "total_matches": summary_df["total_matches"].mean(),
            "known_matches_first_half": summary_df["known_matches_first_half"].mean(),
            "simulated_matches_second_half": summary_df["simulated_matches_second_half"].mean(),
            "first_match_date": "",
            "split_date_last_known": "",
            "split_date_first_future": "",
            "league_home_avg_for_lambda": summary_df["league_home_avg_for_lambda"].mean(),
            "league_away_avg_for_lambda": summary_df["league_away_avg_for_lambda"].mean(),
            "mae": summary_df["mae"].mean(),
            "mean_prob_actual_position": summary_df["mean_prob_actual_position"].mean(),
            "sim_draw_rate": summary_df["sim_draw_rate"].mean(),
            "actual_second_half_draw_rate": summary_df["actual_second_half_draw_rate"].mean(),
            "sim_goals_per_match": summary_df["sim_goals_per_match"].mean(),
            "actual_second_half_goals_per_match": summary_df["actual_second_half_goals_per_match"].mean(),
        }
        summary_df = pd.concat([summary_df, pd.DataFrame([avg_row])], ignore_index=True)

    summary_path = outdir / "no_elo_halfseason_poisson_summary.csv"
    detail_path = outdir / "no_elo_halfseason_poisson_team_detail.csv"
    dist_path = outdir / "no_elo_halfseason_poisson_position_distribution.csv"
    lambdas_path = outdir / "no_elo_halfseason_poisson_lambdas.csv"
    config_path = outdir / "no_elo_halfseason_poisson_config.csv"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    dist_df.to_csv(dist_path, index=False, encoding="utf-8-sig")
    lambdas_df.to_csv(lambdas_path, index=False, encoding="utf-8-sig")
    pd.DataFrame([asdict(cfg)]).to_csv(config_path, index=False, encoding="utf-8-sig")

    print("\n=== saved ===")
    print(summary_path)
    print(detail_path)
    print(dist_path)
    print(lambdas_path)
    print(config_path)

    if len(summary_df):
        last = summary_df.iloc[-1]
        if str(last["year"]) == "ALL_MEAN":
            print("\n=== ALL_MEAN ===")
            print(f"MAE={float(last['mae']):.4f}")
            print(f"mean_prob_actual_position={float(last['mean_prob_actual_position']):.4f}")
            print(f"sim_draw_rate={float(last['sim_draw_rate']):.4f}")
            print(f"actual_second_half_draw_rate={float(last['actual_second_half_draw_rate']):.4f}")
            print(f"sim_goals_per_match={float(last['sim_goals_per_match']):.4f}")
            print(f"actual_second_half_goals_per_match={float(last['actual_second_half_goals_per_match']):.4f}")


if __name__ == "__main__":
    main()
