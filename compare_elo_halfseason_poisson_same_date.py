#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
J1/J2統合Eloを、前半戦→後半戦ポアソン分布シミュレーションで比較検証するコード。

特徴:
- 前半/後半の分割で、同じ日付の試合を前後に分断しない。
- beta_elo=0.00 を Eloなし基準として、beta_elo>0 と横並び比較する。
- ポアソンλの基本形は、前半戦実績 + 過年度prior を用いたホーム/アウェイ別攻守係数。
- Eloは勝敗を直接決めず、λの倍率補正だけに使う。

基本式:
    lambda_home_base = league_home_avg * home_home_attack * away_away_defense
    lambda_away_base = league_away_avg * away_away_attack * home_home_defense

    elo_factor = exp(beta_elo * (home_elo - away_elo) / 400)
    lambda_home = lambda_home_base * elo_factor
    lambda_away = lambda_away_base / elo_factor

デフォルト設定:
    DECAY = 1.0
    PREV_DECAY = 0.995
    PREV_WEIGHT = 0.20
    GOAL_CAP_FOR_STRENGTH = 4
    LAMBDA_CAP = 3.5

想定入力:
    j1_j2_elo_input_1993_2025.csv
    elo_outputs/j1_j2_elo_match_history.csv
    elo_outputs/j1_j2_elo_season_start_ratings.csv

実行例:
    python compare_elo_halfseason_poisson_same_date.py \
      --input j1_j2_elo_input_1993_2025.csv \
      --elo-history elo_outputs/j1_j2_elo_match_history.csv \
      --season-start elo_outputs/j1_j2_elo_season_start_ratings.csv \
      --years 2023,2024,2025 \
      --n-sims 10000 \
      --betas 0,0.05,0.10,0.15,0.20,0.30
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
    elo_history_csv: str = "elo_outputs/j1_j2_elo_match_history.csv"
    season_start_csv: str = "elo_outputs/j1_j2_elo_season_start_ratings.csv"
    outdir: str = "elo_halfseason_same_date_outputs"

    target_division: str = "J1"
    years: str = "2023,2024,2025"
    betas: str = "0,0.05,0.10,0.15,0.20,0.30"
    n_sims: int = 10000
    seed: int = 42

    # Eloなし基準側の設定
    decay: float = 1.0
    prev_decay: float = 0.995
    prev_weight: float = 0.20
    goal_cap_for_strength: float = 4.0
    lambda_cap: float = 3.5
    min_lambda: float = 0.01

    # priorに使う過去データ
    # all: J1/J2両方を使用。昇格組のJ2成績もpriorに入る。
    # same: target_divisionのみ使用。
    prior_division_mode: str = "all"

    # 攻守係数クリップ
    strength_low: float = 0.35
    strength_high: float = 2.50

    # 同日を切らない分割の基準
    # nearest: 試合数半分に最も近い日付を境界にする
    # after: 半分を初めて超えた日付までを前半にする
    # before: 半分を超えない最後の日付までを前半にする
    same_date_policy: str = "nearest"


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Elo比較・同日非分割・前半戦→後半戦ポアソン検証")
    p.add_argument("--input", dest="input_csv", default=Config.input_csv)
    p.add_argument("--elo-history", dest="elo_history_csv", default=Config.elo_history_csv)
    p.add_argument("--season-start", dest="season_start_csv", default=Config.season_start_csv)
    p.add_argument("--outdir", default=Config.outdir)

    p.add_argument("--target-division", default=Config.target_division)
    p.add_argument("--years", default=Config.years, help="例: 2023,2024,2025 または 2012-2025")
    p.add_argument("--betas", default=Config.betas, help="例: 0,0.05,0.10,0.15,0.20")
    p.add_argument("--n-sims", type=int, default=Config.n_sims)
    p.add_argument("--seed", type=int, default=Config.seed)

    p.add_argument("--decay", type=float, default=Config.decay)
    p.add_argument("--prev-decay", type=float, default=Config.prev_decay)
    p.add_argument("--prev-weight", type=float, default=Config.prev_weight)
    p.add_argument("--goal-cap", dest="goal_cap_for_strength", type=float, default=Config.goal_cap_for_strength)
    p.add_argument("--lambda-cap", type=float, default=Config.lambda_cap)
    p.add_argument("--min-lambda", type=float, default=Config.min_lambda)
    p.add_argument("--prior-division-mode", choices=["all", "same"], default=Config.prior_division_mode)
    p.add_argument("--strength-low", type=float, default=Config.strength_low)
    p.add_argument("--strength-high", type=float, default=Config.strength_high)
    p.add_argument("--same-date-policy", choices=["nearest", "after", "before"], default=Config.same_date_policy)
    ns = p.parse_args()
    return Config(**vars(ns))


def parse_years(text: str) -> List[int]:
    text = text.strip()
    if "-" in text and "," not in text:
        a, b = text.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def require_columns(df: pd.DataFrame, cols: Iterable[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} に必要列がありません: {missing}")


# =========================
# 読み込み
# =========================

def load_matches(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    require_columns(df, ["year", "date", "division", "home", "away", "home_goal", "away_goal"], "matches")
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
        df["match_id"] = [f"M_{i+1:07d}" for i in range(len(df))]
    return df.sort_values(["date", "match_id"]).reset_index(drop=True)


def load_elo_history(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    require_columns(df, ["year", "date", "home", "away", "home_rating_post", "away_rating_post"], "elo_history")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home", "away", "home_rating_post", "away_rating_post"])
    if "match_id" not in df.columns:
        df["match_id"] = [f"ELO_{i+1:07d}" for i in range(len(df))]
    return df.sort_values(["date", "match_id"]).reset_index(drop=True)


def load_season_start(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    require_columns(df, ["year", "team", "rating_start"], "season_start")
    df = df.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["rating_start"] = pd.to_numeric(df["rating_start"], errors="coerce")
    df = df.dropna(subset=["year", "team", "rating_start"])
    df["year"] = df["year"].astype(int)
    return df


# =========================
# 同日非分割
# =========================

def split_half_same_date(season: pd.DataFrame, policy: str = "nearest") -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """
    シーズン試合を日付単位で前半/後半に分ける。
    同じ日付の試合は必ず同じ側に入る。
    """
    season = season.sort_values(["date", "match_id"]).reset_index(drop=True)
    n = len(season)
    if n < 2:
        raise ValueError("試合数が少なすぎて前後半に分割できません。")

    target = n / 2.0
    by_date = season.groupby("date", sort=True).size().reset_index(name="n")
    by_date["cum"] = by_date["n"].cumsum()

    if policy == "after":
        idx = int(by_date.index[by_date["cum"] >= target][0])
    elif policy == "before":
        cand = by_date.index[by_date["cum"] <= target]
        idx = int(cand[-1]) if len(cand) else 0
    else:  # nearest
        # 半分に一番近い累積試合数の日付を採用。完全同点なら、情報量を少し多めにするため後ろ側を採用。
        diffs = (by_date["cum"] - target).abs().to_numpy()
        min_diff = diffs.min()
        candidates = np.where(diffs == min_diff)[0]
        idx = int(candidates[-1])

    cutoff_date = pd.Timestamp(by_date.loc[idx, "date"])
    known = season[season["date"] <= cutoff_date].copy().reset_index(drop=True)
    future = season[season["date"] > cutoff_date].copy().reset_index(drop=True)

    meta = {
        "split_policy": policy,
        "target_half_matches": target,
        "cutoff_date": cutoff_date.date().isoformat(),
        "known_matches": len(known),
        "future_matches": len(future),
        "known_ratio": len(known) / n,
        "future_ratio": len(future) / n,
        "same_date_is_split": False,
    }
    return known, future, meta


# =========================
# 順位表
# =========================

def init_table(teams: Iterable[str]) -> Dict[str, Dict[str, int]]:
    return {
        t: {"P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "GD": 0, "Pts": 0}
        for t in teams
    }


def add_match(table: Dict[str, Dict[str, int]], home: str, away: str, hg: int, ag: int) -> None:
    table[home]["P"] += 1
    table[away]["P"] += 1
    table[home]["GF"] += hg
    table[home]["GA"] += ag
    table[away]["GF"] += ag
    table[away]["GA"] += hg
    table[home]["GD"] = table[home]["GF"] - table[home]["GA"]
    table[away]["GD"] = table[away]["GF"] - table[away]["GA"]
    if hg > ag:
        table[home]["W"] += 1
        table[away]["L"] += 1
        table[home]["Pts"] += 3
    elif hg < ag:
        table[away]["W"] += 1
        table[home]["L"] += 1
        table[away]["Pts"] += 3
    else:
        table[home]["D"] += 1
        table[away]["D"] += 1
        table[home]["Pts"] += 1
        table[away]["Pts"] += 1


def build_table_dict(matches: pd.DataFrame, teams: List[str]) -> Dict[str, Dict[str, int]]:
    table = init_table(teams)
    for r in matches.itertuples(index=False):
        add_match(table, str(r.home), str(r.away), int(r.home_goal), int(r.away_goal))
    return table


def positions_from_table(table: Dict[str, Dict[str, int]]) -> Dict[str, int]:
    order = sorted(
        table.keys(),
        key=lambda t: (-table[t]["Pts"], -table[t]["GD"], -table[t]["GF"], -table[t]["W"], t),
    )
    return {t: i + 1 for i, t in enumerate(order)}


# =========================
# 攻守係数
# =========================

def weighted_stats(matches: pd.DataFrame, teams: List[str], goal_cap: float, decay: float) -> Tuple[pd.DataFrame, Dict[str, float]]:
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
        return pd.DataFrame(stats.values()), {"home_avg": 1.0, "away_avg": 1.0}

    ordered = matches.sort_values(["date", "match_id"]).reset_index(drop=True)
    n = len(ordered)
    for idx, r in enumerate(ordered.itertuples(index=False)):
        # 古い試合ほどageが大きい。decay=1なら全試合同重み。
        age = n - 1 - idx
        w = decay ** age
        hg = min(float(r.home_goal), goal_cap)
        ag = min(float(r.away_goal), goal_cap)

        h = str(r.home)
        a = str(r.away)
        if h in stats:
            s = stats[h]
            s["home_games"] += w
            s["home_gf"] += w * hg
            s["home_ga"] += w * ag
        if a in stats:
            s = stats[a]
            s["away_games"] += w
            s["away_gf"] += w * ag
            s["away_ga"] += w * hg

    total_home_games = sum(x["home_games"] for x in stats.values())
    total_away_games = sum(x["away_games"] for x in stats.values())
    total_home_gf = sum(x["home_gf"] for x in stats.values())
    total_away_gf = sum(x["away_gf"] for x in stats.values())
    league = {
        "home_avg": total_home_gf / total_home_games if total_home_games > 0 else 1.0,
        "away_avg": total_away_gf / total_away_games if total_away_games > 0 else 1.0,
    }
    return pd.DataFrame(stats.values()), league


def strengths_from_stats(stats: pd.DataFrame, league: Dict[str, float]) -> pd.DataFrame:
    home_avg = max(float(league["home_avg"]), 1e-9)
    away_avg = max(float(league["away_avg"]), 1e-9)
    rows = []
    for r in stats.itertuples(index=False):
        home_gf_pg = r.home_gf / r.home_games if r.home_games > 0 else home_avg
        home_ga_pg = r.home_ga / r.home_games if r.home_games > 0 else away_avg
        away_gf_pg = r.away_gf / r.away_games if r.away_games > 0 else away_avg
        away_ga_pg = r.away_ga / r.away_games if r.away_games > 0 else home_avg
        rows.append({
            "team": r.team,
            "home_attack": home_gf_pg / home_avg,
            "home_defense": home_ga_pg / away_avg,  # 1より大きいほど失点しやすい
            "away_attack": away_gf_pg / away_avg,
            "away_defense": away_ga_pg / home_avg,  # 1より大きいほど失点しやすい
        })
    return pd.DataFrame(rows)


def estimate_strengths(current_matches: pd.DataFrame, prior_matches: pd.DataFrame, teams: List[str], cfg: Config) -> Tuple[pd.DataFrame, Dict[str, float]]:
    cur_stats, cur_league = weighted_stats(current_matches, teams, cfg.goal_cap_for_strength, cfg.decay)
    prev_stats, _prev_league = weighted_stats(prior_matches, teams, cfg.goal_cap_for_strength, cfg.prev_decay)
    cur_strength = strengths_from_stats(cur_stats, cur_league)
    prev_strength = strengths_from_stats(prev_stats, _prev_league)

    merged = cur_strength.merge(prev_strength, on="team", suffixes=("_cur", "_prev"), how="left")
    rows = []
    for r in merged.itertuples(index=False):
        row = {"team": r.team}
        for key in ["home_attack", "home_defense", "away_attack", "away_defense"]:
            cur = float(getattr(r, f"{key}_cur"))
            prev = getattr(r, f"{key}_prev")
            if pd.isna(prev):
                prev = 1.0
            val = (1.0 - cfg.prev_weight) * cur + cfg.prev_weight * float(prev)
            row[key] = float(np.clip(val, cfg.strength_low, cfg.strength_high))
        rows.append(row)

    base_avgs = {
        "league_home_avg": float(max(cur_league["home_avg"], 0.2)),
        "league_away_avg": float(max(cur_league["away_avg"], 0.2)),
    }
    return pd.DataFrame(rows), base_avgs


# =========================
# Elo取得
# =========================

def get_cutoff_elos(
    elo_history: pd.DataFrame,
    season_start: pd.DataFrame,
    year: int,
    season_start_date: pd.Timestamp,
    cutoff_date: pd.Timestamp,
    teams: List[str],
) -> Dict[str, float]:
    """
    対象年の年初回帰後Eloを初期値にし、cutoff_dateまでの実試合で更新されたEloを返す。
    """
    elos: Dict[str, float] = {}

    ss = season_start[season_start["year"] == year]
    for r in ss.itertuples(index=False):
        elos[str(r.team)] = float(r.rating_start)

    hist = elo_history[(elo_history["date"] >= season_start_date) & (elo_history["date"] <= cutoff_date)].copy()
    hist = hist.sort_values(["date", "match_id"])
    for r in hist.itertuples(index=False):
        elos[str(r.home)] = float(r.home_rating_post)
        elos[str(r.away)] = float(r.away_rating_post)

    return {t: float(elos.get(t, 1500.0)) for t in teams}


# =========================
# λ作成
# =========================

def make_lambdas(
    future_matches: pd.DataFrame,
    strengths: pd.DataFrame,
    base_avgs: Dict[str, float],
    elos: Dict[str, float],
    beta_elo: float,
    cfg: Config,
) -> pd.DataFrame:
    st = strengths.set_index("team").to_dict("index")
    rows = []
    for r in future_matches.itertuples(index=False):
        h = str(r.home)
        a = str(r.away)
        hs = st[h]
        aw = st[a]

        lam_h_base = base_avgs["league_home_avg"] * hs["home_attack"] * aw["away_defense"]
        lam_a_base = base_avgs["league_away_avg"] * aw["away_attack"] * hs["home_defense"]

        elo_diff = float(elos.get(h, 1500.0) - elos.get(a, 1500.0))
        elo_factor = math.exp(float(beta_elo) * elo_diff / 400.0)

        lam_h = lam_h_base * elo_factor
        lam_a = lam_a_base / elo_factor

        lam_h = float(np.clip(lam_h, cfg.min_lambda, cfg.lambda_cap))
        lam_a = float(np.clip(lam_a, cfg.min_lambda, cfg.lambda_cap))

        rows.append({
            "match_id": r.match_id,
            "year": int(r.year),
            "date": r.date,
            "beta_elo": float(beta_elo),
            "home": h,
            "away": a,
            "actual_home_goal": int(r.home_goal),
            "actual_away_goal": int(r.away_goal),
            "home_elo": float(elos.get(h, 1500.0)),
            "away_elo": float(elos.get(a, 1500.0)),
            "elo_diff": elo_diff,
            "elo_factor": float(elo_factor),
            "lambda_home_base": float(lam_h_base),
            "lambda_away_base": float(lam_a_base),
            "lambda_home": lam_h,
            "lambda_away": lam_a,
        })
    return pd.DataFrame(rows)


# =========================
# シミュレーション
# =========================

def simulate_with_lambdas(
    season: pd.DataFrame,
    known: pd.DataFrame,
    future: pd.DataFrame,
    teams: List[str],
    lambdas: pd.DataFrame,
    beta_elo: float,
    cfg: Config,
    rng: np.random.Generator,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
    actual_table = build_table_dict(season, teams)
    actual_pos = positions_from_table(actual_table)

    known_table = build_table_dict(known, teams)

    team_to_idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)

    base_pts = np.array([known_table[t]["Pts"] for t in teams], dtype=np.int16)
    base_gf = np.array([known_table[t]["GF"] for t in teams], dtype=np.int16)
    base_ga = np.array([known_table[t]["GA"] for t in teams], dtype=np.int16)
    base_w = np.array([known_table[t]["W"] for t in teams], dtype=np.int16)
    actual_pos_arr = np.array([actual_pos[t] for t in teams], dtype=np.int16)

    home_idx = np.array([team_to_idx[x] for x in lambdas["home"]], dtype=np.int16)
    away_idx = np.array([team_to_idx[x] for x in lambdas["away"]], dtype=np.int16)
    lam_h_arr = lambdas["lambda_home"].to_numpy(dtype=float)
    lam_a_arr = lambdas["lambda_away"].to_numpy(dtype=float)

    pos_sum = np.zeros(n_teams, dtype=np.float64)
    pos_count = np.zeros((n_teams, n_teams), dtype=np.int32)
    sim_draws = 0
    sim_goals = 0

    team_names_arr = np.array(teams)

    for _ in range(cfg.n_sims):
        pts = base_pts.copy()
        gf = base_gf.copy()
        ga = base_ga.copy()
        wins = base_w.copy()

        hg_arr = rng.poisson(lam_h_arr)
        ag_arr = rng.poisson(lam_a_arr)
        sim_draws += int(np.sum(hg_arr == ag_arr))
        sim_goals += int(np.sum(hg_arr + ag_arr))

        for i in range(len(lambdas)):
            hi = home_idx[i]
            ai = away_idx[i]
            hg = int(hg_arr[i])
            ag = int(ag_arr[i])
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
        order = np.lexsort((team_names_arr, -wins, -gf, -gd, -pts))
        pos = np.empty(n_teams, dtype=np.int16)
        pos[order] = np.arange(1, n_teams + 1)
        pos_sum += pos
        pos_count[np.arange(n_teams), pos - 1] += 1

    detail_rows = []
    for team, idx in team_to_idx.items():
        probs = pos_count[idx] / cfg.n_sims
        act = int(actual_pos[team])
        detail_rows.append({
            "year": int(season["year"].iloc[0]),
            "beta_elo": float(beta_elo),
            "team": team,
            "actual_position": act,
            "avg_pred_position": float(pos_sum[idx] / cfg.n_sims),
            "abs_error": float(abs(pos_sum[idx] / cfg.n_sims - act)),
            "prob_actual_position": float(probs[act - 1]),
            "prob_top1": float(probs[0]),
            "prob_top3": float(probs[:3].sum()),
            "prob_bottom3": float(probs[-3:].sum()),
        })
    team_detail = pd.DataFrame(detail_rows).sort_values(["year", "beta_elo", "avg_pred_position", "team"])

    dist_rows = []
    for team, idx in team_to_idx.items():
        for p in range(1, n_teams + 1):
            dist_rows.append({
                "year": int(season["year"].iloc[0]),
                "beta_elo": float(beta_elo),
                "team": team,
                "position": p,
                "probability": float(pos_count[idx, p - 1] / cfg.n_sims),
            })
    distribution = pd.DataFrame(dist_rows)

    mae = float(team_detail["abs_error"].mean())
    prob_actual = float(team_detail["prob_actual_position"].mean())

    actual_draw_rate = float((future["home_goal"] == future["away_goal"]).mean()) if len(future) else float("nan")
    actual_gpm = float((future["home_goal"] + future["away_goal"]).mean()) if len(future) else float("nan")
    sim_total = cfg.n_sims * len(future)
    metrics = {
        "mae": mae,
        "mean_prob_actual_position": prob_actual,
        "sim_draw_rate": float(sim_draws / sim_total) if sim_total else float("nan"),
        "actual_second_half_draw_rate": actual_draw_rate,
        "sim_goals_per_match": float(sim_goals / sim_total) if sim_total else float("nan"),
        "actual_second_half_goals_per_match": actual_gpm,
        "mean_lambda_home": float(lambdas["lambda_home"].mean()),
        "mean_lambda_away": float(lambdas["lambda_away"].mean()),
        "mean_lambda_total": float((lambdas["lambda_home"] + lambdas["lambda_away"]).mean()),
        "mean_elo_factor": float(lambdas["elo_factor"].mean()),
        "mean_abs_elo_diff": float(lambdas["elo_diff"].abs().mean()),
    }
    return metrics, team_detail, distribution


# =========================
# 年度処理
# =========================

def run_one_year(
    year: int,
    all_matches: pd.DataFrame,
    elo_history: pd.DataFrame,
    season_start: pd.DataFrame,
    betas: List[float],
    cfg: Config,
) -> Tuple[List[Dict[str, object]], List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]]:
    season = all_matches[(all_matches["year"] == year) & (all_matches["division"] == cfg.target_division)].copy()
    season = season.sort_values(["date", "match_id"]).reset_index(drop=True)
    if len(season) == 0:
        raise ValueError(f"{year}年 {cfg.target_division} の試合がありません。")

    teams = sorted(set(season["home"]) | set(season["away"]))
    known, future, split_meta = split_half_same_date(season, cfg.same_date_policy)
    if len(known) == 0 or len(future) == 0:
        raise ValueError(f"{year}年の分割結果が不正です: known={len(known)}, future={len(future)}")

    prior = all_matches[all_matches["year"] < year].copy()
    if cfg.prior_division_mode == "same":
        prior = prior[prior["division"] == cfg.target_division].copy()
    prior = prior[(prior["home"].isin(teams)) | (prior["away"].isin(teams))]
    prior = prior.sort_values(["date", "match_id"]).reset_index(drop=True)

    strengths, base_avgs = estimate_strengths(known, prior, teams, cfg)

    cutoff_date = pd.Timestamp(split_meta["cutoff_date"])
    elos = get_cutoff_elos(
        elo_history=elo_history,
        season_start=season_start,
        year=year,
        season_start_date=season["date"].min(),
        cutoff_date=cutoff_date,
        teams=teams,
    )

    detail_rows = []
    team_dfs = []
    dist_dfs = []
    lambda_dfs = []

    for beta in betas:
        lambdas = make_lambdas(future, strengths, base_avgs, elos, beta, cfg)
        # betaごと・年度ごとに再現可能な乱数。beta=0の値も固定される。
        local_seed = int(cfg.seed + year * 100000 + round(beta * 10000))
        rng = np.random.default_rng(local_seed)
        metrics, team_detail, distribution = simulate_with_lambdas(season, known, future, teams, lambdas, beta, cfg, rng)

        detail_rows.append({
            "year": year,
            "beta_elo": float(beta),
            "teams": len(teams),
            "season_matches": len(season),
            "known_matches": len(known),
            "future_matches": len(future),
            "known_ratio": split_meta["known_ratio"],
            "cutoff_date": split_meta["cutoff_date"],
            "same_date_policy": cfg.same_date_policy,
            "league_home_avg_known": base_avgs["league_home_avg"],
            "league_away_avg_known": base_avgs["league_away_avg"],
            "prior_matches_used": len(prior),
            **metrics,
        })
        team_dfs.append(team_detail)
        dist_dfs.append(distribution)
        lambda_dfs.append(lambdas)

        print(
            f"{year} beta={beta:.3f}: "
            f"MAE={metrics['mae']:.4f}, "
            f"prob={metrics['mean_prob_actual_position']:.4f}, "
            f"draw={metrics['sim_draw_rate']:.4f}"
        )

    return detail_rows, team_dfs, dist_dfs, lambda_dfs


# =========================
# メイン
# =========================

def main() -> None:
    cfg = parse_args()
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_matches = load_matches(cfg.input_csv)
    elo_history = load_elo_history(cfg.elo_history_csv)
    season_start = load_season_start(cfg.season_start_csv)

    years = parse_years(cfg.years)
    betas = parse_floats(cfg.betas)

    all_detail_rows: List[Dict[str, object]] = []
    all_team_dfs: List[pd.DataFrame] = []
    all_dist_dfs: List[pd.DataFrame] = []
    all_lambda_dfs: List[pd.DataFrame] = []

    for year in years:
        rows, team_dfs, dist_dfs, lambda_dfs = run_one_year(year, all_matches, elo_history, season_start, betas, cfg)
        all_detail_rows.extend(rows)
        all_team_dfs.extend(team_dfs)
        all_dist_dfs.extend(dist_dfs)
        all_lambda_dfs.extend(lambda_dfs)

    detail = pd.DataFrame(all_detail_rows).sort_values(["beta_elo", "year"]).reset_index(drop=True)

    summary = (
        detail.groupby("beta_elo", as_index=False)
        .agg(
            n_years=("year", "count"),
            mean_mae=("mae", "mean"),
            std_mae=("mae", "std"),
            mean_prob_actual_position=("mean_prob_actual_position", "mean"),
            mean_sim_draw_rate=("sim_draw_rate", "mean"),
            mean_actual_second_half_draw_rate=("actual_second_half_draw_rate", "mean"),
            mean_sim_goals_per_match=("sim_goals_per_match", "mean"),
            mean_actual_second_half_goals_per_match=("actual_second_half_goals_per_match", "mean"),
            mean_lambda_total=("mean_lambda_total", "mean"),
            mean_abs_elo_diff=("mean_abs_elo_diff", "mean"),
        )
        .sort_values(["mean_mae", "beta_elo"])
        .reset_index(drop=True)
    )

    if all_team_dfs:
        team_detail = pd.concat(all_team_dfs, ignore_index=True)
    else:
        team_detail = pd.DataFrame()
    if all_dist_dfs:
        distribution = pd.concat(all_dist_dfs, ignore_index=True)
    else:
        distribution = pd.DataFrame()
    if all_lambda_dfs:
        lambdas = pd.concat(all_lambda_dfs, ignore_index=True)
    else:
        lambdas = pd.DataFrame()

    config_df = pd.DataFrame([asdict(cfg) | {"parsed_years": ",".join(map(str, years)), "parsed_betas": ",".join(map(str, betas))}])

    detail.to_csv(outdir / "elo_halfseason_same_date_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(outdir / "elo_halfseason_same_date_summary.csv", index=False, encoding="utf-8-sig")
    team_detail.to_csv(outdir / "elo_halfseason_same_date_team_detail.csv", index=False, encoding="utf-8-sig")
    distribution.to_csv(outdir / "elo_halfseason_same_date_position_distribution.csv", index=False, encoding="utf-8-sig")
    lambdas.to_csv(outdir / "elo_halfseason_same_date_lambdas.csv", index=False, encoding="utf-8-sig")
    config_df.to_csv(outdir / "elo_halfseason_same_date_config.csv", index=False, encoding="utf-8-sig")

    print("\n=== summary ===")
    print(summary.to_string(index=False))
    print(f"\n出力先: {outdir.resolve()}")


if __name__ == "__main__":
    main()
