# -*- coding: utf-8 -*-
"""
J3 2026/27 Elo-based season simulation

VSCodeの右上の再生ボタンだけで動く単体Pythonです。

同じフォルダに以下のCSVを置いてください。
必須:
  - j3_2026_27_initial_elo.csv

任意:
  - j3_2026_27_schedule.csv
    ない場合は、20チーム総当たりホーム&アウェイの仮想日程を自動生成します。

出力:
  - j3_elo_sim_output/j3_elo_sim_summary.csv
  - j3_elo_sim_output/j3_elo_sim_position_distribution.csv
  - j3_elo_sim_output/j3_elo_sim_match_lambdas.csv
  - j3_elo_sim_output/j3_elo_sim_schedule_used.csv
"""

from __future__ import annotations

from pathlib import Path
from difflib import get_close_matches
import itertools
import math

import numpy as np
import pandas as pd


# ============================================================
# ここだけ必要に応じて変更
# ============================================================

INITIAL_ELO_FILE_CANDIDATES = [
    "j3_2026_27_initial_elo.csv",
    "j3_2026_27_initial_elo(1).csv",
]

SCHEDULE_FILE_CANDIDATES = [
    "j3_2026_27_schedule.csv",
    "j3_2026_27_schedule(1).csv",
]

OUTPUT_DIR = "j3_elo_sim_output"

N_SIM = 10000
RANDOM_SEED = 42

# J3の1試合あたり平均得点の仮設定。
# 後で2025 J3実績や百年構想データから推定値に置き換えてもOKです。
BASE_HOME_GOALS = 1.34
BASE_AWAY_GOALS = 1.16

# Elo差を得点期待値へ変換する係数。
# 100 Elo差で exp(0.30)=約1.35倍になる設定です。
ELO_TO_GOAL_LOG_COEF = 0.0030

# ホームアドバンテージをElo差として加算。
HOME_ADV_ELO = 50.0

# λの暴走防止。
MIN_LAMBDA = 0.15
MAX_LAMBDA = 3.50

# 確率として見る順位範囲
AUTO_PROMOTION_RANK = 2      # 1〜2位
PLAYOFF_RANK = 6            # 1〜6位
RELEGATION_RANK_START = 19  # 19〜20位を下位2つとして集計


# ============================================================
# ユーティリティ
# ============================================================

def find_existing_file(candidates: list[str], required: bool) -> Path | None:
    """候補ファイル名から存在するものを探す。"""
    for name in candidates:
        path = Path(name)
        if path.exists():
            return path

    if required:
        print("必要なCSVが見つかりません。候補:")
        for name in candidates:
            print(f"  - {name}")
        raise FileNotFoundError("必要なCSVが見つかりません。")

    return None


def normalize_team_name(name: str) -> str:
    """チーム名比較用の軽い正規化。元の表記は出力に残す。"""
    if pd.isna(name):
        return ""

    s = str(name).strip()
    s = s.replace("　", " ")
    s = " ".join(s.split())

    # よく混ざる表記だけ軽く補正。
    # 必要に応じてここに追加してください。
    replacements = {
        "FC大阪": "ＦＣ大阪",
        "FC岐阜": "ＦＣ岐阜",
        "FC琉球": "ＦＣ琉球",
        "SC相模原": "ＳＣ相模原",
        "AC長野パルセイロ": "ＡＣ長野パルセイロ",
        "レイラック滋賀FC": "レイラック滋賀ＦＣ",
        "高知ユナイテッドＳＣ": "高知ユナイテッドSC",
    }
    return replacements.get(s, s)


def read_initial_elo(path: Path) -> pd.DataFrame:
    """J3参加20クラブの初期Eloを読む。"""
    df = pd.read_csv(path)

    lower_cols = {c.lower(): c for c in df.columns}

    if "team" not in lower_cols:
        raise ValueError("初期Elo CSVに team 列が必要です。")
    if "elo" not in lower_cols:
        raise ValueError("初期Elo CSVに elo 列が必要です。")

    team_col = lower_cols["team"]
    elo_col = lower_cols["elo"]

    out = df[[team_col, elo_col]].copy()
    out.columns = ["team", "elo"]
    out["team"] = out["team"].astype(str).str.strip()
    out["team_key"] = out["team"].map(normalize_team_name)
    out["elo"] = pd.to_numeric(out["elo"], errors="raise")

    if out["team_key"].duplicated().any():
        duplicated = out.loc[out["team_key"].duplicated(keep=False), "team"].tolist()
        raise ValueError(f"初期Elo CSVに重複チームがあります: {duplicated}")

    if len(out) != 20:
        print(f"WARNING: 初期Eloのチーム数が20ではありません: {len(out)}")

    return out.sort_values("elo", ascending=False).reset_index(drop=True)


def detect_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    """候補名から列を探す。"""
    normalized = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().strip()
        if key in normalized:
            return normalized[key]

    raise ValueError(
        f"{label} 列が見つかりません。候補: {candidates}\n"
        f"実際の列: {list(df.columns)}"
    )


def read_schedule(path: Path, elo_df: pd.DataFrame) -> pd.DataFrame:
    """日程CSVを読む。なければ後段で自動生成する。"""
    df = pd.read_csv(path)

    home_col = detect_column(
        df,
        ["home", "home_team", "ホーム", "ホームチーム", "Home"],
        "ホームチーム",
    )
    away_col = detect_column(
        df,
        ["away", "away_team", "アウェイ", "アウェイチーム", "Away"],
        "アウェイチーム",
    )

    out = pd.DataFrame()
    out["home"] = df[home_col].astype(str).str.strip()
    out["away"] = df[away_col].astype(str).str.strip()

    if "round" in df.columns:
        out["round"] = df["round"]
    elif "節" in df.columns:
        out["round"] = df["節"]
    else:
        out["round"] = np.arange(1, len(out) + 1)

    if "date" in df.columns:
        out["date"] = df["date"]
    elif "日付" in df.columns:
        out["date"] = df["日付"]
    else:
        out["date"] = ""

    out["home_key"] = out["home"].map(normalize_team_name)
    out["away_key"] = out["away"].map(normalize_team_name)

    validate_schedule_teams(out, elo_df)
    return out[["round", "date", "home", "away", "home_key", "away_key"]]


def make_double_round_robin_schedule(elo_df: pd.DataFrame) -> pd.DataFrame:
    """20チームのホーム&アウェイ総当たりを自動生成する。"""
    teams = elo_df["team"].tolist()
    rows = []
    match_no = 1

    for home, away in itertools.combinations(teams, 2):
        rows.append({"round": "AUTO", "date": "", "home": home, "away": away})
        match_no += 1
        rows.append({"round": "AUTO", "date": "", "home": away, "away": home})
        match_no += 1

    out = pd.DataFrame(rows)

    # 見やすさのためだけにシャッフル。静的Eloなので順位確率には影響しません。
    out = out.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    out["match_no"] = np.arange(1, len(out) + 1)
    out["round"] = out["match_no"]
    out = out.drop(columns=["match_no"])

    out["home_key"] = out["home"].map(normalize_team_name)
    out["away_key"] = out["away"].map(normalize_team_name)
    return out[["round", "date", "home", "away", "home_key", "away_key"]]


def validate_schedule_teams(schedule: pd.DataFrame, elo_df: pd.DataFrame) -> None:
    """日程に出てくるチームが初期Eloに存在するか確認する。"""
    elo_keys = set(elo_df["team_key"])
    schedule_keys = set(schedule["home_key"]) | set(schedule["away_key"])
    missing = sorted(schedule_keys - elo_keys)

    if not missing:
        return

    known = sorted(elo_keys)
    lines = ["日程CSVに、初期Elo CSVへ存在しないチームがあります。"]
    for team in missing:
        suggestion = get_close_matches(team, known, n=3, cutoff=0.45)
        if suggestion:
            lines.append(f"  - {team}  候補: {suggestion}")
        else:
            lines.append(f"  - {team}")

    raise ValueError("\n".join(lines))


def make_lambda_table(schedule: pd.DataFrame, elo_df: pd.DataFrame) -> pd.DataFrame:
    """各カードの期待得点λを計算する。"""
    elo_map = dict(zip(elo_df["team_key"], elo_df["elo"]))

    rows = []
    for _, r in schedule.iterrows():
        home_elo = float(elo_map[r["home_key"]])
        away_elo = float(elo_map[r["away_key"]])
        elo_diff = (home_elo + HOME_ADV_ELO) - away_elo

        lambda_home = BASE_HOME_GOALS * math.exp(ELO_TO_GOAL_LOG_COEF * elo_diff)
        lambda_away = BASE_AWAY_GOALS * math.exp(-ELO_TO_GOAL_LOG_COEF * elo_diff)

        lambda_home = min(max(lambda_home, MIN_LAMBDA), MAX_LAMBDA)
        lambda_away = min(max(lambda_away, MIN_LAMBDA), MAX_LAMBDA)

        rows.append(
            {
                "round": r["round"],
                "date": r["date"],
                "home": r["home"],
                "away": r["away"],
                "home_elo": home_elo,
                "away_elo": away_elo,
                "elo_diff_with_home_adv": elo_diff,
                "lambda_home": lambda_home,
                "lambda_away": lambda_away,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# シミュレーション本体
# ============================================================

def simulate_season(lambda_df: pd.DataFrame, teams: list[str], rng: np.random.Generator) -> pd.DataFrame:
    """1シーズン分をシミュレートして順位表を返す。"""
    stats = {
        team: {
            "team": team,
            "points": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "gf": 0,
            "ga": 0,
        }
        for team in teams
    }

    home_goals = rng.poisson(lambda_df["lambda_home"].to_numpy())
    away_goals = rng.poisson(lambda_df["lambda_away"].to_numpy())

    for i, r in lambda_df.reset_index(drop=True).iterrows():
        home = r["home"]
        away = r["away"]
        hg = int(home_goals[i])
        ag = int(away_goals[i])

        stats[home]["gf"] += hg
        stats[home]["ga"] += ag
        stats[away]["gf"] += ag
        stats[away]["ga"] += hg

        if hg > ag:
            stats[home]["points"] += 3
            stats[home]["wins"] += 1
            stats[away]["losses"] += 1
        elif hg < ag:
            stats[away]["points"] += 3
            stats[away]["wins"] += 1
            stats[home]["losses"] += 1
        else:
            stats[home]["points"] += 1
            stats[away]["points"] += 1
            stats[home]["draws"] += 1
            stats[away]["draws"] += 1

    table = pd.DataFrame(stats.values())
    table["gd"] = table["gf"] - table["ga"]

    # 完全同順位を避けるための微小乱数。順位決定の最後尾だけに使う。
    table["tie_break_noise"] = rng.random(len(table)) * 1e-9

    table = table.sort_values(
        ["points", "gd", "gf", "wins", "tie_break_noise"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    table["rank"] = np.arange(1, len(table) + 1)
    table = table.drop(columns=["tie_break_noise"])
    return table


def run_simulation(lambda_df: pd.DataFrame, elo_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """N_SIM回シミュレーションして、サマリーと順位分布を返す。"""
    rng = np.random.default_rng(RANDOM_SEED)
    teams = elo_df["team"].tolist()
    elo_map = dict(zip(elo_df["team"], elo_df["elo"]))

    accum = {
        team: {
            "rank_sum": 0.0,
            "points_sum": 0.0,
            "wins_sum": 0.0,
            "draws_sum": 0.0,
            "losses_sum": 0.0,
            "gf_sum": 0.0,
            "ga_sum": 0.0,
            "gd_sum": 0.0,
            "champion": 0,
            "top2": 0,
            "top6": 0,
            "playoff_3_6": 0,
            "bottom2": 0,
            "position_counts": np.zeros(len(teams), dtype=int),
        }
        for team in teams
    }

    for sim in range(1, N_SIM + 1):
        table = simulate_season(lambda_df, teams, rng)

        for _, row in table.iterrows():
            team = row["team"]
            rank = int(row["rank"])
            a = accum[team]

            a["rank_sum"] += rank
            a["points_sum"] += row["points"]
            a["wins_sum"] += row["wins"]
            a["draws_sum"] += row["draws"]
            a["losses_sum"] += row["losses"]
            a["gf_sum"] += row["gf"]
            a["ga_sum"] += row["ga"]
            a["gd_sum"] += row["gd"]

            a["position_counts"][rank - 1] += 1
            if rank == 1:
                a["champion"] += 1
            if rank <= AUTO_PROMOTION_RANK:
                a["top2"] += 1
            if rank <= PLAYOFF_RANK:
                a["top6"] += 1
            if 3 <= rank <= PLAYOFF_RANK:
                a["playoff_3_6"] += 1
            if rank >= RELEGATION_RANK_START:
                a["bottom2"] += 1

        if sim % max(1, N_SIM // 10) == 0:
            print(f"  simulation {sim:,}/{N_SIM:,} completed")

    summary_rows = []
    dist_rows = []

    for team in teams:
        a = accum[team]
        summary_rows.append(
            {
                "team": team,
                "initial_elo": elo_map[team],
                "avg_rank": a["rank_sum"] / N_SIM,
                "champion_prob": a["champion"] / N_SIM,
                "top2_prob": a["top2"] / N_SIM,
                "top6_prob": a["top6"] / N_SIM,
                "playoff_3_6_prob": a["playoff_3_6"] / N_SIM,
                "bottom2_prob": a["bottom2"] / N_SIM,
                "avg_points": a["points_sum"] / N_SIM,
                "avg_wins": a["wins_sum"] / N_SIM,
                "avg_draws": a["draws_sum"] / N_SIM,
                "avg_losses": a["losses_sum"] / N_SIM,
                "avg_gf": a["gf_sum"] / N_SIM,
                "avg_ga": a["ga_sum"] / N_SIM,
                "avg_gd": a["gd_sum"] / N_SIM,
            }
        )

        dist = {"team": team}
        for pos, count in enumerate(a["position_counts"], start=1):
            dist[f"rank_{pos}_prob"] = count / N_SIM
        dist_rows.append(dist)

    summary = pd.DataFrame(summary_rows).sort_values(
        ["avg_rank", "avg_points"], ascending=[True, False]
    ).reset_index(drop=True)
    summary.insert(0, "predicted_rank", np.arange(1, len(summary) + 1))

    position_distribution = pd.DataFrame(dist_rows)
    position_distribution = position_distribution.merge(
        summary[["team", "predicted_rank", "avg_rank"]], on="team", how="left"
    ).sort_values("predicted_rank").reset_index(drop=True)

    return summary, position_distribution


# ============================================================
# main
# ============================================================

def main() -> None:
    print("J3 2026/27 Elo-based season simulation")
    print("=" * 60)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    elo_path = find_existing_file(INITIAL_ELO_FILE_CANDIDATES, required=True)
    assert elo_path is not None
    print(f"Initial Elo file: {elo_path}")

    elo_df = read_initial_elo(elo_path)
    print(f"Teams: {len(elo_df)}")

    schedule_path = find_existing_file(SCHEDULE_FILE_CANDIDATES, required=False)
    if schedule_path is not None:
        print(f"Schedule file: {schedule_path}")
        schedule = read_schedule(schedule_path, elo_df)
    else:
        print("Schedule file not found.")
        print("20チームのホーム&アウェイ総当たり380試合を自動生成します。")
        schedule = make_double_round_robin_schedule(elo_df)

    print(f"Matches: {len(schedule)}")

    lambda_df = make_lambda_table(schedule, elo_df)

    schedule_out = output_dir / "j3_elo_sim_schedule_used.csv"
    lambda_out = output_dir / "j3_elo_sim_match_lambdas.csv"
    schedule.drop(columns=["home_key", "away_key"]).to_csv(schedule_out, index=False, encoding="utf-8-sig")
    lambda_df.to_csv(lambda_out, index=False, encoding="utf-8-sig")

    print("Running simulation...")
    summary, position_distribution = run_simulation(lambda_df, elo_df)

    summary_out = output_dir / "j3_elo_sim_summary.csv"
    dist_out = output_dir / "j3_elo_sim_position_distribution.csv"

    summary.to_csv(summary_out, index=False, encoding="utf-8-sig")
    position_distribution.to_csv(dist_out, index=False, encoding="utf-8-sig")

    print("\nDone.")
    print(f"Saved: {summary_out}")
    print(f"Saved: {dist_out}")
    print(f"Saved: {lambda_out}")
    print(f"Saved: {schedule_out}")

    print("\nTop 10 projection")
    show_cols = [
        "predicted_rank",
        "team",
        "initial_elo",
        "avg_rank",
        "champion_prob",
        "top2_prob",
        "top6_prob",
        "avg_points",
    ]
    print(summary[show_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
