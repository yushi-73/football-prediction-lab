"""
J1/J2/J3 Elo calculator

Usage:
    python calculate_jleague_elo.py --input j1_j2_j3_elo_input_1993_2025.csv

Outputs:
    elo_final_ratings.csv
    elo_history_by_match.csv
    elo_season_end_ratings.csv
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


# -----------------------------
# Default settings
# -----------------------------
DEFAULT_INIT_ELO = {
    "J1": 1600.0,
    "J2": 1500.0,
    "J3": 1400.0,
}

DEFAULT_K = {
    "J1": 20.0,
    "J2": 20.0,
    "J3": 20.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate J.League Elo ratings from match CSV.")
    parser.add_argument("--input", required=True, help="Input match CSV path")
    parser.add_argument("--output-dir", default=".", help="Output directory")

    parser.add_argument("--home-adv", type=float, default=50.0, help="Home advantage in Elo points")
    parser.add_argument("--scale", type=float, default=400.0, help="Elo logistic scale")
    parser.add_argument("--season-regression", type=float, default=0.85,
                        help="Season-to-season retention. 0.85 means 15%% regression toward regression-base")
    parser.add_argument("--regression-base", type=float, default=1500.0,
                        help="Base value for season regression")
    parser.add_argument("--no-season-regression", action="store_true",
                        help="Disable season-to-season regression")

    parser.add_argument("--init-j1", type=float, default=DEFAULT_INIT_ELO["J1"])
    parser.add_argument("--init-j2", type=float, default=DEFAULT_INIT_ELO["J2"])
    parser.add_argument("--init-j3", type=float, default=DEFAULT_INIT_ELO["J3"])

    parser.add_argument("--k-j1", type=float, default=DEFAULT_K["J1"])
    parser.add_argument("--k-j2", type=float, default=DEFAULT_K["J2"])
    parser.add_argument("--k-j3", type=float, default=DEFAULT_K["J3"])

    parser.add_argument("--use-official-result", action="store_true",
                        help="Use official_home_result if available. Default uses 90-minute score from home_goal/away_goal.")
    parser.add_argument("--no-margin", action="store_true",
                        help="Disable goal margin multiplier")

    parser.add_argument("--final-output", default="elo_final_ratings.csv")
    parser.add_argument("--history-output", default="elo_history_by_match.csv")
    parser.add_argument("--season-output", default="elo_season_end_ratings.csv")

    return parser.parse_args()


def read_matches(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = ["year", "date", "division", "home", "away", "home_goal", "away_goal"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        bad = df[df["date"].isna()].head(5)
        raise ValueError(f"Date parse error. Examples:\n{bad}")

    df["year"] = df["year"].astype(int)
    df["home_goal"] = df["home_goal"].astype(int)
    df["away_goal"] = df["away_goal"].astype(int)

    # Ensure stable order within the same date.
    sort_cols = ["date", "division", "match_id"] if "match_id" in df.columns else ["date", "division", "home", "away"]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def expected_score(rating_a: float, rating_b: float, scale: float = 400.0) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / scale))


def result_score(home_goal: int, away_goal: int) -> float:
    if home_goal > away_goal:
        return 1.0
    if home_goal < away_goal:
        return 0.0
    return 0.5


def official_result_score(row: pd.Series) -> float:
    """
    official_home_result: W/D/L を使う。
    ない場合は得点から判定する。
    """
    r = row.get("official_home_result", None)
    if isinstance(r, str):
        r = r.strip().upper()
        if r == "W":
            return 1.0
        if r == "L":
            return 0.0
        if r == "D":
            return 0.5
    return result_score(int(row["home_goal"]), int(row["away_goal"]))


def margin_multiplier(goal_diff: int, elo_diff_with_home_adv: float) -> float:
    """
    Margin-of-victory multiplier.
    Draws return 1.0.

    Formula close to common football Elo implementations:
      ln(|GD| + 1) * 2.2 / (0.001 * |elo_diff| + 2.2)
    """
    gd = abs(goal_diff)
    if gd <= 0:
        return 1.0
    return math.log(gd + 1.0) * 2.2 / (0.001 * abs(elo_diff_with_home_adv) + 2.2)


def apply_season_regression(ratings: Dict[str, float], regression_base: float, retention: float) -> None:
    """
    Regress all existing teams toward a fixed base at season boundary.
    retention=0.85 means rating keeps 85% of its gap from the base.
    """
    for team in list(ratings.keys()):
        ratings[team] = regression_base + retention * (ratings[team] - regression_base)


def calculate_elo(
    matches: pd.DataFrame,
    init_elo: Dict[str, float],
    k_values: Dict[str, float],
    home_adv: float,
    scale: float,
    season_regression: float,
    regression_base: float,
    use_season_regression: bool,
    use_official_result: bool,
    use_margin: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ratings: Dict[str, float] = {}
    team_matches = defaultdict(int)
    team_last_division: Dict[str, str] = {}
    team_last_match_date: Dict[str, pd.Timestamp] = {}

    history_rows = []
    season_rows = []

    current_year = None

    for _, row in matches.iterrows():
        year = int(row["year"])
        division = str(row["division"])
        home = str(row["home"])
        away = str(row["away"])
        home_goal = int(row["home_goal"])
        away_goal = int(row["away_goal"])

        # At the first match of a new season, save previous season-end ratings and regress.
        if current_year is None:
            current_year = year
        elif year != current_year:
            for team, elo in ratings.items():
                season_rows.append({
                    "year": current_year,
                    "team": team,
                    "elo": elo,
                    "last_division": team_last_division.get(team),
                    "matches": team_matches[team],
                    "last_match_date": team_last_match_date.get(team),
                })

            if use_season_regression:
                apply_season_regression(ratings, regression_base, season_regression)

            current_year = year

        # Initialize new teams by the division they first appear in.
        base = init_elo.get(division, regression_base)
        if home not in ratings:
            ratings[home] = base
        if away not in ratings:
            ratings[away] = base

        home_pre = ratings[home]
        away_pre = ratings[away]

        elo_diff_with_home_adv = (home_pre + home_adv) - away_pre
        exp_home = expected_score(home_pre + home_adv, away_pre, scale=scale)

        actual_home = official_result_score(row) if use_official_result else result_score(home_goal, away_goal)

        k = k_values.get(division, 20.0)
        gd = home_goal - away_goal
        mov = margin_multiplier(gd, elo_diff_with_home_adv) if use_margin else 1.0

        delta = k * mov * (actual_home - exp_home)

        ratings[home] = home_pre + delta
        ratings[away] = away_pre - delta

        team_matches[home] += 1
        team_matches[away] += 1
        team_last_division[home] = division
        team_last_division[away] = division
        team_last_match_date[home] = row["date"]
        team_last_match_date[away] = row["date"]

        history_rows.append({
            "match_id": row.get("match_id", None),
            "year": year,
            "date": row["date"],
            "division": division,
            "home": home,
            "away": away,
            "home_goal": home_goal,
            "away_goal": away_goal,
            "home_elo_pre": home_pre,
            "away_elo_pre": away_pre,
            "home_adv": home_adv,
            "expected_home": exp_home,
            "actual_home": actual_home,
            "goal_diff": gd,
            "margin_multiplier": mov,
            "k": k,
            "elo_delta_home": delta,
            "home_elo_post": ratings[home],
            "away_elo_post": ratings[away],
        })

    # Save the final season-end ratings.
    if current_year is not None:
        for team, elo in ratings.items():
            season_rows.append({
                "year": current_year,
                "team": team,
                "elo": elo,
                "last_division": team_last_division.get(team),
                "matches": team_matches[team],
                "last_match_date": team_last_match_date.get(team),
            })

    final_rows = []
    for team, elo in ratings.items():
        final_rows.append({
            "team": team,
            "elo": elo,
            "last_division": team_last_division.get(team),
            "matches": team_matches[team],
            "last_match_date": team_last_match_date.get(team),
        })

    history = pd.DataFrame(history_rows)
    season_end = pd.DataFrame(season_rows)
    final = pd.DataFrame(final_rows).sort_values("elo", ascending=False).reset_index(drop=True)
    final.insert(0, "rank", final.index + 1)

    return final, history, season_end


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    init_elo = {
        "J1": args.init_j1,
        "J2": args.init_j2,
        "J3": args.init_j3,
    }
    k_values = {
        "J1": args.k_j1,
        "J2": args.k_j2,
        "J3": args.k_j3,
    }

    matches = read_matches(input_path)

    final, history, season_end = calculate_elo(
        matches=matches,
        init_elo=init_elo,
        k_values=k_values,
        home_adv=args.home_adv,
        scale=args.scale,
        season_regression=args.season_regression,
        regression_base=args.regression_base,
        use_season_regression=not args.no_season_regression,
        use_official_result=args.use_official_result,
        use_margin=not args.no_margin,
    )

    final_path = output_dir / args.final_output
    history_path = output_dir / args.history_output
    season_path = output_dir / args.season_output

    final.to_csv(final_path, index=False, encoding="utf-8-sig")
    history.to_csv(history_path, index=False, encoding="utf-8-sig")
    season_end.to_csv(season_path, index=False, encoding="utf-8-sig")

    print("Done.")
    print(f"Input matches: {len(matches):,}")
    print(f"Teams: {final['team'].nunique():,}")
    print(f"Final ratings: {final_path}")
    print(f"Match history: {history_path}")
    print(f"Season end ratings: {season_path}")
    print()
    print("Top 20 final ratings:")
    print(final.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
