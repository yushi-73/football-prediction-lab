# -*- coding: utf-8 -*-
"""
J1/J2 統合Eloレーティング作成スクリプト

Input:
  j1_j2_elo_input_1993_2025.csv

Outputs:
  elo_outputs/j1_j2_elo_match_history.csv
  elo_outputs/j1_j2_elo_season_start_ratings.csv
  elo_outputs/j1_j2_elo_season_end_ratings.csv
  elo_outputs/j1_j2_elo_latest_ratings.csv
  elo_outputs/j1_j2_elo_league_average_by_year.csv
  elo_outputs/j1_j2_elo_config_summary.csv

使い方例:
  python build_j1_j2_elo.py
  python build_j1_j2_elo.py --input j1_j2_elo_input_1993_2025.csv --outdir elo_outputs
  python build_j1_j2_elo.py --home-adv 60 --k-factor 20 --carryover 0.85 --result-mode normal
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, List, Any

import pandas as pd


REQUIRED_COLUMNS = [
    "match_id", "year", "date", "division", "home", "away",
    "home_goal", "away_goal", "has_pk", "pk_winner",
    "normal_home_result", "normal_away_result",
    "official_home_result", "official_away_result",
]


class SeasonStats:
    """1シーズン内のチーム成績をElo出力に添えるための簡易集計。"""
    def __init__(self) -> None:
        self.mp = 0
        self.w = 0
        self.d = 0
        self.l = 0
        self.gf = 0
        self.ga = 0

    def add(self, gf: int, ga: int) -> None:
        self.mp += 1
        self.gf += int(gf)
        self.ga += int(ga)
        if gf > ga:
            self.w += 1
        elif gf < ga:
            self.l += 1
        else:
            self.d += 1

    def to_dict(self) -> Dict[str, int]:
        return {
            "matches": self.mp,
            "wins": self.w,
            "draws": self.d,
            "losses": self.l,
            "goals_for": self.gf,
            "goals_against": self.ga,
            "goal_diff": self.gf - self.ga,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build integrated J1/J2 Elo ratings.")
    parser.add_argument(
        "--input",
        default="j1_j2_elo_input_1993_2025.csv",
        help="J1/J2統合済みのElo入力CSV",
    )
    parser.add_argument(
        "--outdir",
        default="elo_outputs",
        help="出力フォルダ",
    )
    parser.add_argument(
        "--base-rating",
        type=float,
        default=1500.0,
        help="基準点。平均的なJ1クラブの初期値として使う。",
    )
    parser.add_argument(
        "--initial-j1",
        type=float,
        default=1500.0,
        help="初登場がJ1のクラブの初期Elo。",
    )
    parser.add_argument(
        "--initial-j2",
        type=float,
        default=1450.0,
        help="初登場がJ2のクラブの初期Elo。J1との差を少し置く。",
    )
    parser.add_argument(
        "--k-factor",
        type=float,
        default=20.0,
        help="1試合あたりの更新幅。大きいほど直近結果に敏感。",
    )
    parser.add_argument(
        "--home-adv",
        type=float,
        default=60.0,
        help="ホームアドバンテージ。ホーム側の期待勝点計算時だけEloに加算する。",
    )
    parser.add_argument(
        "--carryover",
        type=float,
        default=0.85,
        help="年初に前年Eloをどれだけ残すか。0.85なら15%平均回帰。",
    )
    parser.add_argument(
        "--result-mode",
        choices=["normal", "official", "soft_pk"],
        default="normal",
        help=(
            "勝敗の扱い。normal=スコア上の勝敗/同点、official=公式勝敗、"
            "soft_pk=PK戦のみ0.55/0.45として軽く反映。"
        ),
    )
    parser.add_argument(
        "--use-mov",
        action="store_true",
        default=True,
        help="得失点差補正を使う。デフォルトで有効。",
    )
    parser.add_argument(
        "--no-mov",
        dest="use_mov",
        action="store_false",
        help="得失点差補正を使わない。",
    )
    parser.add_argument(
        "--mov-cap",
        type=float,
        default=2.5,
        help="得失点差補正の上限。大差試合の過大評価を防ぐ。",
    )
    parser.add_argument(
        "--round-rating",
        type=int,
        default=3,
        help="出力時の小数丸め桁数。",
    )
    return parser.parse_args()


def load_matches(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns are missing: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        bad = df[df["date"].isna()].head(10)
        raise ValueError(f"dateの変換に失敗した行があります。例:\n{bad}")

    for col in ["year", "home_goal", "away_goal"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            bad = df[df[col].isna()].head(10)
            raise ValueError(f"{col} に数値化できない行があります。例:\n{bad}")
        df[col] = df[col].astype(int)

    if df["home"].isna().any() or df["away"].isna().any():
        raise ValueError("home / away に欠損があります。")

    # 同日内は本来同時更新にするため、ここでの順序は表示・出力用だけ。
    # match_idがあるので安定した並びにしておく。
    sort_cols = ["date", "division", "match_id"]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def expected_score(home_rating: float, away_rating: float, home_adv: float) -> float:
    """ホーム側の期待スコア。勝ち=1, 分け=0.5, 負け=0 の期待値。"""
    return 1.0 / (1.0 + 10.0 ** ((away_rating - (home_rating + home_adv)) / 400.0))


def score_from_result_letter(result: str) -> float:
    if result == "W":
        return 1.0
    if result == "D":
        return 0.5
    if result == "L":
        return 0.0
    raise ValueError(f"Unknown result letter: {result}")


def actual_home_score(row: pd.Series, mode: str) -> float:
    """Elo更新に使うホーム側の実績値。"""
    hg = int(row["home_goal"])
    ag = int(row["away_goal"])

    if mode == "normal":
        if hg > ag:
            return 1.0
        if hg < ag:
            return 0.0
        return 0.5

    if mode == "official":
        return score_from_result_letter(str(row["official_home_result"]))

    if mode == "soft_pk":
        # 通常はスコアで判定。PK戦だけは勝者0.55/敗者0.45として軽く反映。
        if hg > ag:
            return 1.0
        if hg < ag:
            return 0.0
        has_pk = bool(row["has_pk"])
        if not has_pk:
            return 0.5
        winner = str(row.get("pk_winner", ""))
        if winner == "home":
            return 0.55
        if winner == "away":
            return 0.45
        return 0.5

    raise ValueError(f"Unknown result-mode: {mode}")


def mov_multiplier(
    home_rating: float,
    away_rating: float,
    home_goal: int,
    away_goal: int,
    use_mov: bool,
    cap: float,
) -> float:
    """得失点差補正。

    ・1点差勝利は概ね1.0
    ・大差勝利は少し強く評価
    ・ただし強豪が弱い相手を大差で倒しただけのときは補正を弱める
    ・番狂わせの大差勝利は補正がやや強くなる
    """
    if not use_mov:
        return 1.0

    gd = abs(int(home_goal) - int(away_goal))
    if gd == 0:
        return 1.0

    # 1点差で1.0、2点差で約1.585、3点差で2.0...
    base = math.log(gd + 1.0) / math.log(2.0)

    if home_goal > away_goal:
        winner_diff = home_rating - away_rating
    else:
        winner_diff = away_rating - home_rating

    # 強い側が勝ったときは少し抑え、弱い側が勝ったときは少し強める。
    # 極端な値にならないよう分母を下限で保護する。
    denom = max(0.6, 2.2 + 0.001 * winner_diff)
    corr = 2.2 / denom
    mult = base * corr
    return max(0.75, min(float(cap), mult))


def init_rating_for_division(division: str, args: argparse.Namespace) -> float:
    if division == "J1":
        return float(args.initial_j1)
    if division == "J2":
        return float(args.initial_j2)
    return float(args.base_rating)


def build_elo(df: pd.DataFrame, args: argparse.Namespace) -> Dict[str, pd.DataFrame]:
    ratings: Dict[str, float] = {}
    first_seen: Dict[str, int] = {}

    match_rows: List[Dict[str, Any]] = []
    season_start_rows: List[Dict[str, Any]] = []
    season_end_rows: List[Dict[str, Any]] = []
    league_avg_rows: List[Dict[str, Any]] = []

    current_year = None
    season_teams: set[str] = set()
    season_team_division: Dict[str, str] = {}
    season_stats: Dict[str, SeasonStats] = defaultdict(SeasonStats)

    def finish_season(year: int | None) -> None:
        if year is None:
            return
        for team in sorted(season_teams):
            row = {
                "year": year,
                "team": team,
                "division": season_team_division.get(team, ""),
                "rating_end": ratings[team],
            }
            row.update(season_stats[team].to_dict())
            season_end_rows.append(row)

        # リーグ別平均Elo。シーズン終了時点でそのリーグに所属したチームの平均。
        for div in ["J1", "J2"]:
            teams = [t for t in season_teams if season_team_division.get(t) == div]
            if teams:
                vals = [ratings[t] for t in teams]
                league_avg_rows.append({
                    "year": year,
                    "division": div,
                    "teams": len(teams),
                    "avg_rating_end": sum(vals) / len(vals),
                    "max_rating_end": max(vals),
                    "min_rating_end": min(vals),
                })

    def start_season(year: int, season_df: pd.DataFrame) -> Tuple[set[str], Dict[str, str], Dict[str, SeasonStats]]:
        # 既存クラブを平均へ回帰。長期の古すぎる影響を薄める。
        for team in list(ratings.keys()):
            ratings[team] = args.base_rating + args.carryover * (ratings[team] - args.base_rating)

        teams_in_year = set(season_df["home"]).union(set(season_df["away"]))

        # その年の所属divisionを作る。基本はその年に最初に出たdivision。
        div_map: Dict[str, str] = {}
        first_rows = season_df.sort_values(["date", "division", "match_id"])
        for _, r in first_rows.iterrows():
            div_map.setdefault(str(r["home"]), str(r["division"]))
            div_map.setdefault(str(r["away"]), str(r["division"]))

        # 新規登場クラブを初期化。
        for team in teams_in_year:
            if team not in ratings:
                div = div_map.get(team, "")
                ratings[team] = init_rating_for_division(div, args)
                first_seen[team] = year

        # シーズン開始時点のレートを保存。
        for team in sorted(teams_in_year):
            season_start_rows.append({
                "year": year,
                "team": team,
                "division": div_map.get(team, ""),
                "rating_start": ratings[team],
                "first_seen_year": first_seen.get(team, year),
                "is_new_team": first_seen.get(team, year) == year,
            })

        return teams_in_year, div_map, defaultdict(SeasonStats)

    # 年単位で処理し、同日試合は一括更新する。
    for year, year_df in df.groupby("year", sort=True):
        if current_year is not None:
            finish_season(current_year)

        current_year = int(year)
        season_teams, season_team_division, season_stats = start_season(current_year, year_df)

        for date_value, day_df in year_df.groupby("date", sort=True):
            # 同日の全試合は「その日の開始時点のレート」で期待値とdeltaを計算する。
            pending_deltas: Dict[str, float] = defaultdict(float)
            pending_rows: List[Dict[str, Any]] = []

            for _, row in day_df.iterrows():
                home = str(row["home"])
                away = str(row["away"])
                hg = int(row["home_goal"])
                ag = int(row["away_goal"])
                div = str(row["division"])

                # 万が一途中で新規チームが出た場合も保護する。
                if home not in ratings:
                    ratings[home] = init_rating_for_division(div, args)
                    first_seen[home] = current_year
                if away not in ratings:
                    ratings[away] = init_rating_for_division(div, args)
                    first_seen[away] = current_year

                home_pre = ratings[home]
                away_pre = ratings[away]
                exp_home = expected_score(home_pre, away_pre, args.home_adv)
                score_home = actual_home_score(row, args.result_mode)
                mult = mov_multiplier(home_pre, away_pre, hg, ag, args.use_mov, args.mov_cap)
                delta_home = args.k_factor * mult * (score_home - exp_home)

                pending_deltas[home] += delta_home
                pending_deltas[away] -= delta_home

                season_stats[home].add(hg, ag)
                season_stats[away].add(ag, hg)

                pending_rows.append({
                    "match_id": row["match_id"],
                    "year": current_year,
                    "date": pd.to_datetime(date_value).date().isoformat(),
                    "division": div,
                    "home": home,
                    "away": away,
                    "home_goal": hg,
                    "away_goal": ag,
                    "has_pk": bool(row["has_pk"]),
                    "pk_winner": row.get("pk_winner", ""),
                    "actual_home_score": score_home,
                    "home_rating_pre": home_pre,
                    "away_rating_pre": away_pre,
                    "home_adv_used": args.home_adv,
                    "expected_home_score": exp_home,
                    "expected_away_score": 1.0 - exp_home,
                    "mov_multiplier": mult,
                    "elo_delta_home": delta_home,
                })

            # ここで同日分をまとめて反映。
            for team, delta in pending_deltas.items():
                ratings[team] += delta

            # postレートを付けて保存。
            for mr in pending_rows:
                mr["home_rating_post"] = ratings[mr["home"]]
                mr["away_rating_post"] = ratings[mr["away"]]
                match_rows.append(mr)

    finish_season(current_year)

    latest_rows = []
    # 最新所属divisionは最後に出たdivisionを使う。
    last_division = {}
    last_year = {}
    for _, r in df.sort_values(["date", "match_id"]).iterrows():
        for side in ["home", "away"]:
            team = str(r[side])
            last_division[team] = str(r["division"])
            last_year[team] = int(r["year"])

    for team, rating in ratings.items():
        latest_rows.append({
            "team": team,
            "latest_rating": rating,
            "latest_division_in_data": last_division.get(team, ""),
            "latest_year_in_data": last_year.get(team, None),
            "first_seen_year": first_seen.get(team, None),
        })

    outputs = {
        "match_history": pd.DataFrame(match_rows),
        "season_start": pd.DataFrame(season_start_rows),
        "season_end": pd.DataFrame(season_end_rows),
        "latest": pd.DataFrame(latest_rows),
        "league_average": pd.DataFrame(league_avg_rows),
    }
    return outputs


def round_numeric(df: pd.DataFrame, digits: int) -> pd.DataFrame:
    out = df.copy()
    num_cols = out.select_dtypes(include=["number"]).columns
    out[num_cols] = out[num_cols].round(digits)
    return out


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_matches(input_path)
    outputs = build_elo(df, args)

    match_history = round_numeric(outputs["match_history"], args.round_rating)
    season_start = round_numeric(outputs["season_start"], args.round_rating)
    season_end = round_numeric(outputs["season_end"], args.round_rating)
    latest = round_numeric(outputs["latest"], args.round_rating)
    league_average = round_numeric(outputs["league_average"], args.round_rating)

    latest = latest.sort_values("latest_rating", ascending=False).reset_index(drop=True)
    latest.insert(0, "rank", latest.index + 1)

    season_start = season_start.sort_values(["year", "division", "rating_start"], ascending=[True, True, False])
    season_end = season_end.sort_values(["year", "division", "rating_end"], ascending=[True, True, False])

    match_history.to_csv(outdir / "j1_j2_elo_match_history.csv", index=False, encoding="utf-8-sig")
    season_start.to_csv(outdir / "j1_j2_elo_season_start_ratings.csv", index=False, encoding="utf-8-sig")
    season_end.to_csv(outdir / "j1_j2_elo_season_end_ratings.csv", index=False, encoding="utf-8-sig")
    latest.to_csv(outdir / "j1_j2_elo_latest_ratings.csv", index=False, encoding="utf-8-sig")
    league_average.to_csv(outdir / "j1_j2_elo_league_average_by_year.csv", index=False, encoding="utf-8-sig")

    config = pd.DataFrame([{
        "input": str(input_path),
        "matches": len(df),
        "year_min": int(df["year"].min()),
        "year_max": int(df["year"].max()),
        "base_rating": args.base_rating,
        "initial_j1": args.initial_j1,
        "initial_j2": args.initial_j2,
        "k_factor": args.k_factor,
        "home_adv": args.home_adv,
        "carryover": args.carryover,
        "result_mode": args.result_mode,
        "use_mov": args.use_mov,
        "mov_cap": args.mov_cap,
    }])
    config.to_csv(outdir / "j1_j2_elo_config_summary.csv", index=False, encoding="utf-8-sig")

    print("=== J1/J2 Integrated Elo completed ===")
    print(f"Input matches: {len(df):,}")
    print(f"Output dir: {outdir.resolve()}")
    print("\nTop 20 latest ratings:")
    print(latest.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
