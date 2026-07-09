import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

COMPETITION_ID = 1003  # SoccerDB: J3
BASE_URL = "https://soccer-db.net/competition/results/{competition_id}/{season}"

START_SEASON = 2014
END_SEASON = 2025

J3_OUTPUT = "j3_elo_input_2014_2025.csv"
MERGED_OUTPUT = "j1_j2_j3_elo_input_1993_2025.csv"

# 目安の試合数。取得漏れ・パース崩れ検出用。
# 制度変更や未消化試合がある場合は適宜変更してください。
EXPECTED_J3_MATCH_COUNTS = {
    2014: 198,
    2015: 234,
    2016: 240,
    2017: 272,
    2018: 272,
    2019: 306,
    2020: 306,
    2021: 210,
    2022: 306,
    2023: 380,
    2024: 380,
    2025: 380,
}

DATE_RE = re.compile(r"^\d{4}\.\d{2}\.\d{2}$")
INT_RE = re.compile(r"^\d+$")


def normalize_text(text: str) -> str:
    text = str(text).replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fetch_html(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        ),
        "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
    }
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    r.encoding = r.apparent_encoding
    return r.text


def parse_match_rows_from_html(html: str, season: int) -> pd.DataFrame:
    """
    SoccerDBのHTMLをtr単位で読む。
    ページ全体テキストへの正規表現では、2020年のように観客欄等が欠けた時に
    複数試合を1行に巻き込むことがあるため、必ずテーブル行単位で処理する。
    """
    soup = BeautifulSoup(html, "html.parser")
    rows = []

    for tr in soup.find_all("tr"):
        cells = [normalize_text(td.get_text(" ")) for td in tr.find_all(["th", "td"])]
        cells = [c for c in cells if c != ""]
        if len(cells) < 6:
            continue

        # date cell
        date_positions = [i for i, c in enumerate(cells) if DATE_RE.match(c)]
        if not date_positions:
            continue
        date_idx = date_positions[0]

        # Result cell
        result_positions = [i for i, c in enumerate(cells) if c == "Result"]
        for result_idx in result_positions:
            # 想定: [date, home, home_goal, Result, away_goal, away, ...]
            if result_idx - 2 < 0 or result_idx + 2 >= len(cells):
                continue
            home = cells[result_idx - 2]
            home_goals = cells[result_idx - 1]
            away_goals = cells[result_idx + 1]
            away = cells[result_idx + 2]

            if not (INT_RE.match(home_goals) and INT_RE.match(away_goals)):
                continue
            if DATE_RE.match(home) or DATE_RE.match(away):
                continue
            if len(home) > 40 or len(away) > 40:
                continue

            rows.append({
                "date": cells[date_idx].replace(".", "-"),
                "competition": "J3",
                "season": season,
                "home": home,
                "away": away,
                "home_goals": int(home_goals),
                "away_goals": int(away_goals),
            })

    df = pd.DataFrame(rows)

    # 念のため重複除去。同一日・同カードが重複したら1件にする。
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df = df.drop_duplicates(["date", "home", "away"], keep="first")
        df = df.sort_values(["date", "home", "away"]).reset_index(drop=True)

    return df


def scrape_j3_results(start_season: int = START_SEASON, end_season: int = END_SEASON) -> pd.DataFrame:
    all_dfs = []

    for season in range(start_season, end_season + 1):
        url = BASE_URL.format(competition_id=COMPETITION_ID, season=season)
        print(f"Scraping {season}: {url}")

        try:
            html = fetch_html(url)
            df_season = parse_match_rows_from_html(html, season)
            print(f"  -> {len(df_season)} matches")
            all_dfs.append(df_season)
        except Exception as e:
            print(f"  ERROR: {season}: {e}")

        time.sleep(1.0)

    if not all_dfs:
        return pd.DataFrame(columns=["date", "competition", "season", "home", "away", "home_goals", "away_goals"])

    df = pd.concat(all_dfs, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values(["date", "season", "home", "away"]).reset_index(drop=True)
    return df


def validate_j3(df: pd.DataFrame, strict: bool = False) -> list[str]:
    problems = []

    required_cols = ["date", "competition", "season", "home", "away", "home_goals", "away_goals"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        problems.append(f"Missing columns: {missing_cols}")
        return problems

    if df.empty:
        problems.append("J3 dataframe is empty.")
        return problems

    # 型・欠損
    for c in required_cols:
        na = df[c].isna().sum()
        if na:
            problems.append(f"Column {c} has {na} missing values.")

    # 長すぎるチーム名は、複数試合の巻き込みを疑う
    long_team_rows = df[(df["home"].astype(str).str.len() > 40) | (df["away"].astype(str).str.len() > 40)]
    if len(long_team_rows):
        problems.append(f"Too-long team names detected: {len(long_team_rows)} row(s).")

    dup = df.duplicated(["date", "home", "away"]).sum()
    if dup:
        problems.append(f"Duplicate date-home-away rows: {dup}")

    # 年度別件数チェック
    counts = df.groupby("season").size().to_dict()
    for season, expected in EXPECTED_J3_MATCH_COUNTS.items():
        if season in counts and counts[season] != expected:
            problems.append(f"Season {season}: expected {expected}, got {counts[season]}.")

    print("\n=== J3 validation ===")
    print(df.groupby("season").size().to_string())
    if problems:
        print("\nProblems:")
        for p in problems:
            print(" -", p)
    else:
        print("No obvious problems.")

    if strict and problems:
        raise ValueError("Validation failed:\n" + "\n".join(problems))

    return problems


def convert_j3_to_elo_schema(j3: pd.DataFrame, j1j2_columns: list[str]) -> pd.DataFrame:
    j3 = j3.copy()
    j3["date"] = pd.to_datetime(j3["date"]).dt.strftime("%Y-%m-%d")
    j3 = j3.sort_values(["season", "date", "home", "away"]).reset_index(drop=True)

    out = pd.DataFrame()
    out["match_id"] = "J3_" + j3["season"].astype(str) + "_" + j3.groupby("season").cumcount().add(1).astype(str).str.zfill(3)
    out["year"] = j3["season"].astype(int)
    out["date"] = j3["date"]
    out["division"] = "J3"
    out["competition_id"] = 1003
    out["home"] = j3["home"]
    out["away"] = j3["away"]
    out["home_goal"] = j3["home_goals"].astype(int)
    out["away_goal"] = j3["away_goals"].astype(int)
    out["home_pk"] = np.nan
    out["away_pk"] = np.nan
    out["has_pk"] = False
    out["pk_winner"] = np.nan

    home_result = np.where(out["home_goal"] > out["away_goal"], "W", np.where(out["home_goal"] < out["away_goal"], "L", "D"))
    away_result = np.where(out["home_goal"] < out["away_goal"], "W", np.where(out["home_goal"] > out["away_goal"], "L", "D"))
    out["normal_home_result"] = home_result
    out["normal_away_result"] = away_result
    out["official_home_result"] = home_result
    out["official_away_result"] = away_result

    # J1/J2 CSVと同じ列順にする
    return out[j1j2_columns]


def merge_j1j2_j3(j1j2_path: str, j3_path: str, output_path: str = MERGED_OUTPUT, strict: bool = True) -> pd.DataFrame:
    j1j2 = pd.read_csv(j1j2_path)
    j3 = pd.read_csv(j3_path)

    problems = validate_j3(j3, strict=strict)
    if problems and strict:
        raise ValueError("J3 validation failed. Fix J3 CSV before merge.")

    j3_elo = convert_j3_to_elo_schema(j3, list(j1j2.columns))
    merged = pd.concat([j1j2, j3_elo], ignore_index=True)
    merged["date_dt"] = pd.to_datetime(merged["date"])
    merged = merged.sort_values(["date_dt", "division", "match_id"]).drop(columns=["date_dt"]).reset_index(drop=True)

    # 最終検査
    if merged["match_id"].duplicated().any():
        raise ValueError("Duplicated match_id detected after merge.")
    if merged.duplicated(["date", "home", "away"]).any():
        raise ValueError("Duplicated date-home-away rows detected after merge.")

    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved merged CSV: {Path(output_path).resolve()}")
    print(merged["division"].value_counts().sort_index().to_string())
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scrape", action="store_true", help="Scrape J3 2014-2025 from SoccerDB")
    parser.add_argument("--merge", action="store_true", help="Merge J1/J2 CSV and J3 CSV")
    parser.add_argument("--start", type=int, default=START_SEASON)
    parser.add_argument("--end", type=int, default=END_SEASON)
    parser.add_argument("--j1j2", type=str, default="j1_j2_elo_input_1993_2025.csv")
    parser.add_argument("--j3", type=str, default=J3_OUTPUT)
    parser.add_argument("--j3-output", type=str, default=J3_OUTPUT)
    parser.add_argument("--merged-output", type=str, default=MERGED_OUTPUT)
    parser.add_argument("--allow-warnings", action="store_true", help="Allow merge even if validation warnings exist")
    args = parser.parse_args()

    if args.scrape:
        df_j3 = scrape_j3_results(args.start, args.end)
        validate_j3(df_j3, strict=False)
        df_j3.to_csv(args.j3_output, index=False, encoding="utf-8-sig")
        print(f"\nSaved J3 CSV: {Path(args.j3_output).resolve()}")

    if args.merge:
        merge_j1j2_j3(
            j1j2_path=args.j1j2,
            j3_path=args.j3,
            output_path=args.merged_output,
            strict=not args.allow_warnings,
        )

    if not args.scrape and not args.merge:
        parser.print_help()


if __name__ == "__main__":
    main()
