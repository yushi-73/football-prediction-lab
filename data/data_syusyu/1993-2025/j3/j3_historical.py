import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


COMPETITION_ID = 1003  # J3
BASE_URL = "https://soccer-db.net/competition/results/{competition_id}/{season}"

START_SEASON = 2014
END_SEASON = 2025

OUTPUT_PATH = "j3_elo_input_2014_2025.csv"


def fetch_html(url: str) -> str:
    """
    SoccerDBのHTMLを取得する。
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()
    response.encoding = response.apparent_encoding
    return response.text


def normalize_text(text: str) -> str:
    """
    余分な空白や改行を整理する。
    """
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_j3_results_from_html(html: str, season: int) -> pd.DataFrame:
    """
    SoccerDBのJ3試合結果ページから、
    date, competition, season, home, away, home_goals, away_goals
    を抽出する。
    """
    soup = BeautifulSoup(html, "html.parser")

    # ページ全体のテキストを取得
    text = soup.get_text(" ")
    text = normalize_text(text)

    # 例:
    # 2025.02.15 栃木シティ 2 Result 1 ＳＣ相模原 観客:3,283 ...
    pattern = re.compile(
        r"(\d{4}\.\d{2}\.\d{2})\s+"
        r"(.+?)\s+"
        r"(\d+)\s+Result\s+(\d+)\s+"
        r"(.+?)\s+"
        r"観客:",
        re.DOTALL,
    )

    rows = []

    for match in pattern.finditer(text):
        date_str = match.group(1)
        home = match.group(2).strip()
        home_goals = int(match.group(3))
        away_goals = int(match.group(4))
        away = match.group(5).strip()

        rows.append(
            {
                "date": date_str.replace(".", "-"),
                "competition": "J3",
                "season": season,
                "home": home,
                "away": away,
                "home_goals": home_goals,
                "away_goals": away_goals,
            }
        )

    df = pd.DataFrame(rows)

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])

    return df


def scrape_j3_results(start_season: int, end_season: int) -> pd.DataFrame:
    """
    指定した年度範囲のJ3試合結果をまとめて取得する。
    """
    all_dfs = []

    for season in range(start_season, end_season + 1):
        url = BASE_URL.format(
            competition_id=COMPETITION_ID,
            season=season,
        )

        print(f"Scraping {season}: {url}")

        try:
            html = fetch_html(url)
            df_season = parse_j3_results_from_html(html, season)

            print(f"  -> {len(df_season)} matches")

            if df_season.empty:
                print(f"  WARNING: {season} の試合結果を取得できませんでした。")

            all_dfs.append(df_season)

        except Exception as e:
            print(f"  ERROR: {season} の取得に失敗しました: {e}")

        # サーバー負荷を避けるため少し待つ
        time.sleep(1.0)

    if not all_dfs:
        return pd.DataFrame()

    df_all = pd.concat(all_dfs, ignore_index=True)

    df_all = df_all.sort_values(
        ["date", "season", "home", "away"]
    ).reset_index(drop=True)

    return df_all


if __name__ == "__main__":
    df_j3 = scrape_j3_results(START_SEASON, END_SEASON)

    print()
    print("取得結果")
    print(df_j3.head())
    print()
    print(df_j3.tail())
    print()
    print(df_j3.groupby("season").size())

    df_j3.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {Path(OUTPUT_PATH).resolve()}")