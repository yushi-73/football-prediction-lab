"""
Soccer D.B. から J2 の試合結果を収集し、既存の J1 CSV と同じ列構造で保存するスクリプト。

入力: なし（任意で J1 CSV を同じフォルダに置く）
出力:
  - j2_historical_results_1999_2025_table_fixed.csv
  - j1_j2_historical_results_1993_2025_table_fixed.csv  ※J1 CSV が存在する場合

必要ライブラリ:
  pip install requests beautifulsoup4 pandas
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Optional, Dict, List

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag


BASE_URL = "https://soccer-db.net/competition/results/{competition_id}/{year}"
J2_COMPETITION_ID = 1002

# アップロード済みのJ1 CSVと同じ列順
OUTPUT_COLUMNS = [
    "year",
    "date",
    "home",
    "away",
    "home_goal",
    "away_goal",
    "home_pk",
    "away_pk",
    "has_pk",
    "pk_winner",
    "normal_home_result",
    "normal_away_result",
    "official_home_result",
    "official_away_result",
    "home_raw",
    "away_raw",
    "parser",
    "source_url",
]


TEAM_NAME_MAP = {
    # 必要になったらここに追加。
    # Soccer D.B.側で表記揺れが出た場合、左: 元表記、右: 統一表記。
    # 例: "大宮アルディージャ": "ＲＢ大宮アルディージャ",
}


def normalize_text(s: str) -> str:
    """空白を潰して前後を削る。"""
    return re.sub(r"\s+", " ", str(s)).strip()


def normalize_team_name(s: str) -> str:
    s = normalize_text(s)
    return TEAM_NAME_MAP.get(s, s)


def parse_score_cell(text: str) -> tuple[Optional[int], Optional[int]]:
    """
    スコアセルから通常得点とPK得点を取り出す。

    想定例:
      "2"       -> (2, None)
      "1 (4)"   -> (1, 4)
      "1(4)"    -> (1, 4)
    """
    nums = re.findall(r"\d+", normalize_text(text))
    if not nums:
        return None, None
    goal = int(nums[0])
    pk = int(nums[1]) if len(nums) >= 2 else None
    return goal, pk


def result_from_goals(home_goal: int, away_goal: int) -> tuple[str, str]:
    if home_goal > away_goal:
        return "W", "L"
    if home_goal < away_goal:
        return "L", "W"
    return "D", "D"


def official_result(
    home_goal: int,
    away_goal: int,
    home_pk: Optional[int],
    away_pk: Optional[int],
) -> tuple[str, str, bool, Optional[str]]:
    """通常スコアとPK情報から公式勝敗を作る。"""
    normal_home, normal_away = result_from_goals(home_goal, away_goal)

    has_pk = home_pk is not None and away_pk is not None
    pk_winner = None

    if has_pk:
        if home_pk > away_pk:
            return "W", "L", True, "home"
        if home_pk < away_pk:
            return "L", "W", True, "away"
        # 通常はPK同点は起きないが、念のため
        return normal_home, normal_away, True, None

    return normal_home, normal_away, False, pk_winner


def cell_text(cell: Tag) -> str:
    return normalize_text(cell.get_text(" ", strip=True))


def anchor_text(cell: Tag) -> str:
    """セル内の最初のリンクテキストを優先して返す。"""
    a = cell.find("a")
    if a:
        return normalize_text(a.get_text(" ", strip=True))
    return cell_text(cell)


def parse_row_by_cells(tr: Tag, year: int, source_url: str) -> Optional[Dict]:
    """Resultリンクを含む1行を、セル位置ベースでパースする。"""
    cells = tr.find_all(["th", "td"])
    if not cells:
        return None

    result_idx = None
    for i, c in enumerate(cells):
        if c.find("a", string=lambda x: x and normalize_text(x) == "Result"):
            result_idx = i
            break
        if cell_text(c) == "Result":
            result_idx = i
            break

    if result_idx is None:
        return None

    # 典型形: date | home | home_goal | Result | away_goal | away
    if result_idx - 2 < 0 or result_idx + 2 >= len(cells):
        return None

    row_text = cell_text(tr)
    m_date = re.search(r"\d{4}\.\d{2}\.\d{2}", row_text)
    if not m_date:
        return None
    date = m_date.group(0).replace(".", "-")

    home_raw = anchor_text(cells[result_idx - 2])
    away_raw = anchor_text(cells[result_idx + 2])
    home = normalize_team_name(home_raw)
    away = normalize_team_name(away_raw)

    home_goal, home_pk = parse_score_cell(cell_text(cells[result_idx - 1]))
    away_goal, away_pk = parse_score_cell(cell_text(cells[result_idx + 1]))

    if home_goal is None or away_goal is None:
        return None

    normal_home, normal_away = result_from_goals(home_goal, away_goal)
    official_home, official_away, has_pk, pk_winner = official_result(
        home_goal, away_goal, home_pk, away_pk
    )

    return {
        "year": year,
        "date": date,
        "home": home,
        "away": away,
        "home_goal": home_goal,
        "away_goal": away_goal,
        "home_pk": home_pk,
        "away_pk": away_pk,
        "has_pk": has_pk,
        "pk_winner": pk_winner,
        "normal_home_result": normal_home,
        "normal_away_result": normal_away,
        "official_home_result": official_home,
        "official_away_result": official_away,
        "home_raw": home_raw,
        "away_raw": away_raw,
        "parser": "cells",
        "source_url": source_url,
    }


def parse_line_fallback(line: str, year: int, source_url: str) -> Optional[Dict]:
    """
    セルで取れなかった場合の保険。
    例: 2025.02.15 ヴァンフォーレ甲府 1 Result 0 レノファ山口ＦＣ
    """
    line = normalize_text(line)
    pat = re.compile(
        r"(?P<date>\d{4}\.\d{2}\.\d{2})\s+"
        r"(?P<home>.+?)\s+"
        r"(?P<hscore>\d+(?:\s*\(\d+\))?)\s+"
        r"Result\s+"
        r"(?P<ascore>\d+(?:\s*\(\d+\))?)\s+"
        r"(?P<away>.+)$"
    )
    m = pat.match(line)
    if not m:
        return None

    home_raw = m.group("home")
    away_raw = m.group("away")

    # 観客情報などが混ざった場合の最低限の切り落とし
    away_raw = re.split(r"\s+観客:|\s+Referee:", away_raw)[0]

    home_goal, home_pk = parse_score_cell(m.group("hscore"))
    away_goal, away_pk = parse_score_cell(m.group("ascore"))
    if home_goal is None or away_goal is None:
        return None

    normal_home, normal_away = result_from_goals(home_goal, away_goal)
    official_home, official_away, has_pk, pk_winner = official_result(
        home_goal, away_goal, home_pk, away_pk
    )

    return {
        "year": year,
        "date": m.group("date").replace(".", "-"),
        "home": normalize_team_name(home_raw),
        "away": normalize_team_name(away_raw),
        "home_goal": home_goal,
        "away_goal": away_goal,
        "home_pk": home_pk,
        "away_pk": away_pk,
        "has_pk": has_pk,
        "pk_winner": pk_winner,
        "normal_home_result": normal_home,
        "normal_away_result": normal_away,
        "official_home_result": official_home,
        "official_away_result": official_away,
        "home_raw": home_raw,
        "away_raw": away_raw,
        "parser": "line_fallback",
        "source_url": source_url,
    }


def parse_matches(html: str, year: int, source_url: str) -> List[Dict]:
    soup = BeautifulSoup(html, "html.parser")
    rows: List[Dict] = []
    seen = set()

    # まずはtable行ベースで抽出
    for tr in soup.find_all("tr"):
        row = parse_row_by_cells(tr, year, source_url)
        if row is None:
            continue
        key = (row["date"], row["home"], row["away"], row["home_goal"], row["away_goal"])
        if key not in seen:
            rows.append(row)
            seen.add(key)

    # 取れなかった/少なすぎる場合はテキスト行で保険
    if len(rows) == 0:
        for line in soup.get_text("\n", strip=True).splitlines():
            row = parse_line_fallback(line, year, source_url)
            if row is None:
                continue
            key = (row["date"], row["home"], row["away"], row["home_goal"], row["away_goal"])
            if key not in seen:
                rows.append(row)
                seen.add(key)

    return rows


def fetch_html(session: requests.Session, url: str, timeout: int = 30) -> str:
    r = session.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


def collect_j2(start_year: int, end_year: int, sleep_sec: float) -> pd.DataFrame:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            ),
            "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
        }
    )

    all_rows: List[Dict] = []

    for year in range(start_year, end_year + 1):
        url = BASE_URL.format(competition_id=J2_COMPETITION_ID, year=year)
        print(f"fetch: {year} {url}")

        try:
            html = fetch_html(session, url)
        except requests.HTTPError as e:
            print(f"  !! HTTP error: {e}")
            print("  !! 時間を置くか、sleep秒数を増やして再実行してください。")
            continue
        except requests.RequestException as e:
            print(f"  !! request error: {e}")
            continue

        rows = parse_matches(html, year, url)
        print(f"  -> {len(rows)} matches")

        if len(rows) == 0:
            print("  !! 0件です。HTML構造が変わった可能性があります。")

        all_rows.extend(rows)
        time.sleep(sleep_sec)

    df = pd.DataFrame(all_rows)
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = df[OUTPUT_COLUMNS].copy()
    df = df.sort_values(["date", "home", "away"]).reset_index(drop=True)

    # 型をJ1 CSVに寄せる
    for col in ["home_pk", "away_pk"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=1999)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--out", type=str, default="j2_historical_results_1999_2025_table_fixed.csv")
    parser.add_argument(
        "--j1-csv",
        type=str,
        default="j1_historical_results_1993_2025_table_fixed.csv",
        help="同じ列構造のJ1 CSV。存在すればJ1+J2結合版も出力します。",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    df_j2 = collect_j2(args.start_year, args.end_year, args.sleep)
    df_j2.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("\n=== J2 summary ===")
    print(f"saved: {out_path}")
    print(f"rows : {len(df_j2)}")
    if not df_j2.empty:
        print(df_j2.groupby("year").size())
        print("has_pk:", int(df_j2["has_pk"].sum()))

    j1_path = Path(args.j1_csv)
    if j1_path.exists() and not df_j2.empty:
        df_j1 = pd.read_csv(j1_path)
        missing_cols = [c for c in OUTPUT_COLUMNS if c not in df_j1.columns]
        if missing_cols:
            print(f"\n!! J1 CSVに足りない列があります: {missing_cols}")
            return

        df_j1 = df_j1[OUTPUT_COLUMNS].copy()
        df_j1.insert(0, "division", "J1")
        df_j2_with_div = df_j2.copy()
        df_j2_with_div.insert(0, "division", "J2")

        merged = pd.concat([df_j1, df_j2_with_div], ignore_index=True)
        merged = merged.sort_values(["date", "division", "home", "away"]).reset_index(drop=True)

        merged_path = Path("j1_j2_historical_results_1993_2025_table_fixed.csv")
        merged.to_csv(merged_path, index=False, encoding="utf-8-sig")

        print("\n=== J1 + J2 summary ===")
        print(f"saved: {merged_path}")
        print(f"rows : {len(merged)}")
        print(merged.groupby(["division", "year"]).size())


if __name__ == "__main__":
    main()
