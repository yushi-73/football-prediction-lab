"""
Soccer D.B. から J2 の試合結果を収集する修正版スクリプト。

前版の問題:
- tableセル位置ベースで読むと、Soccer D.B. 側のHTML構造により home / away が空欄になる場合がある
- home / away が空欄のまま重複除去され、同日同スコアの別試合が落ちる

修正方針:
- ページ全体の表示テキストから「日付 ホーム 得点 Result 得点 アウェイ」を正規表現で抽出する
- チーム名が空欄の行を出力しない
- 収集後に年度別件数、空欄、重複を検査する

必要ライブラリ:
  pip install requests beautifulsoup4

実行例:
  python collect_j2_soccerdb_fixed.py --start-year 1999 --end-year 2025
  python collect_j2_soccerdb_fixed.py --start-year 2025 --end-year 2025 --sleep 2
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://soccer-db.net/competition/results/{competition_id}/{year}"
J2_COMPETITION_ID = 1002

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
    # 表記揺れが出たらここで統一する。
    # "大宮アルディージャ": "ＲＢ大宮アルディージャ",
}

META_STOP_WORDS = (
    "観客:",
    "Att.",
    "Referee:",
    "主審:",
)

SCORE_RE = r"(?:\(\d+\)\s*)?\d+(?:\s*\(\d+\))?"

# 例:
# 2025.02.15 ヴァンフォーレ甲府 1 Result 0 レノファ山口ＦＣ 観客:10,152 ...
# 2025.10.13 Veroskronos Tsuno 0(4) Result (3)0 Cento Cuore Harima Att.140 ...
MATCH_RE = re.compile(
    rf"(?P<date>\d{{4}}\.\d{{2}}\.\d{{2}})\s+"
    rf"(?P<home>.+?)\s+"
    rf"(?P<hscore>{SCORE_RE})\s+"
    rf"(?:[WDL]\s+)?"
    rf"Result\s+"
    rf"(?P<ascore>{SCORE_RE})\s+"
    rf"(?P<away>.+?)"
    rf"(?=\s+(?:{'|'.join(map(re.escape, META_STOP_WORDS))})|\s+\d{{4}}\.\d{{2}}\.\d{{2}}|\s+##|\s+Soccer D\.B\.|$)"
)


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_team_name(name: str) -> str:
    name = normalize_text(name)
    return TEAM_NAME_MAP.get(name, name)


def parse_score(score_text: str) -> tuple[Optional[int], Optional[int]]:
    """
    得点とPK得点を返す。

    対応例:
      2       -> (2, None)
      1(4)    -> (1, 4)
      1 (4)   -> (1, 4)
      (3)0    -> (0, 3)
      (3) 0   -> (0, 3)
    """
    s = normalize_text(score_text).replace(" ", "")

    m = re.fullmatch(r"(?P<goal>\d+)\((?P<pk>\d+)\)", s)
    if m:
        return int(m.group("goal")), int(m.group("pk"))

    m = re.fullmatch(r"\((?P<pk>\d+)\)(?P<goal>\d+)", s)
    if m:
        return int(m.group("goal")), int(m.group("pk"))

    m = re.fullmatch(r"\d+", s)
    if m:
        return int(s), None

    return None, None


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
) -> tuple[str, str, bool, str]:
    normal_home, normal_away = result_from_goals(home_goal, away_goal)
    has_pk = home_pk is not None and away_pk is not None

    if has_pk:
        if home_pk > away_pk:
            return "W", "L", True, "home"
        if home_pk < away_pk:
            return "L", "W", True, "away"
        return normal_home, normal_away, True, ""

    return normal_home, normal_away, False, ""


def row_from_match(match: re.Match[str], year: int, source_url: str) -> Optional[dict]:
    home_raw = normalize_text(match.group("home"))
    away_raw = normalize_text(match.group("away"))

    # 念のためメタ情報を切る。通常は正規表現の先読みで止まる。
    for stop in META_STOP_WORDS:
        if stop in away_raw:
            away_raw = away_raw.split(stop, 1)[0].strip()

    home = normalize_team_name(home_raw)
    away = normalize_team_name(away_raw)

    home_goal, home_pk = parse_score(match.group("hscore"))
    away_goal, away_pk = parse_score(match.group("ascore"))

    if not home or not away:
        return None
    if home_goal is None or away_goal is None:
        return None

    normal_home, normal_away = result_from_goals(home_goal, away_goal)
    official_home, official_away, has_pk, pk_winner = official_result(
        home_goal, away_goal, home_pk, away_pk
    )

    return {
        "year": year,
        "date": match.group("date").replace(".", "-"),
        "home": home,
        "away": away,
        "home_goal": home_goal,
        "away_goal": away_goal,
        "home_pk": "" if home_pk is None else home_pk,
        "away_pk": "" if away_pk is None else away_pk,
        "has_pk": has_pk,
        "pk_winner": pk_winner,
        "normal_home_result": normal_home,
        "normal_away_result": normal_away,
        "official_home_result": official_home,
        "official_away_result": official_away,
        "home_raw": home_raw,
        "away_raw": away_raw,
        "parser": "text_regex",
        "source_url": source_url,
    }


def parse_matches(html: str, year: int, source_url: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")

    # script/style はノイズなので除く。
    for tag in soup(["script", "style"]):
        tag.decompose()

    text = normalize_text(soup.get_text(" ", strip=True))

    rows: list[dict] = []
    seen = set()

    for match in MATCH_RE.finditer(text):
        row = row_from_match(match, year, source_url)
        if row is None:
            continue

        # チーム名まで含めて重複判定する。blankのまま重複除去しないことが重要。
        key = (
            row["date"],
            row["home"],
            row["away"],
            row["home_goal"],
            row["away_goal"],
            row["home_pk"],
            row["away_pk"],
        )
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)

    return rows


def fetch_html(session: requests.Session, url: str, timeout: int = 30) -> str:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def collect(start_year: int, end_year: int, sleep_sec: float) -> list[dict]:
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

    all_rows: list[dict] = []

    for year in range(start_year, end_year + 1):
        url = BASE_URL.format(competition_id=J2_COMPETITION_ID, year=year)
        print(f"fetch: {year} {url}")

        try:
            html = fetch_html(session, url)
        except requests.RequestException as exc:
            print(f"  !! fetch failed: {exc}")
            continue

        rows = parse_matches(html, year, url)
        blank_rows = [r for r in rows if not r["home"] or not r["away"]]
        print(f"  -> {len(rows)} matches / blank_team_rows={len(blank_rows)}")
        all_rows.extend(rows)
        time.sleep(sleep_sec)

    all_rows.sort(key=lambda r: (r["date"], r["home"], r["away"]))
    return all_rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in OUTPUT_COLUMNS})


def read_csv_dicts(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_merged_if_possible(j1_csv: Path, j2_rows: list[dict], merged_out: Path) -> None:
    if not j1_csv.exists():
        return

    j1_rows = read_csv_dicts(j1_csv)
    if not j1_rows:
        return

    missing = [c for c in OUTPUT_COLUMNS if c not in j1_rows[0]]
    if missing:
        print(f"\n!! J1 CSVに必要列がありません: {missing}")
        return

    merged: list[dict] = []
    for row in j1_rows:
        new = {"division": "J1"}
        new.update({c: row.get(c, "") for c in OUTPUT_COLUMNS})
        merged.append(new)
    for row in j2_rows:
        new = {"division": "J2"}
        new.update({c: row.get(c, "") for c in OUTPUT_COLUMNS})
        merged.append(new)

    merged.sort(key=lambda r: (r["date"], r["division"], r["home"], r["away"]))

    with merged_out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["division"] + OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(merged)

    print(f"merged saved: {merged_out} rows={len(merged)}")


def validate(rows: list[dict]) -> None:
    print("\n=== validation ===")
    print(f"rows: {len(rows)}")

    by_year = Counter(str(r["year"]) for r in rows)
    print("\nrows by year:")
    for year in sorted(by_year, key=int):
        teams = set()
        for r in rows:
            if str(r["year"]) == year:
                teams.add(r["home"])
                teams.add(r["away"])
        print(f"  {year}: {by_year[year]} matches / {len(teams)} teams")

    blank = [r for r in rows if not r["home"] or not r["away"]]
    print(f"\nblank home/away rows: {len(blank)}")

    bad_score = []
    for r in rows:
        try:
            int(r["home_goal"])
            int(r["away_goal"])
        except Exception:
            bad_score.append(r)
    print(f"bad score rows: {len(bad_score)}")

    exact_key_counts = Counter(
        (r["year"], r["date"], r["home"], r["away"], r["home_goal"], r["away_goal"])
        for r in rows
    )
    exact_dups = [k for k, v in exact_key_counts.items() if v > 1]
    print(f"exact duplicate rows: {len(exact_dups)}")

    pair_counts_by_year = defaultdict(Counter)
    for r in rows:
        pair_counts_by_year[str(r["year"])][(r["home"], r["away"])] += 1
    repeated_home_away = {
        y: [(pair, n) for pair, n in cnt.items() if n > 1]
        for y, cnt in pair_counts_by_year.items()
    }
    repeated_home_away = {y: v for y, v in repeated_home_away.items() if v}
    print(f"same home-away repeated within same year: {sum(len(v) for v in repeated_home_away.values())}")

    if blank:
        print("\n!! home/away が空欄の行があります。Eloには使わないでください。")
    if exact_dups:
        print("\n!! 完全重複があります。確認してください。")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=1999)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--out", type=str, default="j2_historical_results_1999_2025_table_fixed_v2.csv")
    parser.add_argument("--j1-csv", type=str, default="j1_historical_results_1993_2025_table_fixed.csv")
    parser.add_argument("--merged-out", type=str, default="j1_j2_historical_results_1993_2025_table_fixed_v2.csv")
    args = parser.parse_args()

    rows = collect(args.start_year, args.end_year, args.sleep)
    validate(rows)

    out_path = Path(args.out)
    write_csv(rows, out_path)
    print(f"\nsaved: {out_path}")

    write_merged_if_possible(Path(args.j1_csv), rows, Path(args.merged_out))


if __name__ == "__main__":
    main()
