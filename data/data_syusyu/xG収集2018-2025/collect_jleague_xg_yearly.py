"""
Jリーグ公式サイトから年度別・リーグ別のチームxG / xGAを収集してCSV化するスクリプト。

対象URL例:
  https://www.jleague.jp/stats/j1/club/2024/expected_goals/
  https://www.jleague.jp/stats/j1/club/2024/expected_goals_against/

出力:
  jleague_xg_yearly_2018_2026.csv
  jleague_xg_yearly_fetch_errors.csv

使い方:
  pip install requests beautifulsoup4
  python collect_jleague_xg_yearly.py

オプション例:
  python collect_jleague_xg_yearly.py --start-year 2018 --end-year 2026 --leagues j1 j2 j3
  python collect_jleague_xg_yearly.py --output data/jleague_xg_yearly.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.jleague.jp/stats/{league}/club/{season}/{slug}/"

STAT_SPECS = {
    "xg": {
        "slug": "expected_goals",
        "label": "ゴール期待値",
        "rank_field": "xg_rank",
        "value_field": "xg",
        "url_field": "source_xg_url",
    },
    "xga": {
        "slug": "expected_goals_against",
        "label": "被ゴール期待値",
        "rank_field": "xga_rank",
        "value_field": "xga",
        "url_field": "source_xga_url",
    },
}

TEAM_NAME_MAP = {
    "鹿島アントラーズ": "鹿島",
    "浦和レッズ": "浦和",
    "柏レイソル": "柏",
    "ＦＣ東京": "FC東京",
    "FC東京": "FC東京",
    "東京ヴェルディ": "東京V",
    "東京ヴェルディ１９６９": "東京V",
    "東京ヴェルディ1969": "東京V",
    "ヴェルディ川崎": "東京V",
    "川崎フロンターレ": "川崎F",
    "横浜Ｆ・マリノス": "横浜FM",
    "横浜F・マリノス": "横浜FM",
    "横浜マリノス": "横浜FM",
    "横浜ＦＣ": "横浜FC",
    "横浜FC": "横浜FC",
    "湘南ベルマーレ": "湘南",
    "アルビレックス新潟": "新潟",
    "清水エスパルス": "清水",
    "名古屋グランパス": "名古屋",
    "京都サンガF.C.": "京都",
    "京都サンガＦ.Ｃ.": "京都",
    "京都サンガ": "京都",
    "京都パープルサンガ": "京都",
    "ガンバ大阪": "G大阪",
    "Ｇ大阪": "G大阪",
    "セレッソ大阪": "C大阪",
    "Ｃ大阪": "C大阪",
    "ヴィッセル神戸": "神戸",
    "ファジアーノ岡山": "岡山",
    "サンフレッチェ広島": "広島",
    "アビスパ福岡": "福岡",
    "FC町田ゼルビア": "町田",
    "ＦＣ町田ゼルビア": "町田",
    "町田ゼルビア": "町田",
    "ジュビロ磐田": "磐田",
    "北海道コンサドーレ札幌": "札幌",
    "コンサドーレ札幌": "札幌",
    "サガン鳥栖": "鳥栖",
    "ベガルタ仙台": "仙台",
    "大分トリニータ": "大分",
    "大宮アルディージャ": "大宮",
    "ＲＢ大宮アルディージャ": "大宮",
    "RB大宮アルディージャ": "大宮",
    "ジェフユナイテッド千葉": "千葉",
    "ジェフユナイテッド市原": "千葉",
    "ジェフユナイテッド市原・千葉": "千葉",
    "ヴァンフォーレ甲府": "甲府",
    "松本山雅ＦＣ": "松本",
    "松本山雅FC": "松本",
    "徳島ヴォルティス": "徳島",
    "モンテディオ山形": "山形",
    "ザスパクサツ群馬": "群馬",
    "ザスパ群馬": "群馬",
    "Ｖ・ファーレン長崎": "長崎",
    "V・ファーレン長崎": "長崎",
    "ブラウブリッツ秋田": "秋田",
    "水戸ホーリーホック": "水戸",
    "ツエーゲン金沢": "金沢",
    "栃木ＳＣ": "栃木SC",
    "栃木SC": "栃木SC",
    "栃木シティ": "栃木C",
    "いわきＦＣ": "いわき",
    "いわきFC": "いわき",
    "藤枝ＭＹＦＣ": "藤枝",
    "藤枝MYFC": "藤枝",
    "愛媛ＦＣ": "愛媛",
    "愛媛FC": "愛媛",
    "レノファ山口ＦＣ": "山口",
    "レノファ山口FC": "山口",
    "ロアッソ熊本": "熊本",
    "鹿児島ユナイテッドＦＣ": "鹿児島",
    "鹿児島ユナイテッドFC": "鹿児島",
    "カターレ富山": "富山",
    "FC今治": "今治",
    "テゲバジャーロ宮崎": "宮崎",
    "ヴァンラーレ八戸": "八戸",
    "福島ユナイテッドＦＣ": "福島",
    "福島ユナイテッドFC": "福島",
    "ＳＣ相模原": "相模原",
    "SC相模原": "相模原",
    "ＡＣ長野パルセイロ": "長野",
    "AC長野パルセイロ": "長野",
    "ＦＣ岐阜": "岐阜",
    "FC岐阜": "岐阜",
    "レイラック滋賀ＦＣ": "滋賀",
    "レイラック滋賀": "滋賀",
    "FC大阪": "FC大阪",
    "奈良クラブ": "奈良",
    "ガイナーレ鳥取": "鳥取",
    "カマタマーレ讃岐": "讃岐",
    "高知ユナイテッドＳＣ": "高知",
    "高知ユナイテッドSC": "高知",
    "ギラヴァンツ北九州": "北九州",
    "ＦＣ琉球": "琉球",
    "FC琉球": "琉球",
}

LEAGUE_LEVEL = {"j1": 1, "j2": 2, "j3": 3}
ROW_PATTERN = re.compile(r"^\s*(\d{1,2})\s+(.+?)\s+([0-9]+(?:\.[0-9]+)?)\s*$")


@dataclass
class StatRow:
    season: int
    league: str
    league_level: int
    team: str
    team_std: str
    xg_rank: Optional[int] = None
    xg: Optional[float] = None
    xga_rank: Optional[int] = None
    xga: Optional[float] = None
    xg_diff: Optional[float] = None
    source_xg_url: str = ""
    source_xga_url: str = ""
    scraped_at: str = ""


def standardize_team_name(name: str) -> str:
    name = (
        str(name)
        .replace("【公式】", "")
        .replace("\u3000", " ")
        .replace("\xa0", " ")
        .strip()
    )
    return TEAM_NAME_MAP.get(name, name)


def fetch_html(url: str, timeout: int = 20) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        ),
        "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
    }
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.text


def parse_stat_page(html: str, label: str, season: int) -> List[Tuple[int, str, float]]:
    """Jリーグのチームスタッツページから rank, team, value を抜き出す。"""
    soup = BeautifulSoup(html, "html.parser")

    candidates: List[str] = []

    # 1) li / a / tr / td単位で拾う。親要素の巨大テキストによる誤検出を避ける。
    for tag in soup.find_all(["li", "a", "tr", "td", "div"]):
        text = " ".join(tag.get_text(" ", strip=True).split())
        if text:
            candidates.append(text)

    # 2) ページ全体テキストの該当セクションも保険として使う。
    full_text = soup.get_text("\n", strip=True)
    marker = f"{label} {season}シーズン"
    start = full_text.find(marker)
    if start != -1:
        section = full_text[start:]
        end = section.find("データ提供")
        if end != -1:
            section = section[:end]
        candidates.extend(line.strip() for line in section.splitlines() if line.strip())

    rows: List[Tuple[int, str, float]] = []
    seen = set()
    for text in candidates:
        m = ROW_PATTERN.match(text)
        if not m:
            continue
        rank = int(m.group(1))
        team = m.group(2).strip()
        value = float(m.group(3))

        # スタッツ表は通常1〜24位程度。ナビや年度選択の誤検出を避ける。
        if rank <= 0 or rank > 30:
            continue
        if len(team) < 2:
            continue

        key = (rank, team, value)
        if key in seen:
            continue
        seen.add(key)
        rows.append((rank, team, value))

    # rank順に並べ、同じrankの重複があれば先頭を採用
    rows = sorted(rows, key=lambda x: x[0])
    deduped: List[Tuple[int, str, float]] = []
    used_ranks = set()
    for row in rows:
        if row[0] in used_ranks:
            continue
        used_ranks.add(row[0])
        deduped.append(row)

    return deduped


def collect_one_stat(league: str, season: int, stat_key: str) -> Tuple[List[Tuple[int, str, float]], str]:
    spec = STAT_SPECS[stat_key]
    url = BASE_URL.format(league=league, season=season, slug=spec["slug"])
    html = fetch_html(url)
    rows = parse_stat_page(html, label=spec["label"], season=season)
    return rows, url


def collect_yearly_xg(
    seasons: Iterable[int],
    leagues: Iterable[str],
    sleep_sec: float = 0.5,
) -> Tuple[List[StatRow], List[Dict[str, str]]]:
    scraped_at = datetime.now().isoformat(timespec="seconds")
    records: Dict[Tuple[int, str, str], StatRow] = {}
    errors: List[Dict[str, str]] = []

    for season in seasons:
        for league in leagues:
            league = league.lower()
            if league not in LEAGUE_LEVEL:
                raise ValueError(f"Unsupported league: {league}. Use j1, j2, j3.")

            for stat_key in ["xg", "xga"]:
                spec = STAT_SPECS[stat_key]
                url = BASE_URL.format(league=league, season=season, slug=spec["slug"])

                try:
                    print(f"fetch: {league.upper()} {season} {stat_key} -> {url}")
                    rows, source_url = collect_one_stat(league, season, stat_key)
                    if not rows:
                        raise RuntimeError("No rows parsed")
                except Exception as e:
                    print(f"  ERROR: {e}", file=sys.stderr)
                    errors.append({
                        "season": str(season),
                        "league": league,
                        "stat": stat_key,
                        "url": url,
                        "error": repr(e),
                    })
                    time.sleep(sleep_sec)
                    continue

                for rank, team, value in rows:
                    team_std = standardize_team_name(team)
                    key = (season, league, team_std)
                    if key not in records:
                        records[key] = StatRow(
                            season=season,
                            league=league,
                            league_level=LEAGUE_LEVEL[league],
                            team=team,
                            team_std=team_std,
                            scraped_at=scraped_at,
                        )

                    record = records[key]
                    # xGとxGAで表記揺れがあった場合、最初のraw名を保持しつつ標準名で結合する。
                    setattr(record, spec["rank_field"], rank)
                    setattr(record, spec["value_field"], value)
                    setattr(record, spec["url_field"], source_url)

                time.sleep(sleep_sec)

    final_rows = list(records.values())
    for r in final_rows:
        if r.xg is not None and r.xga is not None:
            r.xg_diff = round(r.xg - r.xga, 3)

    final_rows.sort(key=lambda r: (r.season, r.league_level, r.team_std))
    return final_rows, errors


def write_csv(rows: List[StatRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "season",
        "league",
        "league_level",
        "team",
        "team_std",
        "xg_rank",
        "xg",
        "xga_rank",
        "xga",
        "xg_diff",
        "source_xg_url",
        "source_xga_url",
        "scraped_at",
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_errors_csv(errors: List[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["season", "league", "stat", "url", "error"]
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in errors:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--leagues", nargs="+", default=["j1", "j2", "j3"], choices=["j1", "j2", "j3"])
    parser.add_argument("--sleep", type=float, default=0.5, help="アクセス間隔。公式サイトに負荷をかけないため短くしすぎない。")
    parser.add_argument("--output", type=str, default="jleague_xg_yearly_2018_2026.csv")
    parser.add_argument("--errors-output", type=str, default="jleague_xg_yearly_fetch_errors.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seasons = range(args.start_year, args.end_year + 1)
    rows, errors = collect_yearly_xg(seasons=seasons, leagues=args.leagues, sleep_sec=args.sleep)

    output_path = Path(args.output)
    errors_path = Path(args.errors_output)
    write_csv(rows, output_path)
    write_errors_csv(errors, errors_path)

    print("\n==============================")
    print("収集完了")
    print("==============================")
    print(f"rows: {len(rows)}")
    print(f"errors: {len(errors)}")
    print(f"output: {output_path.resolve()}")
    print(f"errors_output: {errors_path.resolve()}")

    if errors:
        print("\n一部取得できなかったページがあります。詳細は errors_output を確認してください。")


if __name__ == "__main__":
    main()
