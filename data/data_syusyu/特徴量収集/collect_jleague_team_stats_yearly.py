"""
Jリーグ公式サイトから、年度別・リーグ別のチームスタッツをまとめてCSV化するスクリプト。

主な用途:
  ・前年レーティング用の特徴量CSV作成
  ・J1/J2/J3の年度別スタッツ比較
  ・昇格組の前年J2/J3成績を補正に使うための基礎データ作成

対象URL例:
  https://www.jleague.jp/stats/j1/club/2024/score/
  https://www.jleague.jp/stats/j1/club/2024/expected_goals/
  https://www.jleague.jp/stats/j2/club/2024/chance_create/

出力:
  jleague_team_stats_yearly_wide_2019_2025.csv
  jleague_team_stats_yearly_long_2019_2025.csv
  jleague_team_stats_yearly_summary.csv
  jleague_team_stats_yearly_errors.csv

使い方:
  pip install requests beautifulsoup4 pandas
  python collect_jleague_team_stats_yearly.py

年度やリーグを絞る例:
  python collect_jleague_team_stats_yearly.py --start-year 2024 --end-year 2024 --leagues j1 j2

注意:
  ・公式サイトに負荷をかけないよう、デフォルトで0.5秒待機します。
  ・2018は一部スタッツが0や欠損になる可能性があるため、デフォルトは2019開始にしています。
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.jleague.jp/stats/{league}/club/{season}/{slug}/"
LEAGUE_LEVEL = {"j1": 1, "j2": 2, "j3": 3}


@dataclass(frozen=True)
class StatSpec:
    key: str
    slug: str
    label: str
    category: str
    # True: 多いほど良い指標 / False: 多いほど悪い指標 / None: スタイル指標
    higher_is_better: Optional[bool] = True


# ------------------------------------------------------------------
# 収集対象スタッツ
# ------------------------------------------------------------------
# URL slug はJリーグ公式サイトのチームスタッツURLに対応。
# 必要ならここに追加してください。
STAT_SPECS: Dict[str, StatSpec] = {
    # 攻撃・守備の基礎
    "goals": StatSpec("goals", "score", "得点総数", "attack", True),
    "goals_against": StatSpec("goals_against", "lost", "失点総数", "defense", False),

    # シュート系
    "shots": StatSpec("shots", "shoot", "シュート総数", "attack", True),
    "shots_on_target": StatSpec("shots_on_target", "shoot_on_target", "枠内シュート総数", "attack", True),
    "shots_against": StatSpec("shots_against", "suffer_shoot", "被シュート総数", "defense", False),
    "shots_on_target_against": StatSpec(
        "shots_on_target_against",
        "suffer_shoot_on_target",
        "被枠内シュート総数",
        "defense",
        False,
    ),

    # xG系
    "xg": StatSpec("xg", "expected_goals", "ゴール期待値", "attack", True),
    "xga": StatSpec("xga", "expected_goals_against", "被ゴール期待値", "defense", False),
    # PK除外xGAも必要なら後で使えるように収集対象に入れておく
    "xga_non_pk": StatSpec(
        "xga_non_pk",
        "expected_goals_against_excluding_pk",
        "被ゴール期待値 ※PKを除く",
        "defense",
        False,
    ),

    # 補助指標
    "chance_create": StatSpec("chance_create", "chance_create", "チャンスクリエイト総数", "attack", True),
    "clean_sheet": StatSpec("clean_sheet", "clean_sheet", "クリーンシート総数", "defense", True),
    "ball_possession": StatSpec("ball_possession", "ball_rate", "平均ボール支配率", "style", None),
}

CORE_STATS = [
    "goals",
    "goals_against",
    "shots",
    "shots_on_target",
    "shots_against",
    "shots_on_target_against",
    "xg",
    "xga",
    "chance_create",
    "clean_sheet",
    "ball_possession",
]

# xga_non_pkはURLが年度・リーグによって取れない可能性があるため、拡張セットに分離
EXTENDED_STATS = CORE_STATS + ["xga_non_pk"]


TEAM_NAME_MAP = {
    # J1・近年主要
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
    "Ｖ・ファーレン長崎": "長崎",
    "V・ファーレン長崎": "長崎",

    # J2/J3
    "水戸ホーリーホック": "水戸",
    "ジェフユナイテッド千葉": "千葉",
    "ジェフユナイテッド市原": "千葉",
    "ジェフユナイテッド市原・千葉": "千葉",
    "ベガルタ仙台": "仙台",
    "ブラウブリッツ秋田": "秋田",
    "モンテディオ山形": "山形",
    "いわきＦＣ": "いわき",
    "いわきFC": "いわき",
    "栃木ＳＣ": "栃木SC",
    "栃木SC": "栃木SC",
    "栃木シティ": "栃木C",
    "ＲＢ大宮アルディージャ": "大宮",
    "RB大宮アルディージャ": "大宮",
    "大宮アルディージャ": "大宮",
    "ヴァンフォーレ甲府": "甲府",
    "カターレ富山": "富山",
    "藤枝ＭＹＦＣ": "藤枝",
    "藤枝MYFC": "藤枝",
    "徳島ヴォルティス": "徳島",
    "FC今治": "今治",
    "大分トリニータ": "大分",
    "テゲバジャーロ宮崎": "宮崎",
    "ヴァンラーレ八戸": "八戸",
    "福島ユナイテッドＦＣ": "福島",
    "福島ユナイテッドFC": "福島",
    "ザスパクサツ群馬": "群馬",
    "ザスパ群馬": "群馬",
    "ＳＣ相模原": "相模原",
    "SC相模原": "相模原",
    "松本山雅ＦＣ": "松本",
    "松本山雅FC": "松本",
    "ＡＣ長野パルセイロ": "長野",
    "AC長野パルセイロ": "長野",
    "ツエーゲン金沢": "金沢",
    "ＦＣ岐阜": "岐阜",
    "FC岐阜": "岐阜",
    "レイラック滋賀ＦＣ": "滋賀",
    "レイラック滋賀": "滋賀",
    "レイラック滋賀FC": "滋賀",
    "FC大阪": "FC大阪",
    "奈良クラブ": "奈良",
    "ガイナーレ鳥取": "鳥取",
    "レノファ山口ＦＣ": "山口",
    "レノファ山口FC": "山口",
    "カマタマーレ讃岐": "讃岐",
    "愛媛ＦＣ": "愛媛",
    "愛媛FC": "愛媛",
    "高知ユナイテッドＳＣ": "高知",
    "高知ユナイテッドSC": "高知",
    "ギラヴァンツ北九州": "北九州",
    "ロアッソ熊本": "熊本",
    "鹿児島ユナイテッドＦＣ": "鹿児島",
    "鹿児島ユナイテッドFC": "鹿児島",
    "ＦＣ琉球": "琉球",
    "FC琉球": "琉球",
    "アスルクラロ沼津": "沼津",
    "Y．S．C．C．横浜": "YS横浜",
    "Ｙ．Ｓ．Ｃ．Ｃ．横浜": "YS横浜",
    "Ｙ．Ｓ．Ｃ．Ｃ．横浜": "YS横浜",
    "いわてグルージャ盛岡": "岩手",
    "グルージャ盛岡": "岩手",
}

# 例: "1 サンフレッチェ広島 72 GOALS", "1 横浜Ｆ・マリノス 57.6", "1 チーム 57.6%"
ROW_PATTERN = re.compile(
    r"^\s*(\d{1,2})\s+(.+?)\s+(-?[0-9]+(?:\.[0-9]+)?)(?:\s*(?:%|％|[A-Za-z]+))?\s*$"
)


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

    # 1) リストやリンク単位のテキストを拾う。
    # 親要素の巨大テキストは誤検出の原因になるので、最後にセクション単位で補助する。
    for tag in soup.find_all(["li", "a", "tr", "td"]):
        text = " ".join(tag.get_text(" ", strip=True).split())
        if text:
            candidates.append(text)

    # 2) ページ全体テキストから「{label} {season}シーズン」以降のセクションも保険で拾う。
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
        # 説明文を含む行は除外
        if "項目を選択" in text or "シーズンを選択" in text:
            continue
        m = ROW_PATTERN.match(text)
        if not m:
            continue

        rank = int(m.group(1))
        team = m.group(2).strip()
        value = float(m.group(3))

        # ナビや年度選択などの誤検出回避。J3でも通常30チーム未満。
        if rank <= 0 or rank > 30:
            continue
        if len(team) < 2:
            continue
        if team.isdigit():
            continue

        key = (rank, team, value)
        if key in seen:
            continue
        seen.add(key)
        rows.append((rank, team, value))

    rows = sorted(rows, key=lambda x: x[0])

    # 同じ順位が同値で複数ある場合があるので、rankではなくチーム単位で重複除去
    deduped: List[Tuple[int, str, float]] = []
    used = set()
    for rank, team, value in rows:
        team_std = standardize_team_name(team)
        if team_std in used:
            continue
        used.add(team_std)
        deduped.append((rank, team, value))

    return deduped


def collect_stat(league: str, season: int, stat_key: str) -> Tuple[List[Dict[str, object]], Optional[Dict[str, str]]]:
    spec = STAT_SPECS[stat_key]
    url = BASE_URL.format(league=league, season=season, slug=spec.slug)

    try:
        html = fetch_html(url)
        rows = parse_stat_page(html, label=spec.label, season=season)
        if not rows:
            raise RuntimeError("No rows parsed")
    except Exception as e:
        return [], {
            "season": str(season),
            "league": league,
            "stat_key": stat_key,
            "slug": spec.slug,
            "url": url,
            "error": repr(e),
        }

    out = []
    for rank, team, value in rows:
        out.append({
            "season": season,
            "league": league,
            "league_level": LEAGUE_LEVEL[league],
            "team": team,
            "team_std": standardize_team_name(team),
            "stat_key": stat_key,
            "stat_label": spec.label,
            "stat_category": spec.category,
            "higher_is_better": spec.higher_is_better,
            "rank": rank,
            "value": value,
            "source_url": url,
        })
    return out, None


def collect_stats(
    seasons: Iterable[int],
    leagues: Iterable[str],
    stat_keys: Iterable[str],
    sleep_sec: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    scraped_at = datetime.now().isoformat(timespec="seconds")
    all_rows: List[Dict[str, object]] = []
    errors: List[Dict[str, str]] = []

    for season in seasons:
        for league in leagues:
            league = league.lower()
            if league not in LEAGUE_LEVEL:
                raise ValueError(f"Unsupported league: {league}. Use j1, j2, j3.")

            for stat_key in stat_keys:
                if stat_key not in STAT_SPECS:
                    raise ValueError(f"Unknown stat key: {stat_key}")
                spec = STAT_SPECS[stat_key]
                url = BASE_URL.format(league=league, season=season, slug=spec.slug)
                print(f"fetch: {league.upper()} {season} {stat_key} -> {url}")

                rows, err = collect_stat(league, season, stat_key)
                if err is not None:
                    print(f"  ERROR: {err['error']}", file=sys.stderr)
                    errors.append(err)
                else:
                    for row in rows:
                        row["scraped_at"] = scraped_at
                    all_rows.extend(rows)

                time.sleep(sleep_sec)

    long_df = pd.DataFrame(all_rows)
    errors_df = pd.DataFrame(errors)
    return long_df, errors_df


def make_wide_df(long_df: pd.DataFrame, stat_keys: List[str]) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame()

    index_cols = ["season", "league", "league_level", "team_std"]

    # team raw nameは最初のものを採用
    team_df = (
        long_df.sort_values(index_cols + ["stat_key"])
        .groupby(index_cols, as_index=False)["team"]
        .first()
    )

    value_wide = long_df.pivot_table(
        index=index_cols,
        columns="stat_key",
        values="value",
        aggfunc="first",
    ).reset_index()

    rank_wide = long_df.pivot_table(
        index=index_cols,
        columns="stat_key",
        values="rank",
        aggfunc="first",
    ).reset_index()
    rank_wide = rank_wide.rename(columns={k: f"{k}_rank" for k in stat_keys if k in rank_wide.columns})

    wide = team_df.merge(value_wide, on=index_cols, how="outer").merge(rank_wide, on=index_cols, how="outer")

    # 列順を整える
    front_cols = ["season", "league", "league_level", "team", "team_std"]
    value_cols = [k for k in stat_keys if k in wide.columns]
    rank_cols = [f"{k}_rank" for k in stat_keys if f"{k}_rank" in wide.columns]
    other_cols = [c for c in wide.columns if c not in front_cols + value_cols + rank_cols]
    wide = wide[front_cols + value_cols + rank_cols + other_cols]

    return wide.sort_values(["season", "league_level", "team_std"]).reset_index(drop=True)


def add_league_ratios(wide_df: pd.DataFrame, stat_keys: List[str]) -> pd.DataFrame:
    if wide_df.empty:
        return wide_df
    df = wide_df.copy()
    group_cols = ["season", "league"]

    for key in stat_keys:
        if key not in df.columns:
            continue
        avg_col = f"league_avg_{key}"
        ratio_col = f"{key}_ratio"
        df[avg_col] = df.groupby(group_cols)[key].transform("mean")
        df[ratio_col] = df[key] / df[avg_col]

    # 代表的な差分
    if {"goals", "goals_against"}.issubset(df.columns):
        df["goal_diff"] = df["goals"] - df["goals_against"]
    if {"xg", "xga"}.issubset(df.columns):
        df["xg_diff"] = df["xg"] - df["xga"]
    if {"shots", "shots_against"}.issubset(df.columns):
        df["shot_diff"] = df["shots"] - df["shots_against"]
    if {"shots_on_target", "shots_on_target_against"}.issubset(df.columns):
        df["sot_diff"] = df["shots_on_target"] - df["shots_on_target_against"]

    # そのまま前年レーティングのたたき台にできる簡易合成指標。
    # 欠損がある場合は、存在する比率だけで重みを再正規化する。
    attack_weights = {
        "goals_ratio": 0.40,
        "xg_ratio": 0.30,
        "shots_on_target_ratio": 0.20,
        "shots_ratio": 0.05,
        "chance_create_ratio": 0.05,
    }
    defense_bad_weights = {
        "goals_against_ratio": 0.40,
        "xga_ratio": 0.30,
        "shots_on_target_against_ratio": 0.20,
        "shots_against_ratio": 0.10,
    }

    def weighted_score(row: pd.Series, weights: Dict[str, float]) -> float:
        total = 0.0
        w_sum = 0.0
        for col, w in weights.items():
            val = row.get(col, pd.NA)
            if pd.notna(val):
                total += float(val) * w
                w_sum += w
        return total / w_sum if w_sum > 0 else float("nan")

    df["prev_attack_ratio_simple"] = df.apply(lambda r: weighted_score(r, attack_weights), axis=1)
    df["prev_defense_bad_ratio_simple"] = df.apply(lambda r: weighted_score(r, defense_bad_weights), axis=1)

    # 守備の良さとして見たい場合はこちら。数値が高いほど守備が良い。
    df["prev_defense_good_ratio_simple"] = 1.0 / df["prev_defense_bad_ratio_simple"]

    return df


def make_summary(long_df: pd.DataFrame, errors_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame()

    summary = (
        long_df.groupby(["season", "league", "stat_key"], as_index=False)
        .agg(
            rows=("team_std", "count"),
            teams=("team_std", "nunique"),
            min_value=("value", "min"),
            max_value=("value", "max"),
            mean_value=("value", "mean"),
        )
        .sort_values(["season", "league", "stat_key"])
    )

    if errors_df is not None and not errors_df.empty:
        err_count = (
            errors_df.groupby(["season", "league", "stat_key"], as_index=False)
            .size()
            .rename(columns={"size": "error_count"})
        )
        summary = summary.merge(err_count, on=["season", "league", "stat_key"], how="left")
    else:
        summary["error_count"] = 0

    summary["error_count"] = summary["error_count"].fillna(0).astype(int)
    return summary


def write_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--leagues", nargs="+", default=["j1", "j2", "j3"], choices=["j1", "j2", "j3"])
    parser.add_argument(
        "--stats",
        nargs="+",
        default=CORE_STATS,
        help=(
            "収集するstat_key。'core' または 'extended' も指定可。"
            f" 個別指定候補: {', '.join(STAT_SPECS.keys())}"
        ),
    )
    parser.add_argument("--sleep", type=float, default=0.5, help="アクセス間隔。短くしすぎないこと。")
    parser.add_argument("--output-prefix", type=str, default="jleague_team_stats_yearly")
    return parser.parse_args()


def resolve_stat_keys(raw_stats: List[str]) -> List[str]:
    if len(raw_stats) == 1 and raw_stats[0].lower() == "core":
        return CORE_STATS
    if len(raw_stats) == 1 and raw_stats[0].lower() == "extended":
        return EXTENDED_STATS

    out = []
    for s in raw_stats:
        s = s.lower()
        if s == "core":
            out.extend(CORE_STATS)
        elif s == "extended":
            out.extend(EXTENDED_STATS)
        elif s in STAT_SPECS:
            out.append(s)
        else:
            raise ValueError(f"Unknown stat key: {s}")

    # 順序を保って重複除去
    deduped = []
    for s in out:
        if s not in deduped:
            deduped.append(s)
    return deduped


def main() -> None:
    args = parse_args()
    stat_keys = resolve_stat_keys(args.stats)
    seasons = list(range(args.start_year, args.end_year + 1))

    print("==============================")
    print("Jリーグ チームスタッツ収集")
    print("==============================")
    print("seasons:", seasons)
    print("leagues:", args.leagues)
    print("stats:", stat_keys)

    long_df, errors_df = collect_stats(
        seasons=seasons,
        leagues=args.leagues,
        stat_keys=stat_keys,
        sleep_sec=args.sleep,
    )

    wide_df = make_wide_df(long_df, stat_keys)
    wide_df = add_league_ratios(wide_df, stat_keys)
    summary_df = make_summary(long_df, errors_df)

    suffix = f"{args.start_year}_{args.end_year}"
    prefix = Path(args.output_prefix)

    long_path = prefix.with_name(f"{prefix.name}_long_{suffix}.csv")
    wide_path = prefix.with_name(f"{prefix.name}_wide_{suffix}.csv")
    summary_path = prefix.with_name(f"{prefix.name}_summary_{suffix}.csv")
    errors_path = prefix.with_name(f"{prefix.name}_errors_{suffix}.csv")

    write_df(long_df, long_path)
    write_df(wide_df, wide_path)
    write_df(summary_df, summary_path)
    if errors_df.empty:
        errors_df = pd.DataFrame(columns=["season", "league", "stat_key", "slug", "url", "error"])
    write_df(errors_df, errors_path)

    print("\n==============================")
    print("収集完了")
    print("==============================")
    print("long rows:", len(long_df))
    print("wide rows:", len(wide_df))
    print("errors:", len(errors_df))
    print("long:", long_path.resolve())
    print("wide:", wide_path.resolve())
    print("summary:", summary_path.resolve())
    print("errors:", errors_path.resolve())

    if len(errors_df) > 0:
        print("\n一部取得できなかったページがあります。errors CSVを確認してください。")
        print("xga_non_pkなどは年度・リーグによって取得できない可能性があります。")


if __name__ == "__main__":
    main()
