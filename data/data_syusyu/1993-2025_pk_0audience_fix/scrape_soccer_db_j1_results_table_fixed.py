import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

# =========================
# 設定
# =========================

START_YEAR = 1993
END_YEAR = 2025

BASE_URL = "https://soccer-db.net/competition/results/1001/{year}"

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_MATCH_CSV = OUTPUT_DIR / "j1_historical_results_1993_2025_table_fixed.csv"
OUTPUT_YEAR_COUNT_CSV = OUTPUT_DIR / "j1_historical_year_counts_table_fixed.csv"
OUTPUT_H2H_CSV = OUTPUT_DIR / "j1_historical_headtohead_summary_table_fixed.csv"
OUTPUT_SUSPICIOUS_CSV = OUTPUT_DIR / "j1_historical_suspicious_rows_table_fixed.csv"

REQUEST_INTERVAL_SEC = 1.5

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

DATE_RE = re.compile(r"\d{4}\.\d{2}\.\d{2}")
SCORE_RE = re.compile(r"(?:\d+\s*\(\d+\)|\(\d+\)\s*\d+|\d+)")
SCORE_FULL_RE = re.compile(r"^(?:\d+\s*\(\d+\)|\(\d+\)\s*\d+|\d+)$")


# =========================
# 基本処理
# =========================

def clean_text(text):
    return (
        str(text)
        .replace("\u3000", " ")
        .replace("\xa0", " ")
        .replace("\n", " ")
        .replace("\r", " ")
        .strip()
    )


def normalize_spaces(text):
    return re.sub(r"\s+", " ", clean_text(text)).strip()


def standardize_team_name(name):
    name = normalize_spaces(name)

    name_map = {
        "京都パープルサンガ": "京都サンガF.C.",
        "京都サンガ": "京都サンガF.C.",

        "横浜マリノス": "横浜Ｆ・マリノス",

        "ヴェルディ川崎": "東京ヴェルディ",
        "東京ヴェルディ１９６９": "東京ヴェルディ",
        "東京ヴェルディ1969": "東京ヴェルディ",

        "ジェフユナイテッド市原": "ジェフユナイテッド千葉",
        "ジェフユナイテッド市原・千葉": "ジェフユナイテッド千葉",

        "コンサドーレ札幌": "北海道コンサドーレ札幌",
    }

    return name_map.get(name, name)


def fetch_html(url):
    response = requests.get(url, headers=HEADERS, timeout=20)
    response.raise_for_status()
    response.encoding = response.apparent_encoding
    return response.text


# =========================
# スコア処理
# =========================

def parse_score_with_pk(score_text):
    """
    例:
      1       -> goal=1, pk=NaN
      1 (4)   -> goal=1, pk=4
      1(4)    -> goal=1, pk=4
      (3) 1   -> goal=1, pk=3
      (3)1    -> goal=1, pk=3
    """
    s = normalize_spaces(score_text)
    s = re.sub(r"\s+", "", s)

    m = re.fullmatch(r"(?P<goal>\d+)\((?P<pk>\d+)\)", s)
    if m:
        return int(m.group("goal")), int(m.group("pk"))

    m = re.fullmatch(r"\((?P<pk>\d+)\)(?P<goal>\d+)", s)
    if m:
        return int(m.group("goal")), int(m.group("pk"))

    m = re.fullmatch(r"\d+", s)
    if m:
        return int(s), np.nan

    raise ValueError(f"スコアを解析できません: {score_text}")


def judge_results(home_goal, away_goal, home_pk=np.nan, away_pk=np.nan):
    has_pk = pd.notna(home_pk) or pd.notna(away_pk)

    if home_goal > away_goal:
        normal_home_result = "W"
        normal_away_result = "L"
    elif home_goal < away_goal:
        normal_home_result = "L"
        normal_away_result = "W"
    else:
        normal_home_result = "D"
        normal_away_result = "D"

    if has_pk and home_goal == away_goal:
        if home_pk > away_pk:
            official_home_result = "W"
            official_away_result = "L"
            pk_winner = "home"
        elif home_pk < away_pk:
            official_home_result = "L"
            official_away_result = "W"
            pk_winner = "away"
        else:
            official_home_result = "D"
            official_away_result = "D"
            pk_winner = "draw"
    else:
        official_home_result = normal_home_result
        official_away_result = normal_away_result
        pk_winner = ""

    return {
        "has_pk": bool(has_pk),
        "pk_winner": pk_winner,
        "normal_home_result": normal_home_result,
        "normal_away_result": normal_away_result,
        "official_home_result": official_home_result,
        "official_away_result": official_away_result,
    }


# =========================
# 行・セル解析
# =========================

def is_noise_cell(cell):
    """
    Soccer D.B. の行内に混じる S/Y/R などのアイコン文字を除外する。
    """
    c = normalize_spaces(cell)
    if c == "":
        return True

    # S S S S や Y Y など
    if re.fullmatch(r"(?:[A-Z]\s*){1,20}", c):
        return True

    return False


def is_meta_cell(cell):
    c = normalize_spaces(cell)
    meta_words = [
        "Referee:",
        "観客:",
        "Attendance:",
        "Stadium:",
        "天候:",
        "Weather:",
        "気温:",
        "湿度:",
    ]
    return any(word in c for word in meta_words)


def strip_tail_noise(text):
    """
    テキスト解析用。
    away_raw に S S S ... Referee: ... が入った場合に、チーム名部分だけ残す。
    """
    s = normalize_spaces(text)

    # メタ情報以降を削除
    s = re.split(r"\s+(?:Referee:|観客:|Attendance:|Stadium:|天候:|Weather:|気温:|湿度:)", s)[0]

    # S S S S や Y Y 以降を削除
    s = re.split(r"\s+(?:[A-Z]\s+){2,}", s)[0]

    return normalize_spaces(s)


def make_row(year, date_text, home_raw, away_raw, home_score_text, away_score_text, url, parser):
    home_goal, home_pk = parse_score_with_pk(home_score_text)
    away_goal, away_pk = parse_score_with_pk(away_score_text)

    result_info = judge_results(
        home_goal=home_goal,
        away_goal=away_goal,
        home_pk=home_pk,
        away_pk=away_pk,
    )

    home_raw = strip_tail_noise(home_raw)
    away_raw = strip_tail_noise(away_raw)

    home = standardize_team_name(home_raw)
    away = standardize_team_name(away_raw)

    return {
        "year": year,
        "date": pd.to_datetime(date_text, format="%Y.%m.%d"),
        "home": home,
        "away": away,
        "home_goal": home_goal,
        "away_goal": away_goal,
        "home_pk": home_pk,
        "away_pk": away_pk,
        "has_pk": result_info["has_pk"],
        "pk_winner": result_info["pk_winner"],
        "normal_home_result": result_info["normal_home_result"],
        "normal_away_result": result_info["normal_away_result"],
        "official_home_result": result_info["official_home_result"],
        "official_away_result": result_info["official_away_result"],
        "home_raw": home_raw,
        "away_raw": away_raw,
        "parser": parser,
        "source_url": url,
    }


def parse_row_from_cells(cells, year, url):
    """
    <tr> の <td> を使う優先パーサー。
    ページ全体テキストより安全で、別試合を飲み込みにくい。
    """
    cells = [normalize_spaces(c) for c in cells if normalize_spaces(c) != ""]
    if not cells:
        return None

    date_i = None
    for i, c in enumerate(cells):
        if DATE_RE.fullmatch(c):
            date_i = i
            break

    if date_i is None:
        return None

    result_i = None
    for i, c in enumerate(cells):
        if c == "Result" or re.fullmatch(r"[WDL]\s+Result", c):
            result_i = i
            break

    if result_i is None:
        return None

    home_score_i = None
    for i in range(result_i - 1, date_i, -1):
        if SCORE_FULL_RE.fullmatch(cells[i]):
            home_score_i = i
            break

    away_score_i = None
    for i in range(result_i + 1, len(cells)):
        if SCORE_FULL_RE.fullmatch(cells[i]):
            away_score_i = i
            break

    if home_score_i is None or away_score_i is None:
        return None

    home_parts = []
    for c in cells[date_i + 1:home_score_i]:
        if not is_noise_cell(c) and not is_meta_cell(c):
            home_parts.append(c)

    away_parts = []
    for c in cells[away_score_i + 1:]:
        if is_meta_cell(c):
            break

        # チーム名を拾った後にアイコン文字が来たらそこで終了
        if away_parts and is_noise_cell(c):
            break

        if not is_noise_cell(c):
            away_parts.append(c)

        # 通常、away teamは1セルで完結する
        if away_parts:
            break

    if not home_parts or not away_parts:
        return None

    date_text = cells[date_i]
    home_raw = " ".join(home_parts)
    away_raw = " ".join(away_parts)
    home_score = cells[home_score_i]
    away_score = cells[away_score_i]

    return make_row(
        year=year,
        date_text=date_text,
        home_raw=home_raw,
        away_raw=away_raw,
        home_score_text=home_score,
        away_score_text=away_score,
        url=url,
        parser="cells",
    )


def parse_block_from_text(block, year, url):
    """
    fallback用。
    ページ全体ではなく、1日付ブロックごとに解析する。
    これで別試合の大量飲み込みを防ぐ。
    """
    block = normalize_spaces(block)

    pattern = re.compile(
        rf"^(?P<date>\d{{4}}\.\d{{2}}\.\d{{2}})\s+"
        rf"(?P<home>.+?)\s+"
        rf"(?P<home_score>{SCORE_RE.pattern})\s+"
        rf"(?:[WDL]\s+)?Result\s+"
        rf"(?P<away_score>{SCORE_RE.pattern})\s+"
        rf"(?P<away>.+?)"
        rf"(?=\s+(?:Referee:|観客:|Attendance:|Stadium:|天候:|Weather:|気温:|湿度:)|$)"
    )

    m = pattern.search(block)
    if not m:
        return None

    return make_row(
        year=year,
        date_text=m.group("date"),
        home_raw=m.group("home"),
        away_raw=m.group("away"),
        home_score_text=m.group("home_score"),
        away_score_text=m.group("away_score"),
        url=url,
        parser="text_block",
    )


# =========================
# 1年分取得
# =========================

def parse_year(year):
    url = BASE_URL.format(year=year)
    print(f"取得中: {year} {url}")

    html = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")

    rows = []
    seen = set()

    # 1. まず table row / list row 単位で解析
    candidate_elements = soup.select("tr")
    for elem in candidate_elements:
        cells = [c.get_text(" ", strip=True) for c in elem.find_all(["td", "th"])]
        row = parse_row_from_cells(cells, year, url)

        if row is None:
            continue

        key = (row["date"], row["home_raw"], row["away_raw"], row["home_goal"], row["away_goal"], row["home_pk"], row["away_pk"])
        if key not in seen:
            rows.append(row)
            seen.add(key)

    # 2. trで十分取れなかった年だけ、日付ブロックfallback
    if len(rows) == 0:
        text = soup.get_text(" ", strip=True)
        text = normalize_spaces(text)

        blocks = re.split(r"(?=\d{4}\.\d{2}\.\d{2}\s+)", text)

        for block in blocks:
            if not DATE_RE.match(block):
                continue

            row = parse_block_from_text(block, year, url)
            if row is None:
                continue

            key = (row["date"], row["home_raw"], row["away_raw"], row["home_goal"], row["away_goal"], row["home_pk"], row["away_pk"])
            if key not in seen:
                rows.append(row)
                seen.add(key)

    print(f"{year}: {len(rows)}試合取得 / PK戦 {sum(r['has_pk'] for r in rows)}試合")

    if len(rows) == 0:
        print(f"警告: {year}年は0試合でした。HTML構造が違う可能性があります。")

    return rows


# =========================
# 集計
# =========================

def make_headtohead_summary(match_df, result_mode="official"):
    if result_mode not in ["official", "normal"]:
        raise ValueError("result_mode は 'official' か 'normal' にしてください。")

    records = []

    for _, row in match_df.iterrows():
        home = row["home"]
        away = row["away"]
        hg = int(row["home_goal"])
        ag = int(row["away_goal"])

        if result_mode == "official":
            home_result = row["official_home_result"]
            away_result = row["official_away_result"]
        else:
            home_result = row["normal_home_result"]
            away_result = row["normal_away_result"]

        records.append({
            "team": home,
            "opponent": away,
            "is_home": 1,
            "gf": hg,
            "ga": ag,
            "result": home_result,
        })

        records.append({
            "team": away,
            "opponent": home,
            "is_home": 0,
            "gf": ag,
            "ga": hg,
            "result": away_result,
        })

    long_df = pd.DataFrame(records)

    summary = (
        long_df
        .groupby(["team", "opponent"], as_index=False)
        .agg(
            matches=("result", "count"),
            wins=("result", lambda s: (s == "W").sum()),
            draws=("result", lambda s: (s == "D").sum()),
            losses=("result", lambda s: (s == "L").sum()),
            gf=("gf", "sum"),
            ga=("ga", "sum"),
            home_matches=("is_home", "sum"),
        )
    )

    summary["gd"] = summary["gf"] - summary["ga"]
    summary["win_rate"] = summary["wins"] / summary["matches"]
    summary["draw_rate"] = summary["draws"] / summary["matches"]
    summary["loss_rate"] = summary["losses"] / summary["matches"]
    summary["points"] = summary["wins"] * 3 + summary["draws"]
    summary["points_per_match"] = summary["points"] / summary["matches"]

    summary = summary.sort_values(
        ["team", "matches", "points_per_match"],
        ascending=[True, False, False]
    )

    return summary


def validate_scraped_data(df):
    suspicious = df[
        (df["home_raw"].astype(str).str.contains(r"\d{4}\.\d{2}\.\d{2}|Result|Referee|観客:", regex=True, na=False))
        | (df["away_raw"].astype(str).str.contains(r"\d{4}\.\d{2}\.\d{2}|Result|Referee|観客:", regex=True, na=False))
        | (df["home_raw"].astype(str).str.len() > 35)
        | (df["away_raw"].astype(str).str.len() > 35)
    ].copy()

    print("\n==============================")
    print("データ検査")
    print("==============================")
    print("総試合数:", len(df))
    print("PK戦数:", int(df["has_pk"].sum()))
    print("怪しい行数:", len(suspicious))

    if len(suspicious) > 0:
        suspicious.to_csv(OUTPUT_SUSPICIOUS_CSV, index=False, encoding="utf-8-sig")
        print("怪しい行を保存しました:", OUTPUT_SUSPICIOUS_CSV)
        print(
            suspicious[
                ["year", "date", "home_raw", "home_goal", "home_pk", "away_goal", "away_pk", "away_raw", "parser"]
            ].head(20).to_string(index=False)
        )
    else:
        print("怪しい行は見つかりませんでした。")

    return suspicious


# =========================
# メイン
# =========================

def main():
    all_rows = []

    for year in range(START_YEAR, END_YEAR + 1):
        try:
            rows = parse_year(year)
            all_rows.extend(rows)
        except Exception as e:
            print(f"{year}年でエラー:", e)

        time.sleep(REQUEST_INTERVAL_SEC)

    if not all_rows:
        print("取得できた試合がありません。処理を終了します。")
        return

    df = pd.DataFrame(all_rows)

    df = df.drop_duplicates(
        subset=["date", "home", "away", "home_goal", "away_goal", "home_pk", "away_pk"]
    ).sort_values(["date", "home", "away"]).reset_index(drop=True)

    validate_scraped_data(df)

    df.to_csv(OUTPUT_MATCH_CSV, index=False, encoding="utf-8-sig")

    year_counts = (
        df.groupby("year")
        .agg(
            matches=("date", "count"),
            pk_matches=("has_pk", "sum"),
            cell_parser=("parser", lambda s: (s == "cells").sum()),
            text_parser=("parser", lambda s: (s == "text_block").sum()),
        )
        .reset_index()
        .sort_values("year")
    )
    year_counts.to_csv(OUTPUT_YEAR_COUNT_CSV, index=False, encoding="utf-8-sig")

    h2h_summary = make_headtohead_summary(df, result_mode="official")
    h2h_summary.to_csv(OUTPUT_H2H_CSV, index=False, encoding="utf-8-sig")

    print("\n==============================")
    print("保存完了")
    print("==============================")
    print("試合データ:", OUTPUT_MATCH_CSV)
    print("年度別試合数:", OUTPUT_YEAR_COUNT_CSV)
    print("対戦成績集計:", OUTPUT_H2H_CSV)

    print("\n==============================")
    print("年度別試合数")
    print("==============================")
    print(year_counts.to_string(index=False))

    print("\n==============================")
    print("先頭5行")
    print("==============================")
    print(df.head().to_string(index=False))

    print("\n==============================")
    print("末尾5行")
    print("==============================")
    print(df.tail().to_string(index=False))


if __name__ == "__main__":
    main()
