import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path

# =========================
# 設定
# =========================

START_YEAR = 1993
END_YEAR = 2025

BASE_URL = "https://soccer-db.net/competition/results/1001/{year}"

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_MATCH_CSV = OUTPUT_DIR / "j1_historical_results_1993_2025.csv"
OUTPUT_YEAR_COUNT_CSV = OUTPUT_DIR / "j1_historical_year_counts.csv"
OUTPUT_H2H_CSV = OUTPUT_DIR / "j1_historical_headtohead_summary.csv"

REQUEST_INTERVAL_SEC = 1.5

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


# =========================
# チーム名処理
# =========================

def clean_text(text):
    return (
        str(text)
        .replace("\u3000", " ")
        .replace("\xa0", " ")
        .strip()
    )


def standardize_team_name(name):
    """
    チーム名の表記ゆれをある程度統一する。
    raw名もCSVに残すので、ここは後から調整可能。
    """
    name = clean_text(name)

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
        "北海道コンサドーレ札幌": "北海道コンサドーレ札幌",
    }

    return name_map.get(name, name)


def normal_score(score_text):
    """
    PK表記などが混じっても通常スコア部分だけ取り出す。
    例:
      "1"      -> 1
      "1(4)"   -> 1
      "(3)1"   -> 1
    """
    score_text = clean_text(score_text)

    nums = re.findall(r"\d+", score_text)

    if not nums:
        raise ValueError(f"スコアを数値化できません: {score_text}")

    # "1(4)" の場合は最初の 1
    # "(3)1" の場合は最後の 1
    if score_text.startswith("("):
        return int(nums[-1])
    else:
        return int(nums[0])


# =========================
# 1年分を取得・解析
# =========================

def fetch_html(url):
    response = requests.get(url, headers=HEADERS, timeout=20)
    response.raise_for_status()

    # 文字化け対策
    response.encoding = response.apparent_encoding

    return response.text


def parse_year(year):
    url = BASE_URL.format(year=year)
    print(f"取得中: {year} {url}")

    html = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")

    # ページ全体をテキスト化
    text = soup.get_text(" ", strip=True)

    # Soccer D.B. の試合結果形式:
    # 2025.02.14 ガンバ大阪 2 Result 5 セレッソ大阪 観客:34,860 ...
    score_pattern = r"(?:\(\d+\))?\d+(?:\(\d+\))?"

    pattern = re.compile(
        rf"(?P<date>\d{{4}}\.\d{{2}}\.\d{{2}})\s+"
        rf"(?P<home>.+?)\s+"
        rf"(?P<home_score>{score_pattern})\s+"
        rf"(?:W\s+|D\s+|L\s+)?Result\s+"
        rf"(?P<away_score>{score_pattern})\s+"
        rf"(?P<away>.+?)\s+"
        rf"観客:",
        re.DOTALL
    )

    rows = []

    for match in pattern.finditer(text):
        date_text = match.group("date")
        home_raw = clean_text(match.group("home"))
        away_raw = clean_text(match.group("away"))

        home_goal = normal_score(match.group("home_score"))
        away_goal = normal_score(match.group("away_score"))

        home = standardize_team_name(home_raw)
        away = standardize_team_name(away_raw)

        rows.append({
            "year": year,
            "date": pd.to_datetime(date_text, format="%Y.%m.%d"),
            "home": home,
            "away": away,
            "home_goal": home_goal,
            "away_goal": away_goal,
            "home_raw": home_raw,
            "away_raw": away_raw,
            "source_url": url,
        })

    print(f"{year}: {len(rows)}試合取得")

    if len(rows) == 0:
        print(f"警告: {year}年は0試合でした。HTML構造が違う可能性があります。")

    return rows


# =========================
# 対戦成績集計
# =========================

def make_headtohead_summary(match_df):
    records = []

    for _, row in match_df.iterrows():
        home = row["home"]
        away = row["away"]
        hg = row["home_goal"]
        ag = row["away_goal"]

        # ホームチーム視点
        if hg > ag:
            home_result = "W"
            away_result = "L"
        elif hg < ag:
            home_result = "L"
            away_result = "W"
        else:
            home_result = "D"
            away_result = "D"

        records.append({
            "team": home,
            "opponent": away,
            "is_home": 1,
            "gf": hg,
            "ga": ag,
            "result": home_result,
        })

        # アウェイチーム視点
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


# =========================
# メイン処理
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
        subset=["date", "home", "away", "home_goal", "away_goal"]
    ).sort_values(["date", "home", "away"]).reset_index(drop=True)

    df.to_csv(OUTPUT_MATCH_CSV, index=False, encoding="utf-8-sig")

    year_counts = (
        df.groupby("year")
        .size()
        .reset_index(name="matches")
        .sort_values("year")
    )

    year_counts.to_csv(OUTPUT_YEAR_COUNT_CSV, index=False, encoding="utf-8-sig")

    h2h_summary = make_headtohead_summary(df)
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
    print(year_counts)

    print("\n==============================")
    print("先頭5行")
    print("==============================")
    print(df.head())

    print("\n==============================")
    print("末尾5行")
    print("==============================")
    print(df.tail())


if __name__ == "__main__":
    main()