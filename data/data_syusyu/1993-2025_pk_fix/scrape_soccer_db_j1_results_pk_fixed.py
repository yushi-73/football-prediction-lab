import re
import time
import requests
import pandas as pd
import numpy as np
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

OUTPUT_MATCH_CSV = OUTPUT_DIR / "j1_historical_results_1993_2025_pk_fixed.csv"
OUTPUT_YEAR_COUNT_CSV = OUTPUT_DIR / "j1_historical_year_counts_pk_fixed.csv"
OUTPUT_H2H_CSV = OUTPUT_DIR / "j1_historical_headtohead_summary_pk_fixed.csv"

REQUEST_INTERVAL_SEC = 1.5

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


# =========================
# 基本処理
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
    Soccer D.B. のPK表記を処理する。

    想定表記:
        "1"       -> regular=1, pk=NaN
        "1 (4)"   -> regular=1, pk=4
        "1(4)"    -> regular=1, pk=4
        "(3) 1"   -> regular=1, pk=3
        "(3)1"    -> regular=1, pk=3

    戻り値:
        regular_goal, pk_score
    """
    s = clean_text(score_text)
    s = re.sub(r"\s+", "", s)

    # 例: 1(4)
    m = re.fullmatch(r"(?P<goal>\d+)\((?P<pk>\d+)\)", s)
    if m:
        return int(m.group("goal")), int(m.group("pk"))

    # 例: (3)1
    m = re.fullmatch(r"\((?P<pk>\d+)\)(?P<goal>\d+)", s)
    if m:
        return int(m.group("goal")), int(m.group("pk"))

    # 例: 1
    m = re.fullmatch(r"\d+", s)
    if m:
        return int(s), np.nan

    raise ValueError(f"スコアを解析できません: {score_text}")


def judge_results(home_goal, away_goal, home_pk=np.nan, away_pk=np.nan):
    """
    通常スコアとPKスコアから結果を作る。

    normal_result:
        90分/延長を含む表示上のスコアだけで見た結果。
        同点ならD。

    official_result:
        PKがある場合はPK勝敗まで反映。
        相性分析で「公式上の勝ち負け」を見る時に使える。
    """
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
            # 通常あり得ないが、壊れたデータ対策
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
# 1年分を取得・解析
# =========================

def parse_year(year):
    url = BASE_URL.format(year=year)
    print(f"取得中: {year} {url}")

    html = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)

    # 重要:
    # PK表記には
    #   1 (4)
    #   (3) 1
    # のようなパターンがある。
    #
    # 以前のコードではこれを拾えず、PK戦の1試合をチーム名側に飲み込んで、
    # 次の試合とくっつけてしまうことがあった。
    score_pattern = r"(?:\d+\s*\(\d+\)|\(\d+\)\s*\d+|\d+)"

    # 次の試合日付をまたがないように home 側を制限する
    # away は「観客:」の直前まで。
    pattern = re.compile(
        rf"(?P<date>\d{{4}}\.\d{{2}}\.\d{{2}})\s+"
        rf"(?P<home>(?:(?!\d{{4}}\.\d{{2}}\.\d{{2}}).)+?)\s+"
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

        home_goal, home_pk = parse_score_with_pk(match.group("home_score"))
        away_goal, away_pk = parse_score_with_pk(match.group("away_score"))

        result_info = judge_results(
            home_goal=home_goal,
            away_goal=away_goal,
            home_pk=home_pk,
            away_pk=away_pk,
        )

        home = standardize_team_name(home_raw)
        away = standardize_team_name(away_raw)

        rows.append({
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
            "source_url": url,
        })

    print(f"{year}: {len(rows)}試合取得 / PK戦 {sum(r['has_pk'] for r in rows)}試合")

    if len(rows) == 0:
        print(f"警告: {year}年は0試合でした。HTML構造が違う可能性があります。")

    return rows


# =========================
# 対戦成績集計
# =========================

def make_headtohead_summary(match_df, result_mode="official"):
    """
    result_mode:
        "official" -> PK勝敗まで含めてW/L判定
        "normal"   -> 通常スコアが同点ならD扱い

    相性分析では、まず official を使うのが自然。
    得点力モデルでは home_goal / away_goal を使う。
    """
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


# =========================
# チェック
# =========================

def validate_scraped_data(df):
    """
    正規表現が崩れて、チーム名に別試合の情報を飲み込んでいないか確認する。
    """
    suspicious = df[
        (df["home_raw"].astype(str).str.len() > 25)
        | (df["away_raw"].astype(str).str.len() > 25)
        | (df["home_raw"].astype(str).str.contains("Result|Referee|観客:", regex=True, na=False))
        | (df["away_raw"].astype(str).str.contains("Result|Referee|観客:", regex=True, na=False))
    ].copy()

    print("\n==============================")
    print("データ検査")
    print("==============================")
    print("総試合数:", len(df))
    print("PK戦数:", int(df["has_pk"].sum()))
    print("怪しい行数:", len(suspicious))

    if len(suspicious) > 0:
        print("\n怪しい行の例:")
        print(
            suspicious[
                ["year", "date", "home_raw", "home_goal", "home_pk", "away_goal", "away_pk", "away_raw"]
            ].head(20).to_string(index=False)
        )
        print("\n警告: 怪しい行があります。正規表現を再調整してください。")
    else:
        print("怪しい行は見つかりませんでした。")


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
        subset=["date", "home", "away", "home_goal", "away_goal", "home_pk", "away_pk"]
    ).sort_values(["date", "home", "away"]).reset_index(drop=True)

    validate_scraped_data(df)

    df.to_csv(OUTPUT_MATCH_CSV, index=False, encoding="utf-8-sig")

    year_counts = (
        df.groupby("year")
        .agg(
            matches=("date", "count"),
            pk_matches=("has_pk", "sum"),
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
