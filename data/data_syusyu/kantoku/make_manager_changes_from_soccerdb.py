import re
import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


# ============================================================
# Soccer D.B. から試合ごとの監督名を取得し、
# 監督交代が反映された最初の試合日を推定する
# ------------------------------------------------------------
# 出力:
#   soccerdb_match_managers.csv
#   manager_changes_inferred.csv
# ============================================================


BASE_URL = "https://soccer-db.net"

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_MATCH_MANAGERS_CSV = BASE_DIR / "soccerdb_match_managers.csv"
OUTPUT_MANAGER_CHANGES_CSV = BASE_DIR / "manager_changes_inferred.csv"

# 必要に応じて変更
COMPETITIONS = {
    1001: "J1",
    1002: "J2",
}

YEARS = [2023, 2024, 2025]

SLEEP_SECONDS = 0.4

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}


# =========================
# チーム名標準化
# =========================

def standardize_team_name(name):
    name = (
        str(name)
        .replace("【公式】", "")
        .replace("\u3000", " ")
        .replace("\xa0", " ")
        .strip()
    )

    name_map = {
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
        "V・ファーレン長崎": "長崎",
        "Ｖ・ファーレン長崎": "長崎",
        "レノファ山口ＦＣ": "山口",
        "レノファ山口FC": "山口",
        "ブラウブリッツ秋田": "秋田",
        "ロアッソ熊本": "熊本",
        "いわきＦＣ": "いわき",
        "いわきFC": "いわき",
        "ＦＣ今治": "今治",
        "FC今治": "今治",
        "藤枝ＭＹＦＣ": "藤枝",
        "藤枝MYFC": "藤枝",
        "カターレ富山": "富山",
        "水戸ホーリーホック": "水戸",
        "愛媛ＦＣ": "愛媛",
        "愛媛FC": "愛媛",
    }

    return name_map.get(name, name)


# =========================
# HTML取得
# =========================

def fetch_soup(url):
    res = requests.get(url, headers=HEADERS, timeout=20)
    res.raise_for_status()
    return BeautifulSoup(res.text, "html.parser")


# =========================
# 試合一覧ページからResultリンクを取得
# =========================

def get_result_links(competition_id, year):
    url = f"{BASE_URL}/competition/results/{competition_id}/{year}"
    soup = fetch_soup(url)

    links = []

    for a in soup.find_all("a", href=True):
        href = str(a.get("href", "")).strip()
        text = a.get_text(strip=True)

        # Soccer D.B. の Result リンクを拾う
        if "Result" not in text:
            continue

        if "/result/" not in href:
            continue

        full_url = urljoin(BASE_URL, href)

        m = re.search(r"/result/(\d+)", href)
        if not m:
            continue

        match_id = m.group(1)

        links.append({
            "competition_id": competition_id,
            "season": year,
            "match_id": match_id,
            "match_url": full_url,
        })

    unique = {}
    for item in links:
        unique[item["match_id"]] = item

    result = list(unique.values())

    print(f"  Resultリンク検出数: {len(result)}")

    return result


# =========================
# 試合詳細ページの基本情報を抽出
# =========================

def extract_match_basic_info_from_text(lines):
    """
    Soccer D.B.の詳細ページから、
    日付・ホーム・アウェイ・スコアをざっくり抽出する。

    試合詳細ページでは、本文中に
      25.02.14
      明治安田 Ｊ１リーグ
      ガンバ大阪
      2
      ...
      5
      セレッソ大阪
    のような並びが出るため、完全な構造依存ではなく補助的に使う。
    """

    date = None

    for line in lines:
        m = re.search(r"(\d{2})\.(\d{2})\.(\d{2})", line)
        if m:
            yy, mm, dd = m.groups()
            year = 2000 + int(yy)
            date = f"{year:04d}-{int(mm):02d}-{int(dd):02d}"
            break

    return date


def clean_manager_name(line):
    line = str(line).strip()
    line = re.sub(r"^\(?\d+\)?", "", line).strip()
    line = re.sub(r"^[-–—:：\s]+", "", line).strip()

    # 明らかなヘッダーは除外
    if line in {"nat.age name", "name", "nat", "age"}:
        return ""

    # 空や数字だけは除外
    if not line:
        return ""
    if re.fullmatch(r"[\d()\s]+", line):
        return ""

    return line


def parse_head_coaches(lines):
    """
    Head Coaches セクションから2人の監督名を抽出する。
    1人目をホーム監督、2人目をアウェイ監督として扱う。
    """

    try:
        start_idx = lines.index("Head Coaches")
    except ValueError:
        return None, None

    coaches = []

    for line in lines[start_idx + 1:]:
        if line in {"Stats", "Results", "直近の結果"}:
            break

        cleaned = clean_manager_name(line)

        if not cleaned:
            continue

        # 「nat.age name」は除外済み。
        # 監督名はリンクテキストとして出ることが多い。
        coaches.append(cleaned)

        if len(coaches) >= 2:
            break

    if len(coaches) < 2:
        return None, None

    return coaches[0], coaches[1]


def parse_score_and_teams_from_title(title):
    """
    title例:
    Soccer D.B. : 2025 明治安田 Ｊ１リーグ 25/02/14 ガンバ大阪 - セレッソ大阪 試合結果,スタメン,フォーメーション
    タイトルから日付・チームだけ補助的に取る。
    得点は本文側から取るのが難しい場合があるので、ここではチーム名中心。
    """

    title = str(title)

    date = None
    home = None
    away = None

    m = re.search(r"(\d{2})/(\d{2})/(\d{2})\s+(.+?)\s+-\s+(.+?)\s+試合結果", title)
    if m:
        yy, mm, dd, home, away = m.groups()
        date = f"{2000 + int(yy):04d}-{int(mm):02d}-{int(dd):02d}"
        home = standardize_team_name(home)
        away = standardize_team_name(away)

    return date, home, away


def parse_match_detail(match_url):
    soup = fetch_soup(match_url)

    title = soup.title.get_text(strip=True) if soup.title else ""
    title_date, title_home, title_away = parse_score_and_teams_from_title(title)

    text = soup.get_text("\n", strip=True)
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    text_date = extract_match_basic_info_from_text(lines)
    home_manager, away_manager = parse_head_coaches(lines)

    return {
        "date": title_date or text_date,
        "home": title_home,
        "away": title_away,
        "home_manager": home_manager,
        "away_manager": away_manager,
        "page_title": title,
    }


# =========================
# 試合ごとの監督名CSVを作成
# =========================

def build_match_managers():
    rows = []

    for competition_id, competition_name in COMPETITIONS.items():
        for year in YEARS:
            print(f"\n{competition_name} {year}: 試合リンク取得中...")
            links = get_result_links(competition_id, year)
            print(f"  {len(links)}試合を取得")

            for i, item in enumerate(links, start=1):
                url = item["match_url"]

                try:
                    info = parse_match_detail(url)
                except Exception as e:
                    print(f"  [WARN] {url} の取得に失敗: {e}")
                    continue

                row = {
                    "competition": competition_name,
                    "competition_id": competition_id,
                    "season": year,
                    "match_id": item["match_id"],
                    "match_url": url,
                    **info,
                }

                rows.append(row)

                if i % 50 == 0:
                    print(f"  {i}/{len(links)}試合完了")

                time.sleep(SLEEP_SECONDS)

    df = pd.DataFrame(rows)

    # チーム名を念のため標準化
    for col in ["home", "away"]:
        if col in df.columns:
            df[col] = df[col].apply(standardize_team_name)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["competition", "season", "date", "match_id"]).reset_index(drop=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    return df


# =========================
# 監督交代を推定
# =========================

def build_team_manager_timeline(match_managers_df):
    rows = []

    for row in match_managers_df.itertuples(index=False):
        row_dict = row._asdict()

        rows.append({
            "competition": row_dict["competition"],
            "season": row_dict["season"],
            "date": row_dict["date"],
            "team": row_dict["home"],
            "opponent": row_dict["away"],
            "is_home": True,
            "manager": row_dict["home_manager"],
            "match_id": row_dict["match_id"],
            "match_url": row_dict["match_url"],
        })

        rows.append({
            "competition": row_dict["competition"],
            "season": row_dict["season"],
            "date": row_dict["date"],
            "team": row_dict["away"],
            "opponent": row_dict["home"],
            "is_home": False,
            "manager": row_dict["away_manager"],
            "match_id": row_dict["match_id"],
            "match_url": row_dict["match_url"],
        })

    timeline = pd.DataFrame(rows)
    timeline["date"] = pd.to_datetime(timeline["date"], errors="coerce")
    timeline = timeline.dropna(subset=["date", "team", "manager"])
    timeline = timeline.sort_values(["competition", "season", "team", "date", "match_id"]).reset_index(drop=True)

    return timeline


def infer_manager_changes(timeline_df):
    changes = []

    group_cols = ["competition", "season", "team"]

    for (competition, season, team), g in timeline_df.groupby(group_cols):
        g = g.sort_values(["date", "match_id"]).reset_index(drop=True)

        prev_manager = None
        prev_date = None
        prev_match_id = None
        prev_match_url = None

        for row in g.itertuples(index=False):
            manager = row.manager

            if prev_manager is None:
                prev_manager = manager
                prev_date = row.date
                prev_match_id = row.match_id
                prev_match_url = row.match_url
                continue

            if manager != prev_manager:
                changes.append({
                    "competition": competition,
                    "season": season,
                    "team": team,
                    "old_manager": prev_manager,
                    "new_manager": manager,
                    "last_old_manager_match_date": prev_date.strftime("%Y-%m-%d"),
                    "first_new_manager_match_date": row.date.strftime("%Y-%m-%d"),
                    "effective_change_date": row.date.strftime("%Y-%m-%d"),
                    "last_old_manager_match_id": prev_match_id,
                    "first_new_manager_match_id": row.match_id,
                    "last_old_manager_match_url": prev_match_url,
                    "first_new_manager_match_url": row.match_url,
                    "change_type": "inferred_from_match_record",
                    "source": "soccer-db",
                    "note": "Soccer D.B.の試合別監督名から推定。実際の解任発表日とは異なる可能性あり。",
                })

                prev_manager = manager

            prev_date = row.date
            prev_match_id = row.match_id
            prev_match_url = row.match_url

    changes_df = pd.DataFrame(changes)

    if not changes_df.empty:
        changes_df = changes_df.sort_values(
            ["competition", "season", "team", "effective_change_date"]
        ).reset_index(drop=True)

    return changes_df


# =========================
# main
# =========================

def main():
    match_managers_df = build_match_managers()
    match_managers_df.to_csv(OUTPUT_MATCH_MANAGERS_CSV, index=False, encoding="utf-8-sig")

    timeline_df = build_team_manager_timeline(match_managers_df)
    changes_df = infer_manager_changes(timeline_df)
    changes_df.to_csv(OUTPUT_MANAGER_CHANGES_CSV, index=False, encoding="utf-8-sig")

    print("\n==============================")
    print("出力完了")
    print("==============================")
    print("試合別監督CSV:", OUTPUT_MATCH_MANAGERS_CSV)
    print("監督交代推定CSV:", OUTPUT_MANAGER_CHANGES_CSV)

    if changes_df.empty:
        print("\n監督交代は検出されませんでした。")
    else:
        print("\n検出された監督交代:")
        show_cols = [
            "competition",
            "season",
            "team",
            "old_manager",
            "new_manager",
            "last_old_manager_match_date",
            "first_new_manager_match_date",
        ]
        print(changes_df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()