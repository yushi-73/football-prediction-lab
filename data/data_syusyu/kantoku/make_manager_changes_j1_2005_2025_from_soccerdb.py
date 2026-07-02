import re
import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ============================================================
# Soccer D.B. J1 2005-2025 監督交代推定CSV 作成コード
# ------------------------------------------------------------
# 目的:
#   1. Soccer D.B. の大会結果ページから試合詳細ページURLを取得
#   2. 各試合詳細ページの Head Coaches からホーム/アウェイ監督名を取得
#   3. 同一シーズン・同一チーム内で監督名が変わった試合を検出
#   4. 監督交代が試合記録上で反映された日をCSV化
#
# 出力:
#   soccerdb_j1_2005_2025_match_managers.csv
#   soccerdb_j1_2005_2025_team_manager_timeline.csv
#   manager_changes_j1_2005_2025_inferred.csv
#   soccerdb_j1_2005_2025_manager_parse_errors.csv
#
# 注意:
#   このコードで作る日付は厳密な「解任発表日」ではなく、
#   「新監督が試合記録上で確認できる最初の試合日」です。
# ============================================================


# =========================
# 1. 設定
# =========================

BASE_URL = "https://soccer-db.net"
BASE_DIR = Path(__file__).resolve().parent

# J1=1001
COMPETITIONS = {
    1001: "J1",
}

YEARS = list(range(2005, 2026))

SLEEP_SECONDS = 0.35
REQUEST_TIMEOUT = 25
USE_HTML_CACHE = True
RESUME_FROM_EXISTING = True
CHECKPOINT_EVERY_N_MATCHES = 50

CACHE_DIR = BASE_DIR / "soccerdb_html_cache_j1_2005_2025"
CACHE_DIR.mkdir(exist_ok=True)

OUTPUT_MATCH_MANAGERS_CSV = BASE_DIR / "soccerdb_j1_2005_2025_match_managers.csv"
OUTPUT_TIMELINE_CSV = BASE_DIR / "soccerdb_j1_2005_2025_team_manager_timeline.csv"
OUTPUT_MANAGER_CHANGES_CSV = BASE_DIR / "manager_changes_j1_2005_2025_inferred.csv"
OUTPUT_PARSE_ERRORS_CSV = BASE_DIR / "soccerdb_j1_2005_2025_manager_parse_errors.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}


# =========================
# 2. チーム名標準化
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
        "ザスパクサツ群馬": "群馬",
        "ザスパ群馬": "群馬",
        "栃木ＳＣ": "栃木",
        "栃木SC": "栃木",
        "ファジアーノ岡山": "岡山",
        "ツエーゲン金沢": "金沢",
        "大分トリニータ": "大分",
        "ヴァンラーレ八戸": "八戸",
    }

    return name_map.get(name, name)


# =========================
# 3. HTML取得
# =========================

def get_cache_path(url):
    safe = re.sub(r"[^0-9A-Za-z_-]+", "_", url.replace(BASE_URL, ""))
    return CACHE_DIR / f"{safe}.html"


def fetch_html(url):
    cache_path = get_cache_path(url)

    if USE_HTML_CACHE and cache_path.exists():
        return cache_path.read_text(encoding="utf-8", errors="ignore")

    response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding
    html = response.text

    if USE_HTML_CACHE:
        cache_path.write_text(html, encoding="utf-8", errors="ignore")

    return html


def fetch_soup(url):
    html = fetch_html(url)
    return BeautifulSoup(html, "html.parser")


# =========================
# 4. 結果ページからResultリンク取得
# =========================

def get_result_links(competition_id, year):
    url = f"{BASE_URL}/competition/results/{competition_id}/{year}"
    soup = fetch_soup(url)

    links = []

    for a in soup.find_all("a", href=True):
        href = str(a.get("href", "")).strip()
        text = a.get_text(" ", strip=True)

        # textがResultでなくても /result/xxxxxxxxxxx を拾う
        if "/result/" not in href:
            continue

        m = re.search(r"/result/(\d+)", href)
        if not m:
            continue

        match_id = m.group(1)
        full_url = urljoin(BASE_URL, href)

        links.append({
            "competition_id": competition_id,
            "season": year,
            "match_id": match_id,
            "match_url": full_url,
            "link_text": text,
        })

    # 重複除去
    unique = {}
    for item in links:
        unique[item["match_id"]] = item

    result = list(unique.values())
    result = sorted(result, key=lambda x: x["match_id"])

    print(f"  Resultリンク検出数: {len(result)}")
    return result


# =========================
# 5. 試合情報・監督名の抽出
# =========================

def parse_match_info_from_title(title):
    """
    title例:
    Soccer D.B. : 2025 明治安田 Ｊ１リーグ 25/02/14 ガンバ大阪 - セレッソ大阪 試合結果,スタメン,フォーメーション
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


def normalize_cell_text(value):
    text = str(value)
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_noise_manager_text(text):
    text = normalize_cell_text(text)
    lower = text.lower()

    ignore_exact = {
        "",
        "nan",
        "none",
        "nat.",
        "nat",
        "age",
        "name",
        "nat.age name",
        "head coaches",
        "head coach",
        "coach",
        "監督",
        "stats",
        "results",
        "starting lineup",
        "lineups",
        "formation",
    }

    if lower in ignore_exact:
        return True

    # 数字・記号だけ
    if re.fullmatch(r"[\d\.\-–—:：/()（）\s]+", text):
        return True

    # 国籍3文字だけなど
    if re.fullmatch(r"[A-Z]{2,3}", text):
        return True

    return False


def looks_like_person_name(text):
    text = normalize_cell_text(text)

    if is_noise_manager_text(text):
        return False

    # チーム名・大会名っぽいものを除外
    bad_keywords = [
        "リーグ", "カップ", "天皇杯", "明治安田", "Ｊ１", "Ｊ２", "J1", "J2",
        "試合", "結果", "スタメン", "フォーメーション", "順位", "観客",
        "得点", "警告", "退場", "選手", "Stats", "Result",
    ]
    if any(k in text for k in bad_keywords):
        return False

    # 長すぎる文は除外
    if len(text) > 40:
        return False

    # 日本語名、カタカナ名、欧文名を許容
    if re.search(r"[一-龥ぁ-んァ-ヴーA-Za-z]", text):
        return True

    return False


def extract_coaches_from_table_near_heading(soup):
    """
    Head Coaches 見出しの直後にある table を優先して監督名を取る。
    pandasではなくHTML構造から直接取るので、列ズレに比較的強い。
    """
    heading_node = soup.find(string=lambda s: s and "Head Coaches" in str(s))
    if heading_node is None:
        return None, None, "Head Coaches heading not found"

    table = heading_node.find_parent().find_next("table") if heading_node.find_parent() else None
    if table is None:
        # 見出しの親から取れない場合、見出し以降の最初のtableを探す
        table = soup.find("table")

    if table is None:
        return None, None, "table near Head Coaches not found"

    # まず name 列があるテーブルとして読む
    try:
        dfs = pd.read_html(str(table))
    except Exception as e:
        dfs = []

    candidates = []

    for df in dfs:
        if df.empty:
            continue

        # MultiIndex列を文字列化
        df = df.copy()
        df.columns = [" ".join(map(str, c)).strip() if isinstance(c, tuple) else str(c).strip() for c in df.columns]

        # name列を優先
        name_cols = [c for c in df.columns if "name" in c.lower() or "名前" in c]
        if name_cols:
            for col in name_cols:
                for v in df[col].tolist():
                    t = normalize_cell_text(v)
                    if looks_like_person_name(t):
                        candidates.append(t)
        else:
            # name列が見つからなければ全セルを走査
            for v in df.astype(str).values.ravel():
                t = normalize_cell_text(v)
                if looks_like_person_name(t):
                    candidates.append(t)

    # HTMLテーブルから取れない場合、tr単位で最後のtdをname候補として拾う
    if len(candidates) < 2:
        for tr in table.find_all("tr"):
            cells = [normalize_cell_text(td.get_text(" ", strip=True)) for td in tr.find_all(["td", "th"])]
            cells = [c for c in cells if not is_noise_manager_text(c)]
            if cells:
                t = cells[-1]
                if looks_like_person_name(t):
                    candidates.append(t)

    # 重複除去しつつ順序保持
    unique = []
    for c in candidates:
        if c not in unique:
            unique.append(c)

    if len(unique) >= 2:
        return unique[0], unique[1], "table"

    return None, None, f"table parse failed candidates={unique[:5]}"


def extract_coaches_from_text_lines(soup):
    """
    table方式が失敗した場合のフォールバック。
    Head Coaches 以降のテキスト行から、nat.などを除外して2名取る。
    """
    text = soup.get_text("\n", strip=True)
    lines = [normalize_cell_text(line) for line in text.splitlines() if normalize_cell_text(line)]

    try:
        start_idx = next(i for i, line in enumerate(lines) if "Head Coaches" in line)
    except StopIteration:
        return None, None, "Head Coaches not found in text"

    stop_keywords = {
        "Stats", "Results", "Starting Lineup", "Substitutes", "Head Coaches Stats",
        "Players", "Formation", "Referee", "直近の結果",
    }

    candidates = []
    for line in lines[start_idx + 1: start_idx + 30]:
        if line in stop_keywords:
            break
        if looks_like_person_name(line):
            candidates.append(line)

    # 重複除去
    unique = []
    for c in candidates:
        if c not in unique:
            unique.append(c)

    if len(unique) >= 2:
        return unique[0], unique[1], "text"

    return None, None, f"text parse failed candidates={unique[:5]}"


def parse_score_from_text(soup):
    """
    スコアは今回の監督交代検出には必須ではない。
    取れそうな場合だけ補助的に取得する。
    失敗時はNaNにする。
    """
    # ページ構造に依存しすぎるため、ここでは無理に取らない。
    return pd.NA, pd.NA


def parse_match_detail(match_url):
    soup = fetch_soup(match_url)

    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    date, home, away = parse_match_info_from_title(title)

    home_manager, away_manager, method = extract_coaches_from_table_near_heading(soup)
    if home_manager is None or away_manager is None:
        home_manager, away_manager, method = extract_coaches_from_text_lines(soup)

    home_goal, away_goal = parse_score_from_text(soup)

    return {
        "date": date,
        "home": home,
        "away": away,
        "home_goal": home_goal,
        "away_goal": away_goal,
        "home_manager": home_manager,
        "away_manager": away_manager,
        "manager_parse_method": method,
        "page_title": title,
    }



# =========================
# 6. 途中保存・再開用ヘルパー
# =========================

def save_checkpoint(rows, errors):
    """長時間実行に備えて途中結果を保存する。"""
    if rows:
        tmp_df = pd.DataFrame(rows)
        tmp_df.to_csv(OUTPUT_MATCH_MANAGERS_CSV, index=False, encoding="utf-8-sig")
    if errors:
        tmp_errors = pd.DataFrame(errors)
        tmp_errors.to_csv(OUTPUT_PARSE_ERRORS_CSV, index=False, encoding="utf-8-sig")


def load_existing_rows_for_resume():
    """既存CSVがあれば読み込み、取得済みmatch_idをスキップできるようにする。"""
    if not RESUME_FROM_EXISTING or not OUTPUT_MATCH_MANAGERS_CSV.exists():
        return [], set()

    try:
        existing_df = pd.read_csv(OUTPUT_MATCH_MANAGERS_CSV)
    except Exception:
        return [], set()

    if existing_df.empty or "match_id" not in existing_df.columns:
        return [], set()

    existing_df["match_id"] = existing_df["match_id"].astype(str)
    existing_rows = existing_df.to_dict("records")
    done_match_ids = set(existing_df["match_id"].dropna().astype(str).tolist())

    print(f"既存CSVから再開: {len(existing_rows)}行 / 取得済みmatch_id {len(done_match_ids)}件")
    return existing_rows, done_match_ids

# =========================
# 6. 試合別監督CSV作成
# =========================

def build_match_managers():
    rows, done_match_ids = load_existing_rows_for_resume()
    errors = []

    for competition_id, competition_name in COMPETITIONS.items():
        for year in YEARS:
            print(f"\n{competition_name} {year}: 試合リンク取得中...")
            links = get_result_links(competition_id, year)

            if not links:
                errors.append({
                    "competition": competition_name,
                    "season": year,
                    "match_id": "",
                    "match_url": f"{BASE_URL}/competition/results/{competition_id}/{year}",
                    "error": "Result link count is zero",
                })
                continue

            print(f"  {len(links)}試合を取得")

            for i, item in enumerate(links, start=1):
                url = item["match_url"]
                match_id = str(item["match_id"])

                if match_id in done_match_ids:
                    if i % 100 == 0:
                        print(f"  {i}/{len(links)}試合確認済み")
                    continue

                try:
                    info = parse_match_detail(url)
                except Exception as e:
                    errors.append({
                        "competition": competition_name,
                        "season": year,
                        "match_id": item.get("match_id"),
                        "match_url": url,
                        "error": repr(e),
                    })
                    print(f"  [WARN] {url} の取得に失敗: {e}")
                    continue

                row = {
                    "competition": competition_name,
                    "competition_id": competition_id,
                    "season": year,
                    "match_id": item["match_id"],
                    "match_url": url,
                    "date": info.get("date"),
                    "home": info.get("home"),
                    "away": info.get("away"),
                    "home_goal": info.get("home_goal"),
                    "away_goal": info.get("away_goal"),
                    "home_manager": info.get("home_manager"),
                    "away_manager": info.get("away_manager"),
                    "manager_parse_method": info.get("manager_parse_method"),
                    "page_title": info.get("page_title"),
                }

                # 監督名が取れていない場合も行は残し、エラーにも記録
                if not row["home_manager"] or not row["away_manager"]:
                    errors.append({
                        "competition": competition_name,
                        "season": year,
                        "match_id": item.get("match_id"),
                        "match_url": url,
                        "error": f"manager parse failed: {row['manager_parse_method']}",
                    })

                rows.append(row)
                done_match_ids.add(match_id)

                if i % CHECKPOINT_EVERY_N_MATCHES == 0:
                    save_checkpoint(rows, errors)

                if i % 50 == 0:
                    print(f"  {i}/{len(links)}試合完了")

                time.sleep(SLEEP_SECONDS)

    if not rows:
        raise ValueError("試合データを1件も取得できませんでした。")

    df = pd.DataFrame(rows)

    for col in ["home", "away"]:
        if col in df.columns:
            df[col] = df[col].apply(standardize_team_name)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # ソート前に欠損確認
    missing_manager = df["home_manager"].isna().sum() + df["away_manager"].isna().sum()
    missing_date = df["date"].isna().sum()
    print(f"\n取得行数: {len(df)}")
    print(f"date欠損: {missing_date}")
    print(f"manager欠損セル数: {missing_manager}")

    df = df.sort_values(["competition", "season", "date", "match_id"]).reset_index(drop=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    errors_df = pd.DataFrame(errors)
    return df, errors_df


# =========================
# 7. 監督交代推定
# =========================

def build_team_manager_timeline(match_managers_df):
    rows = []

    for row in match_managers_df.itertuples(index=False):
        rd = row._asdict()

        if pd.notna(rd.get("home_manager")):
            rows.append({
                "competition": rd["competition"],
                "season": rd["season"],
                "date": rd["date"],
                "team": rd["home"],
                "opponent": rd["away"],
                "is_home": True,
                "manager": rd["home_manager"],
                "match_id": rd["match_id"],
                "match_url": rd["match_url"],
            })

        if pd.notna(rd.get("away_manager")):
            rows.append({
                "competition": rd["competition"],
                "season": rd["season"],
                "date": rd["date"],
                "team": rd["away"],
                "opponent": rd["home"],
                "is_home": False,
                "manager": rd["away_manager"],
                "match_id": rd["match_id"],
                "match_url": rd["match_url"],
            })

    timeline = pd.DataFrame(rows)
    if timeline.empty:
        return timeline

    timeline["date"] = pd.to_datetime(timeline["date"], errors="coerce")
    timeline = timeline.dropna(subset=["date", "team", "manager"]).copy()
    timeline["manager"] = timeline["manager"].astype(str).str.strip()
    timeline = timeline[timeline["manager"].ne("")].copy()

    timeline = timeline.sort_values(["competition", "season", "team", "date", "match_id"]).reset_index(drop=True)
    return timeline


def infer_manager_changes(timeline_df):
    changes = []

    if timeline_df.empty:
        return pd.DataFrame(changes)

    group_cols = ["competition", "season", "team"]

    for (competition, season, team), g in timeline_df.groupby(group_cols):
        g = g.sort_values(["date", "match_id"]).reset_index(drop=True)

        prev_manager = None
        prev_date = None
        prev_match_id = None
        prev_match_url = None

        for row in g.itertuples(index=False):
            manager = str(row.manager).strip()

            if not manager:
                continue

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
        changes_df = changes_df.sort_values(["competition", "season", "team", "effective_change_date"]).reset_index(drop=True)

    return changes_df


# =========================
# 8. 簡易チェック
# =========================

def quality_check(match_managers_df, changes_df):
    print("\n==============================")
    print("品質チェック")
    print("==============================")

    print("試合別監督CSV rows:", len(match_managers_df))
    print("home_manager unique sample:")
    print(match_managers_df["home_manager"].dropna().astype(str).head(10).to_string(index=False))
    print("away_manager unique sample:")
    print(match_managers_df["away_manager"].dropna().astype(str).head(10).to_string(index=False))

    bad_values = {"nat.", "nat", "age", "name", "nat.age name"}
    bad_home = match_managers_df["home_manager"].astype(str).str.lower().isin(bad_values).sum()
    bad_away = match_managers_df["away_manager"].astype(str).str.lower().isin(bad_values).sum()
    print("bad manager values:", bad_home + bad_away)

    print("\n推定監督交代 rows:", len(changes_df))
    if not changes_df.empty:
        show_cols = [
            "competition", "season", "team", "old_manager", "new_manager",
            "last_old_manager_match_date", "first_new_manager_match_date",
        ]
        print(changes_df[show_cols].to_string(index=False))


# =========================
# 9. main
# =========================

def main():
    match_managers_df, errors_df = build_match_managers()
    match_managers_df.to_csv(OUTPUT_MATCH_MANAGERS_CSV, index=False, encoding="utf-8-sig")
    errors_df.to_csv(OUTPUT_PARSE_ERRORS_CSV, index=False, encoding="utf-8-sig")

    timeline_df = build_team_manager_timeline(match_managers_df)
    timeline_df.to_csv(OUTPUT_TIMELINE_CSV, index=False, encoding="utf-8-sig")

    changes_df = infer_manager_changes(timeline_df)
    changes_df.to_csv(OUTPUT_MANAGER_CHANGES_CSV, index=False, encoding="utf-8-sig")

    quality_check(match_managers_df, changes_df)

    print("\n==============================")
    print("出力完了")
    print("==============================")
    print("試合別監督CSV:", OUTPUT_MATCH_MANAGERS_CSV)
    print("チーム別監督タイムラインCSV:", OUTPUT_TIMELINE_CSV)
    print("監督交代推定CSV:", OUTPUT_MANAGER_CHANGES_CSV)
    print("解析エラーCSV:", OUTPUT_PARSE_ERRORS_CSV)


if __name__ == "__main__":
    main()
