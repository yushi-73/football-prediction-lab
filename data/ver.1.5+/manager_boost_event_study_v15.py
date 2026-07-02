import pandas as pd
import numpy as np
from pathlib import Path
from html import escape

# ============================================================
# 監督交代ブースト 順解析コード v15
# ------------------------------------------------------------
# 目的:
#   ・manager_events_j1_for_boost_v2.csv と
#     v15_base_multiyear_match_log.csv を結合する
#   ・監督交代後1〜3試合、4〜6試合、7〜10試合で
#     ver1.5の期待値に対して実績が上振れしたか確認する
#
# 入力:
#   manager_events_j1_for_boost_v2.csv
#   v15_base_multiyear_match_log.csv
#   soccerdb_match_managers_v2.csv  ※任意。ただしある方が望ましい
#
# 出力:
#   manager_boost_event_matches_v15.csv
#   manager_boost_event_unmatched_schedule_v15.csv
#   manager_boost_event_summary_v15.csv
#   manager_boost_event_by_change_v15.csv
#   manager_boost_event_by_year_v15.csv
#   manager_boost_event_summary_v15.html
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MANAGER_EVENTS_CSV = BASE_DIR / "manager_events_j1_for_boost_v2.csv"
V15_MATCH_LOG_CSV = BASE_DIR / "v15_base_multiyear_match_log.csv"
# 任意: これがあると「監督交代後何試合目か」を実際の日程ベースで数えられる
SOCCERDB_MATCH_MANAGERS_CSV = BASE_DIR / "soccerdb_match_managers_v2.csv"

OUTPUT_EVENT_MATCHES_CSV = BASE_DIR / "manager_boost_event_matches_v15.csv"
OUTPUT_UNMATCHED_SCHEDULE_CSV = BASE_DIR / "manager_boost_event_unmatched_schedule_v15.csv"
OUTPUT_SUMMARY_CSV = BASE_DIR / "manager_boost_event_summary_v15.csv"
OUTPUT_BY_CHANGE_CSV = BASE_DIR / "manager_boost_event_by_change_v15.csv"
OUTPUT_BY_YEAR_CSV = BASE_DIR / "manager_boost_event_by_year_v15.csv"
OUTPUT_HTML = BASE_DIR / "manager_boost_event_summary_v15.html"

# 監督交代後、何試合目まで集計するか
MAX_GAMES_AFTER_CHANGE = 10

# 集計ウィンドウ
WINDOWS = {
    "after_1_3": (1, 3),
    "after_4_6": (4, 6),
    "after_7_10": (7, 10),
    "after_1_5": (1, 5),
    "after_1_10": (1, 10),
}

BOOTSTRAP_N = 5000
RANDOM_SEED = 42


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
    }

    return name_map.get(name, name)


# =========================
# 読み込み
# =========================

def read_csv_required(path):
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} が見つかりません。\n"
            f"同じフォルダに配置してから実行してください: {path}"
        )
    return pd.read_csv(path, encoding="utf-8-sig")


def load_manager_events():
    df = read_csv_required(MANAGER_EVENTS_CSV)

    required = [
        "season", "team", "old_manager", "new_manager",
        "last_old_manager_match_date", "first_new_manager_match_date", "effective_change_date",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"manager events CSVに必要な列がありません: {missing}")

    # review_status がある場合は likely_include だけを使う
    if "review_status" in df.columns:
        before = len(df)
        df = df[df["review_status"].astype(str).str.contains("include", na=False)].copy()
        print(f"manager events: review_statusで絞り込み {before} -> {len(df)}")

    df["team"] = df["team"].apply(standardize_team_name)
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")

    for col in ["last_old_manager_match_date", "first_new_manager_match_date", "effective_change_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df = df.dropna(subset=["season", "team", "effective_change_date"]).copy()
    df["season"] = df["season"].astype(int)

    # イベントIDを付与
    df = df.sort_values(["season", "team", "effective_change_date"]).reset_index(drop=True)
    df.insert(0, "event_id", [f"E{i+1:03d}" for i in range(len(df))])

    return df


def load_v15_match_log():
    df = read_csv_required(V15_MATCH_LOG_CSV)

    required = [
        "target_year", "date", "home", "away",
        "actual_home_goal", "actual_away_goal",
        "lambda_home", "lambda_away",
        "home_win_prob", "draw_prob", "away_win_prob",
        "home_expected_points", "away_expected_points",
        "home_actual_points", "away_actual_points",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"v15 match log CSVに必要な列がありません: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["target_year"] = pd.to_numeric(df["target_year"], errors="coerce").astype("Int64")

    for col in ["home", "away"]:
        df[col] = df[col].apply(standardize_team_name)

    df = df.dropna(subset=["date", "target_year", "home", "away"]).copy()
    df["target_year"] = df["target_year"].astype(int)

    return df


# =========================
# チーム視点の試合ログを作る
# =========================

def build_team_match_log(match_log_df):
    rows = []

    for row in match_log_df.itertuples(index=False):
        d = row._asdict()

        # ホームチーム視点
        rows.append({
            "target_year": d["target_year"],
            "date": d["date"],
            "team": d["home"],
            "opponent": d["away"],
            "is_home": True,
            "home": d["home"],
            "away": d["away"],
            "actual_goals_for": d["actual_home_goal"],
            "actual_goals_against": d["actual_away_goal"],
            "lambda_for": d["lambda_home"],
            "lambda_against": d["lambda_away"],
            "team_win_prob": d["home_win_prob"],
            "team_draw_prob": d["draw_prob"],
            "team_loss_prob": d["away_win_prob"],
            "expected_points": d["home_expected_points"],
            "actual_points": d["home_actual_points"],
            "points_residual": d["home_actual_points"] - d["home_expected_points"],
            "goal_for_residual": d["actual_home_goal"] - d["lambda_home"],
            # プラスなら守備改善 = 期待失点より実失点が少ない
            "goals_against_improvement": d["lambda_away"] - d["actual_away_goal"],
            "actual_home_goal": d["actual_home_goal"],
            "actual_away_goal": d["actual_away_goal"],
            "lambda_home": d["lambda_home"],
            "lambda_away": d["lambda_away"],
        })

        # アウェイチーム視点
        rows.append({
            "target_year": d["target_year"],
            "date": d["date"],
            "team": d["away"],
            "opponent": d["home"],
            "is_home": False,
            "home": d["home"],
            "away": d["away"],
            "actual_goals_for": d["actual_away_goal"],
            "actual_goals_against": d["actual_home_goal"],
            "lambda_for": d["lambda_away"],
            "lambda_against": d["lambda_home"],
            "team_win_prob": d["away_win_prob"],
            "team_draw_prob": d["draw_prob"],
            "team_loss_prob": d["home_win_prob"],
            "expected_points": d["away_expected_points"],
            "actual_points": d["away_actual_points"],
            "points_residual": d["away_actual_points"] - d["away_expected_points"],
            "goal_for_residual": d["actual_away_goal"] - d["lambda_away"],
            # プラスなら守備改善 = 期待失点より実失点が少ない
            "goals_against_improvement": d["lambda_home"] - d["actual_home_goal"],
            "actual_home_goal": d["actual_home_goal"],
            "actual_away_goal": d["actual_away_goal"],
            "lambda_home": d["lambda_home"],
            "lambda_away": d["lambda_away"],
        })

    team_log = pd.DataFrame(rows)
    team_log["team"] = team_log["team"].apply(standardize_team_name)
    team_log["opponent"] = team_log["opponent"].apply(standardize_team_name)
    return team_log


# =========================
# 実際の日程から「交代後何試合目」を作る
# =========================

def load_fixture_timeline_if_available():
    if not SOCCERDB_MATCH_MANAGERS_CSV.exists():
        print("[INFO] soccerdb_match_managers_v2.csv がないため、v15 match log内の試合だけで交代後試合数を数えます。")
        return None

    df = pd.read_csv(SOCCERDB_MATCH_MANAGERS_CSV, encoding="utf-8-sig")
    required = ["competition", "season", "date", "home", "away"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[WARN] soccerdb_match_managers_v2.csv に必要列がありません: {missing}")
        return None

    # J1だけ使う
    if "competition" in df.columns:
        df = df[df["competition"].astype(str) == "J1"].copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    for col in ["home", "away"]:
        df[col] = df[col].apply(standardize_team_name)
    df = df.dropna(subset=["date", "season", "home", "away"]).copy()
    df["season"] = df["season"].astype(int)

    rows = []
    for row in df.itertuples(index=False):
        d = row._asdict()
        common = {
            "season": d["season"],
            "date": d["date"],
            "home": d["home"],
            "away": d["away"],
            "match_id": d.get("match_id", np.nan),
            "match_url": d.get("match_url", ""),
        }
        rows.append({**common, "team": d["home"], "opponent": d["away"], "is_home_schedule": True})
        rows.append({**common, "team": d["away"], "opponent": d["home"], "is_home_schedule": False})

    timeline = pd.DataFrame(rows)
    timeline["team"] = timeline["team"].apply(standardize_team_name)
    timeline["opponent"] = timeline["opponent"].apply(standardize_team_name)
    return timeline


def build_event_schedule(events_df, fixture_timeline, team_log):
    """
    fixture_timelineがある場合:
      実際の日程全体から交代後1〜10試合を作り、そのうちv15ログがある試合だけ残差を付ける。
    fixture_timelineがない場合:
      v15ログ内で交代後に出てくる試合だけを1〜10試合として扱う。
    """
    event_schedule_rows = []

    for event in events_df.itertuples(index=False):
        e = event._asdict()
        event_id = e["event_id"]
        season = int(e["season"])
        team = standardize_team_name(e["team"])
        change_date = e["effective_change_date"]

        if fixture_timeline is not None:
            g = fixture_timeline[
                (fixture_timeline["season"] == season)
                & (fixture_timeline["team"] == team)
                & (fixture_timeline["date"] >= change_date)
            ].copy()
        else:
            g = team_log[
                (team_log["target_year"] == season)
                & (team_log["team"] == team)
                & (team_log["date"] >= change_date)
            ].copy()
            g = g.rename(columns={"target_year": "season"})
            g["match_id"] = np.nan
            g["match_url"] = ""
            g["is_home_schedule"] = g["is_home"]

        g = g.sort_values(["date", "opponent"]).reset_index(drop=True)
        g["games_after_change"] = np.arange(1, len(g) + 1)
        g = g[g["games_after_change"] <= MAX_GAMES_AFTER_CHANGE].copy()

        for row in g.itertuples(index=False):
            r = row._asdict()
            event_schedule_rows.append({
                "event_id": event_id,
                "season": season,
                "team": team,
                "old_manager": e.get("old_manager", ""),
                "new_manager": e.get("new_manager", ""),
                "last_old_manager_match_date": e.get("last_old_manager_match_date", pd.NaT),
                "first_new_manager_match_date": e.get("first_new_manager_match_date", pd.NaT),
                "effective_change_date": change_date,
                "games_after_change": int(r["games_after_change"]),
                "date": r["date"],
                "opponent": standardize_team_name(r["opponent"]),
                "is_home_schedule": bool(r.get("is_home_schedule", False)),
                "match_id": r.get("match_id", np.nan),
                "match_url": r.get("match_url", ""),
            })

    event_schedule = pd.DataFrame(event_schedule_rows)
    return event_schedule


def attach_v15_residuals(event_schedule, team_log):
    if event_schedule.empty:
        return pd.DataFrame(), pd.DataFrame()

    left = event_schedule.copy()
    right = team_log.copy()

    # 結合キー
    left["merge_date"] = pd.to_datetime(left["date"], errors="coerce")
    right["merge_date"] = pd.to_datetime(right["date"], errors="coerce")

    merged = left.merge(
        right,
        left_on=["season", "merge_date", "team", "opponent"],
        right_on=["target_year", "merge_date", "team", "opponent"],
        how="left",
        suffixes=("", "_v15"),
    )

    matched = merged[~merged["expected_points"].isna()].copy()
    unmatched = merged[merged["expected_points"].isna()].copy()

    # 日付を文字列に戻す
    for df in [matched, unmatched]:
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            df["effective_change_date"] = pd.to_datetime(df["effective_change_date"], errors="coerce").dt.strftime("%Y-%m-%d")
            df["last_old_manager_match_date"] = pd.to_datetime(df["last_old_manager_match_date"], errors="coerce").dt.strftime("%Y-%m-%d")
            df["first_new_manager_match_date"] = pd.to_datetime(df["first_new_manager_match_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    return matched, unmatched


# =========================
# 集計
# =========================

def bootstrap_ci_mean(values, n_boot=BOOTSTRAP_N, seed=RANDOM_SEED):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    n = len(values)
    if n == 0:
        return np.nan, np.nan
    if n == 1:
        return values[0], values[0]

    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, n), replace=True)
    means = samples.mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize_df(df, group_cols):
    if df.empty:
        return pd.DataFrame()

    rows = []

    grouped = df.groupby(group_cols, dropna=False) if group_cols else [((), df)]

    for key, g in grouped:
        if not isinstance(key, tuple):
            key = (key,)

        row = {}
        for col, value in zip(group_cols, key):
            row[col] = value

        ci_low, ci_high = bootstrap_ci_mean(g["points_residual"])

        row.update({
            "n_matches": int(len(g)),
            "n_events": int(g["event_id"].nunique()) if "event_id" in g.columns else np.nan,
            "mean_expected_points": float(g["expected_points"].mean()),
            "mean_actual_points": float(g["actual_points"].mean()),
            "mean_points_residual": float(g["points_residual"].mean()),
            "sum_expected_points": float(g["expected_points"].sum()),
            "sum_actual_points": float(g["actual_points"].sum()),
            "sum_points_residual": float(g["points_residual"].sum()),
            "points_residual_ci_low": ci_low,
            "points_residual_ci_high": ci_high,
            "mean_lambda_for": float(g["lambda_for"].mean()),
            "mean_actual_goals_for": float(g["actual_goals_for"].mean()),
            "mean_goal_for_residual": float(g["goal_for_residual"].mean()),
            "mean_lambda_against": float(g["lambda_against"].mean()),
            "mean_actual_goals_against": float(g["actual_goals_against"].mean()),
            "mean_goals_against_improvement": float(g["goals_against_improvement"].mean()),
            "mean_team_win_prob": float(g["team_win_prob"].mean()),
            "mean_team_draw_prob": float(g["team_draw_prob"].mean()),
            "mean_team_loss_prob": float(g["team_loss_prob"].mean()),
            "home_match_rate": float(g["is_home"].mean()) if "is_home" in g.columns else np.nan,
        })

        rows.append(row)

    return pd.DataFrame(rows)


def add_window_labels(event_matches):
    rows = []
    for window_name, (start, end) in WINDOWS.items():
        g = event_matches[
            (event_matches["games_after_change"] >= start)
            & (event_matches["games_after_change"] <= end)
        ].copy()
        if g.empty:
            continue
        g["window"] = window_name
        g["window_start"] = start
        g["window_end"] = end
        rows.append(g)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


# =========================
# HTML出力
# =========================

def fmt_num(x, digits=3):
    if pd.isna(x):
        return "-"
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def fmt_pct(x, digits=1):
    if pd.isna(x):
        return "-"
    try:
        return f"{float(x) * 100:.{digits}f}%"
    except Exception:
        return str(x)


def format_for_html(df):
    out = df.copy()
    for col in out.columns:
        if col in {"home_match_rate", "mean_team_win_prob", "mean_team_draw_prob", "mean_team_loss_prob"}:
            out[col] = out[col].apply(lambda x: fmt_pct(x, 1))
        elif pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].apply(lambda x: fmt_num(x, 3))
    return out


def html_table(df):
    if df.empty:
        return "<p>該当データなし</p>"
    return format_for_html(df).to_html(index=False, classes="result-table", border=0, escape=False)


def export_html(summary_df, by_change_df, by_year_df, unmatched_df):
    n_unmatched = 0 if unmatched_df is None or unmatched_df.empty else len(unmatched_df)

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>監督交代ブースト順解析 | Football Prediction Lab</title>
  <link rel="stylesheet" href="style.css">
  <style>
    .result-table {{
      width: 100%; border-collapse: collapse; font-size: 13px; background: white;
    }}
    .result-table th, .result-table td {{
      border: 1px solid #ddd; padding: 8px 10px; text-align: center; white-space: nowrap;
    }}
    .result-table th {{ background: #222; color: white; }}
    .result-table tbody tr:nth-child(even) {{ background: #f5f5f5; }}
    .table-wrap {{ overflow-x: auto; }}
    .note-box {{
      background: #f8fafc; border-left: 5px solid #9400d3;
      padding: 16px 18px; border-radius: 10px; line-height: 1.8;
    }}
  </style>
</head>
<body>
  <header>
    <h1>監督交代ブースト順解析</h1>
    <p>ver1.5の期待値に対して、監督交代後の実績が上振れしたかを確認</p>
  </header>

  <main>
    <section>
      <h2>概要</h2>
      <div class="note-box">
        このページでは、監督交代が試合記録上で反映された後の試合について、
        ver1.5基準モデルの期待勝点・期待得点・期待失点と実績を比較する。
        使用している日付は厳密な解任発表日ではなく、Soccer D.B.の試合記録から推定した
        「新監督が確認できる最初の試合日」である。
        <br><br>
        v15ログと結合できなかった対象日程数: {n_unmatched}
      </div>
    </section>

    <section>
      <h2>ウィンドウ別集計</h2>
      <div class="table-wrap">
        {html_table(summary_df)}
      </div>
    </section>

    <section>
      <h2>年度別集計</h2>
      <div class="table-wrap">
        {html_table(by_year_df)}
      </div>
    </section>

    <section>
      <h2>監督交代イベント別集計</h2>
      <div class="table-wrap">
        {html_table(by_change_df)}
      </div>
    </section>
  </main>
</body>
</html>
"""
    OUTPUT_HTML.write_text(html, encoding="utf-8")


# =========================
# main
# =========================

def main():
    print("==============================")
    print("監督交代ブースト 順解析 v15")
    print("==============================")

    events_df = load_manager_events()
    match_log_df = load_v15_match_log()
    team_log = build_team_match_log(match_log_df)
    fixture_timeline = load_fixture_timeline_if_available()

    print("manager events:", len(events_df))
    print("v15 match log matches:", len(match_log_df))
    print("team perspective rows:", len(team_log))
    if fixture_timeline is not None:
        print("fixture timeline rows:", len(fixture_timeline))

    event_schedule = build_event_schedule(events_df, fixture_timeline, team_log)
    event_matches, unmatched = attach_v15_residuals(event_schedule, team_log)

    print("event schedule rows:", len(event_schedule))
    print("matched event rows:", len(event_matches))
    print("unmatched schedule rows:", len(unmatched))

    if event_matches.empty:
        raise ValueError(
            "v15 match logと結合できる監督交代後試合がありません。\n"
            "v15_base_multiyear_match_log.csv が後半戦のみを対象にしているため、\n"
            "監督交代直後の試合が前半戦にあると結合されない場合があります。"
        )

    # ウィンドウ別に展開
    windowed = add_window_labels(event_matches)

    summary_df = summarize_df(windowed, ["window", "window_start", "window_end"])
    summary_df = summary_df.sort_values(["window_start", "window_end"]).reset_index(drop=True)

    by_year_df = summarize_df(windowed, ["season", "window", "window_start", "window_end"])
    by_year_df = by_year_df.sort_values(["season", "window_start", "window_end"]).reset_index(drop=True)

    by_change_df = summarize_df(windowed, [
        "event_id", "season", "team", "old_manager", "new_manager",
        "effective_change_date", "window", "window_start", "window_end"
    ])
    by_change_df = by_change_df.sort_values(["season", "team", "window_start"]).reset_index(drop=True)

    # 出力用に日付を文字列化
    event_matches_out = event_matches.copy()
    for col in ["date", "effective_change_date", "last_old_manager_match_date", "first_new_manager_match_date"]:
        if col in event_matches_out.columns:
            event_matches_out[col] = pd.to_datetime(event_matches_out[col], errors="coerce").dt.strftime("%Y-%m-%d")

    event_matches_out.to_csv(OUTPUT_EVENT_MATCHES_CSV, index=False, encoding="utf-8-sig")
    unmatched.to_csv(OUTPUT_UNMATCHED_SCHEDULE_CSV, index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    by_change_df.to_csv(OUTPUT_BY_CHANGE_CSV, index=False, encoding="utf-8-sig")
    by_year_df.to_csv(OUTPUT_BY_YEAR_CSV, index=False, encoding="utf-8-sig")
    export_html(summary_df, by_change_df, by_year_df, unmatched)

    print("\n==============================")
    print("出力完了")
    print("==============================")
    print("event matches:", OUTPUT_EVENT_MATCHES_CSV)
    print("unmatched schedule:", OUTPUT_UNMATCHED_SCHEDULE_CSV)
    print("summary:", OUTPUT_SUMMARY_CSV)
    print("by change:", OUTPUT_BY_CHANGE_CSV)
    print("by year:", OUTPUT_BY_YEAR_CSV)
    print("html:", OUTPUT_HTML)

    print("\nウィンドウ別集計:")
    show_cols = [
        "window", "n_matches", "n_events",
        "mean_expected_points", "mean_actual_points", "mean_points_residual",
        "points_residual_ci_low", "points_residual_ci_high",
        "mean_goal_for_residual", "mean_goals_against_improvement",
    ]
    print(summary_df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
