"""
manager_boost_pseudo_event_control.py

目的:
  「監督交代したチーム」と「同じように低迷したが監督交代しなかったチーム」を比較する。
  監督交代ブーストに見える上振れが、単なる平均回帰ではないかを確認するための疑似イベント比較コード。

このコードで行うこと:
  1. J1 2005〜2025の試合結果から、監督交代していない低迷タイミングを疑似イベントとして作る。
  2. 既存のcutoff反実仮想コードを利用して、疑似イベント後の1〜3/1〜5/1〜10試合を予測する。
  3. 本物の監督交代イベントと疑似イベントの残差を比較する。

必要ファイル:
  j1_j2_elo_input_1993_2025.csv
  manager_events_j1_2005_2025_boost_candidates.csv
  manager_boost_counterfactual_j1_2005_2025_v2.py
  manager_boost_counterfactual_j1_2005_2025_by_event_v15.csv  # 本物イベント比較用

出力:
  manager_boost_pseudo_events_j1_2005_2025.csv
  manager_boost_pseudo_counterfactual_matches_v15.csv
  manager_boost_pseudo_counterfactual_summary_v15.csv
  manager_boost_pseudo_counterfactual_by_event_v15.csv
  manager_boost_pseudo_counterfactual_by_year_v15.csv
  manager_boost_pseudo_counterfactual_unmatched_v15.csv
  manager_boost_pseudo_counterfactual_summary_v15.html
  manager_boost_true_vs_pseudo_summary.csv
  manager_boost_true_vs_pseudo_diff.csv
  manager_boost_true_vs_pseudo_by_timing.csv

使い方:
  python manager_boost_pseudo_event_control.py

注意:
  - RUN_COUNTERFACTUAL_FOR_PSEUDO = True の場合、既存の反実仮想コードを内部で呼び出すため少し時間がかかります。
  - 先に疑似イベントCSVだけ作りたい場合は、RUN_COUNTERFACTUAL_FOR_PSEUDO = False にしてください。
"""

from __future__ import annotations

import importlib.util
import contextlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

HISTORICAL_CSV = BASE_DIR / "j1_j2_elo_input_1993_2025.csv"
TRUE_EVENTS_CSV = BASE_DIR / "manager_events_j1_2005_2025_boost_candidates.csv"
TRUE_BY_EVENT_CSV = BASE_DIR / "manager_boost_counterfactual_j1_2005_2025_by_event_v15.csv"
COUNTERFACTUAL_SCRIPT = BASE_DIR / "manager_boost_counterfactual_j1_2005_2025_v2.py"

PSEUDO_EVENTS_CSV = BASE_DIR / "manager_boost_pseudo_events_j1_2005_2025.csv"
PSEUDO_MATCHES_CSV = BASE_DIR / "manager_boost_pseudo_counterfactual_matches_v15.csv"
PSEUDO_SUMMARY_CSV = BASE_DIR / "manager_boost_pseudo_counterfactual_summary_v15.csv"
PSEUDO_BY_EVENT_CSV = BASE_DIR / "manager_boost_pseudo_counterfactual_by_event_v15.csv"
PSEUDO_BY_YEAR_CSV = BASE_DIR / "manager_boost_pseudo_counterfactual_by_year_v15.csv"
PSEUDO_UNMATCHED_CSV = BASE_DIR / "manager_boost_pseudo_counterfactual_unmatched_v15.csv"
PSEUDO_HTML = BASE_DIR / "manager_boost_pseudo_counterfactual_summary_v15.html"

OUTPUT_TRUE_VS_PSEUDO_SUMMARY = BASE_DIR / "manager_boost_true_vs_pseudo_summary.csv"
OUTPUT_TRUE_VS_PSEUDO_DIFF = BASE_DIR / "manager_boost_true_vs_pseudo_diff.csv"
OUTPUT_TRUE_VS_PSEUDO_BY_TIMING = BASE_DIR / "manager_boost_true_vs_pseudo_by_timing.csv"

# =========================
# 設定
# =========================

TARGET_DIVISION = "J1"
TARGET_START_YEAR = 2005
TARGET_END_YEAR = 2025

# 疑似イベント条件
LAST_N = 5
LAST_N_POINTS_MAX = 3          # 直近5試合勝点3以下
LAST_N_GOAL_DIFF_MAX = None    # 例: -4 にすると「直近5試合得失点差-4以下」も条件にできる。Noneなら使わない。
MIN_MATCHES_PLAYED = 5
MIN_REMAINING_MATCHES = 5
MAX_GAMES_AFTER_CUTOFF_FOR_NO_REAL_CHANGE = 5

# 本物の監督交代の前後を疑似イベントから除外する。
# 平均回帰比較に本物の監督交代直前・直後が混ざるのを避けるため。
EXCLUDE_DAYS_BEFORE_TRUE_CHANGE = 14
EXCLUDE_DAYS_AFTER_TRUE_CHANGE = 35

# 同一チーム・同一シーズンで疑似イベントが密集しすぎないようにする。
MIN_GAP_GAMES_SAME_TEAM_SEASON = 5

# trueイベントと似た分布にするため、season×timing_bucketごとにtrue件数×倍率でサンプルする。
SAMPLE_BY_SEASON_AND_TIMING = True
PSEUDO_PER_TRUE = 1
RANDOM_SEED = 42

# Trueなら、疑似イベント作成後に既存の反実仮想コードを内部で実行する。
RUN_COUNTERFACTUAL_FOR_PSEUDO = True
REDIRECT_COUNTERFACTUAL_VERBOSE_TO_LOG = True
COUNTERFACTUAL_LOG = BASE_DIR / "manager_boost_pseudo_counterfactual_run.log"

WINDOWS = ["after_1_3", "after_1_5", "after_1_10", "after_4_6", "after_7_10"]


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"必要なファイルが見つかりません: {path.name}")


def standardize_team_name(name: object) -> str:
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
    }
    return name_map.get(name, name)


def clean_team_cols(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(standardize_team_name)
    return df


def load_historical() -> pd.DataFrame:
    require_file(HISTORICAL_CSV)
    hist = pd.read_csv(HISTORICAL_CSV, encoding="utf-8-sig")
    hist = clean_team_cols(hist, ["home", "away"])
    hist = hist[hist["division"].astype(str) == TARGET_DIVISION].copy()
    hist["year"] = pd.to_numeric(hist["year"], errors="coerce").astype("Int64")
    hist = hist[(hist["year"] >= TARGET_START_YEAR) & (hist["year"] <= TARGET_END_YEAR)].copy()
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    hist["home_goal"] = pd.to_numeric(hist["home_goal"], errors="coerce")
    hist["away_goal"] = pd.to_numeric(hist["away_goal"], errors="coerce")
    hist = hist.dropna(subset=["year", "date", "home", "away", "home_goal", "away_goal"]).copy()
    return hist.sort_values(["year", "date", "match_id" if "match_id" in hist.columns else "home"]).reset_index(drop=True)


def load_true_events() -> pd.DataFrame:
    require_file(TRUE_EVENTS_CSV)
    events = pd.read_csv(TRUE_EVENTS_CSV, encoding="utf-8-sig")
    events = clean_team_cols(events, ["team"])
    events["season"] = pd.to_numeric(events["season"], errors="coerce").astype("Int64")
    for col in ["effective_change_date", "last_old_manager_match_date", "first_new_manager_match_date", "next_change_date"]:
        if col in events.columns:
            events[col] = pd.to_datetime(events[col], errors="coerce")
    return events.dropna(subset=["season", "team", "effective_change_date"]).copy()


def team_match_log(hist: pd.DataFrame, season: int, team: str) -> pd.DataFrame:
    team = standardize_team_name(team)
    df = hist[
        (hist["year"] == season)
        & ((hist["home"] == team) | (hist["away"] == team))
    ].copy().sort_values("date").reset_index(drop=True)

    rows = []
    for i, r in df.iterrows():
        is_home = r["home"] == team
        gf = int(r["home_goal"] if is_home else r["away_goal"])
        ga = int(r["away_goal"] if is_home else r["home_goal"])
        if gf > ga:
            pts = 3
        elif gf == ga:
            pts = 1
        else:
            pts = 0
        rows.append({
            "season": season,
            "team": team,
            "match_index": i + 1,
            "date": r["date"],
            "home": r["home"],
            "away": r["away"],
            "opponent": r["away"] if is_home else r["home"],
            "is_home": is_home,
            "gf": gf,
            "ga": ga,
            "gd": gf - ga,
            "points": pts,
        })
    return pd.DataFrame(rows)


def timing_bucket(matches_played: int | float) -> str:
    if pd.isna(matches_played):
        return "unknown"
    x = int(matches_played)
    if x <= 8:
        return "00_08"
    if x <= 16:
        return "09_16"
    if x <= 24:
        return "17_24"
    return "25_plus"


def is_near_true_change(team: str, season: int, cutoff_date: pd.Timestamp, next_dates: list[pd.Timestamp], true_events: pd.DataFrame) -> bool:
    ev = true_events[(true_events["season"] == season) & (true_events["team"] == team)].copy()
    if ev.empty:
        return False

    for _, r in ev.iterrows():
        change_date = r["effective_change_date"]
        if pd.isna(change_date):
            continue

        # 本物の交代日前後のブラックアウト
        if (cutoff_date >= change_date - pd.Timedelta(days=EXCLUDE_DAYS_BEFORE_TRUE_CHANGE)) and (
            cutoff_date <= change_date + pd.Timedelta(days=EXCLUDE_DAYS_AFTER_TRUE_CHANGE)
        ):
            return True

        # 疑似イベント後の評価期間内に本物の監督交代が起きる場合は除外
        if next_dates:
            last_eval_date = max(next_dates[:MAX_GAMES_AFTER_CUTOFF_FOR_NO_REAL_CHANGE])
            if cutoff_date < change_date <= last_eval_date:
                return True
    return False


def low_form_condition(last: pd.DataFrame) -> bool:
    if len(last) < LAST_N:
        return False
    points = last["points"].sum()
    gd = last["gd"].sum()
    ok = points <= LAST_N_POINTS_MAX
    if LAST_N_GOAL_DIFF_MAX is not None:
        ok = ok and gd <= LAST_N_GOAL_DIFF_MAX
    return bool(ok)


def generate_pseudo_events(hist: pd.DataFrame, true_events: pd.DataFrame) -> pd.DataFrame:
    teams_by_season = {}
    for season, g in hist.groupby("year"):
        teams_by_season[int(season)] = sorted(set(g["home"]).union(g["away"]))

    candidates = []
    for season, teams in teams_by_season.items():
        for team in teams:
            log = team_match_log(hist, season, team)
            if log.empty:
                continue
            selected_match_indices = []

            # iはcutoff対象試合の0-based index。次戦が疑似イベント開始。
            for i in range(len(log) - 1):
                matches_played = i + 1
                remaining = len(log) - matches_played
                if matches_played < MIN_MATCHES_PLAYED or remaining < MIN_REMAINING_MATCHES:
                    continue

                # 同一チームシーズン内の疑似イベント間隔を空ける
                if selected_match_indices and (matches_played - selected_match_indices[-1]) < MIN_GAP_GAMES_SAME_TEAM_SEASON:
                    continue

                last = log.iloc[max(0, i - LAST_N + 1): i + 1]
                if not low_form_condition(last):
                    continue

                cutoff_date = log.iloc[i]["date"]
                next_matches = log.iloc[i + 1: i + 1 + max(MAX_GAMES_AFTER_CUTOFF_FOR_NO_REAL_CHANGE, 1)]
                next_dates = list(pd.to_datetime(next_matches["date"], errors="coerce"))
                if is_near_true_change(team, season, cutoff_date, next_dates, true_events):
                    continue

                first_after = log.iloc[i + 1]
                selected_match_indices.append(matches_played)
                candidates.append({
                    "event_id": f"PSEUDO_{season}_{team}_{matches_played:02d}",
                    "competition": "J1",
                    "season": season,
                    "team": team,
                    "old_manager": "no_change_control",
                    "new_manager": "no_change_control",
                    "bridge_manager": "",
                    "last_old_manager_match_date": cutoff_date,
                    "first_new_manager_match_date": first_after["date"],
                    "effective_change_date": first_after["date"],
                    "first_stable_new_manager_match_date": first_after["date"],
                    "matches_played_at_change": matches_played,
                    "remaining_matches_at_change_including_first_new": remaining,
                    "old_manager_matches_before_change": matches_played,
                    "new_manager_matches_until_next_change_or_season_end": remaining,
                    "team_season_matches": len(log),
                    "next_manager": "",
                    "next_change_date": pd.NaT,
                    "event_kind_auto": "pseudo_no_manager_change_low_form",
                    "temporary_returns_removed_in_team_season": 0,
                    "is_boost_candidate_auto": True,
                    "source": "pseudo_low_form_no_manager_change",
                    "pseudo_last5_points": last["points"].sum(),
                    "pseudo_last5_gf": last["gf"].sum(),
                    "pseudo_last5_ga": last["ga"].sum(),
                    "pseudo_last5_gd": last["gd"].sum(),
                    "timing_bucket": timing_bucket(matches_played),
                })
    pseudo = pd.DataFrame(candidates)
    if pseudo.empty:
        return pseudo

    pseudo = pseudo.sort_values(["season", "team", "effective_change_date"]).reset_index(drop=True)
    return pseudo


def sample_pseudo_like_true(pseudo: pd.DataFrame, true_events: pd.DataFrame) -> pd.DataFrame:
    if pseudo.empty or not SAMPLE_BY_SEASON_AND_TIMING:
        return pseudo

    true = true_events.copy()
    true["timing_bucket"] = true["matches_played_at_change"].apply(timing_bucket)
    true_counts = true.groupby(["season", "timing_bucket"]).size().reset_index(name="true_n")

    sampled_parts = []
    rng = np.random.default_rng(RANDOM_SEED)
    for _, row in true_counts.iterrows():
        season = int(row["season"])
        bucket = row["timing_bucket"]
        n = int(row["true_n"] * PSEUDO_PER_TRUE)
        sub = pseudo[(pseudo["season"] == season) & (pseudo["timing_bucket"] == bucket)].copy()
        if sub.empty:
            continue
        if len(sub) > n:
            sampled_idx = rng.choice(sub.index.to_numpy(), size=n, replace=False)
            sub = sub.loc[sampled_idx]
        sampled_parts.append(sub)

    if not sampled_parts:
        return pseudo
    sampled = pd.concat(sampled_parts, ignore_index=True)
    sampled = sampled.sort_values(["season", "team", "effective_change_date"]).reset_index(drop=True)

    # event_idを連番にする。元情報はmemo列に保持。
    sampled["original_pseudo_event_id"] = sampled["event_id"]
    sampled["event_id"] = [f"PSEUDO_{i:04d}" for i in range(1, len(sampled) + 1)]
    return sampled


def save_pseudo_events(pseudo: pd.DataFrame) -> None:
    out = pseudo.copy()
    for col in out.columns:
        if "date" in col:
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%Y-%m-%d")
    out.to_csv(PSEUDO_EVENTS_CSV, index=False, encoding="utf-8-sig")


def run_counterfactual_for_pseudo() -> None:
    require_file(COUNTERFACTUAL_SCRIPT)
    require_file(PSEUDO_EVENTS_CSV)

    spec = importlib.util.spec_from_file_location("counterfactual_module", COUNTERFACTUAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"反実仮想コードを読み込めません: {COUNTERFACTUAL_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # 既存コードの入力・出力先を疑似イベント用に差し替える。
    module.EVENTS_CSV_CANDIDATES = [PSEUDO_EVENTS_CSV.name]
    module.OUTPUT_MATCHES_CSV = PSEUDO_MATCHES_CSV
    module.OUTPUT_SUMMARY_CSV = PSEUDO_SUMMARY_CSV
    module.OUTPUT_BY_EVENT_CSV = PSEUDO_BY_EVENT_CSV
    module.OUTPUT_BY_YEAR_CSV = PSEUDO_BY_YEAR_CSV
    module.OUTPUT_UNMATCHED_CSV = PSEUDO_UNMATCHED_CSV
    module.OUTPUT_HTML = PSEUDO_HTML

    print("\n=== 疑似イベントのcutoff反実仮想を実行 ===")
    if REDIRECT_COUNTERFACTUAL_VERBOSE_TO_LOG:
        with open(COUNTERFACTUAL_LOG, "w", encoding="utf-8") as f, contextlib.redirect_stdout(f):
            module.main()
        print(f"counterfactual log: {COUNTERFACTUAL_LOG.name}")
    else:
        module.main()


def summarize_by_event(df: pd.DataFrame, group_name: str) -> pd.DataFrame:
    df = df.copy()
    df["group"] = group_name
    return df


def compare_true_vs_pseudo() -> None:
    require_file(TRUE_BY_EVENT_CSV)
    require_file(PSEUDO_BY_EVENT_CSV)

    true = pd.read_csv(TRUE_BY_EVENT_CSV, encoding="utf-8-sig")
    pseudo = pd.read_csv(PSEUDO_BY_EVENT_CSV, encoding="utf-8-sig")

    true = summarize_by_event(true, "true_manager_change")
    pseudo = summarize_by_event(pseudo, "pseudo_no_change")
    both = pd.concat([true, pseudo], ignore_index=True)

    # イベント単位の比較。sum_points_residualは「各イベントの該当ウィンドウ合計残差」。
    summary = both.groupby(["group", "window"], as_index=False).agg(
        n_events=("event_id", "nunique"),
        n_matches=("n_matches", "sum"),
        mean_event_sum_points_residual=("sum_points_residual", "mean"),
        median_event_sum_points_residual=("sum_points_residual", "median"),
        mean_match_points_residual=("mean_points_residual", "mean"),
        mean_goal_for_residual=("mean_goal_for_residual", "mean"),
        mean_goals_against_improvement=("mean_goals_against_improvement", "mean"),
        boost_rate=("sum_points_residual", lambda s: (pd.to_numeric(s, errors="coerce") > 0).mean()),
        strong_boost_rate=("sum_points_residual", lambda s: (pd.to_numeric(s, errors="coerce") >= 2.0).mean()),
        strong_negative_rate=("sum_points_residual", lambda s: (pd.to_numeric(s, errors="coerce") <= -2.0).mean()),
    )
    summary.to_csv(OUTPUT_TRUE_VS_PSEUDO_SUMMARY, index=False, encoding="utf-8-sig")

    # true - pseudo の差分
    metric_cols = [
        "mean_event_sum_points_residual",
        "median_event_sum_points_residual",
        "mean_match_points_residual",
        "mean_goal_for_residual",
        "mean_goals_against_improvement",
        "boost_rate",
        "strong_boost_rate",
        "strong_negative_rate",
    ]
    piv = summary.pivot(index="window", columns="group", values=metric_cols)
    diff_rows = []
    for window in piv.index:
        row = {"window": window}
        for metric in metric_cols:
            true_val = piv.loc[window].get((metric, "true_manager_change"), np.nan)
            pseudo_val = piv.loc[window].get((metric, "pseudo_no_change"), np.nan)
            row[f"true_{metric}"] = true_val
            row[f"pseudo_{metric}"] = pseudo_val
            row[f"diff_true_minus_pseudo_{metric}"] = true_val - pseudo_val
        diff_rows.append(row)
    diff = pd.DataFrame(diff_rows)
    diff.to_csv(OUTPUT_TRUE_VS_PSEUDO_DIFF, index=False, encoding="utf-8-sig")

    # timing_bucket別比較も可能なら作成
    # true側にはtiming_bucketがないため、events CSVから付与。
    true_events = load_true_events()
    true_events["timing_bucket"] = true_events["matches_played_at_change"].apply(timing_bucket)
    pseudo_events = pd.read_csv(PSEUDO_EVENTS_CSV, encoding="utf-8-sig")
    if "timing_bucket" not in pseudo_events.columns:
        pseudo_events["timing_bucket"] = pseudo_events["matches_played_at_change"].apply(timing_bucket)

    true_labeled = true.merge(true_events[["event_id", "timing_bucket"]], on="event_id", how="left")
    pseudo_labeled = pseudo.merge(pseudo_events[["event_id", "timing_bucket"]], on="event_id", how="left")
    both_labeled = pd.concat([true_labeled, pseudo_labeled], ignore_index=True)
    by_timing = both_labeled.groupby(["group", "timing_bucket", "window"], as_index=False).agg(
        n_events=("event_id", "nunique"),
        mean_event_sum_points_residual=("sum_points_residual", "mean"),
        boost_rate=("sum_points_residual", lambda s: (pd.to_numeric(s, errors="coerce") > 0).mean()),
        strong_boost_rate=("sum_points_residual", lambda s: (pd.to_numeric(s, errors="coerce") >= 2.0).mean()),
    )
    by_timing.to_csv(OUTPUT_TRUE_VS_PSEUDO_BY_TIMING, index=False, encoding="utf-8-sig")

    print("\n=== true vs pseudo 比較 ===")
    print(summary.round(3).to_string(index=False))
    print("\n=== true - pseudo 差分 ===")
    print(diff.round(3).to_string(index=False))


def main() -> None:
    for p in [HISTORICAL_CSV, TRUE_EVENTS_CSV, TRUE_BY_EVENT_CSV]:
        require_file(p)

    hist = load_historical()
    true_events = load_true_events()

    print(f"historical J1 matches: {len(hist)}")
    print(f"true manager events: {len(true_events)}")

    pseudo_all = generate_pseudo_events(hist, true_events)
    print(f"pseudo candidates before sampling: {len(pseudo_all)}")

    pseudo = sample_pseudo_like_true(pseudo_all, true_events)
    print(f"pseudo events after sampling: {len(pseudo)}")

    if pseudo.empty:
        raise ValueError("疑似イベントが0件でした。条件を緩めてください。")

    save_pseudo_events(pseudo)
    print(f"saved: {PSEUDO_EVENTS_CSV.name}")

    if RUN_COUNTERFACTUAL_FOR_PSEUDO:
        run_counterfactual_for_pseudo()
        compare_true_vs_pseudo()
    else:
        print("RUN_COUNTERFACTUAL_FOR_PSEUDO = False のため、反実仮想と比較集計は実行しません。")

    print("\n=== 出力予定ファイル ===")
    for p in [
        PSEUDO_EVENTS_CSV,
        PSEUDO_MATCHES_CSV,
        PSEUDO_SUMMARY_CSV,
        PSEUDO_BY_EVENT_CSV,
        PSEUDO_BY_YEAR_CSV,
        PSEUDO_UNMATCHED_CSV,
        PSEUDO_HTML,
        OUTPUT_TRUE_VS_PSEUDO_SUMMARY,
        OUTPUT_TRUE_VS_PSEUDO_DIFF,
        OUTPUT_TRUE_VS_PSEUDO_BY_TIMING,
    ]:
        print(p.name)


if __name__ == "__main__":
    main()
