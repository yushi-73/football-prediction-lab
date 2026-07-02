"""
make_manager_boost_condition_dataset.py

目的:
  監督交代ブースト検証の出力を、条件探索しやすい「1イベント=1行」の横長CSVに変換する。
  あわせて、交代時期・Elo・期待勝点・相手強度・直近不調度などの条件別サマリーを出力する。

入力:
  manager_boost_counterfactual_j1_2005_2025_by_event_v15.csv
  manager_boost_counterfactual_j1_2005_2025_matches_v15.csv
  manager_events_j1_2005_2025_boost_candidates.csv
  j1_j2_elo_input_1993_2025.csv

出力:
  manager_boost_condition_dataset.csv
  manager_boost_condition_summary_by_timing.csv
  manager_boost_condition_summary_by_expected_points.csv
  manager_boost_condition_summary_by_elo.csv
  manager_boost_condition_summary_by_opponent_elo.csv
  manager_boost_condition_summary_by_last5_points.csv
  manager_boost_condition_summary_by_pre_context.csv
  manager_boost_top_positive_events.csv
  manager_boost_top_negative_events.csv
  manager_boost_condition_summary.html

使い方:
  python make_manager_boost_condition_dataset.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

BY_EVENT_CSV = BASE_DIR / "manager_boost_counterfactual_j1_2005_2025_by_event_v15.csv"
MATCHES_CSV = BASE_DIR / "manager_boost_counterfactual_j1_2005_2025_matches_v15.csv"
EVENTS_CSV = BASE_DIR / "manager_events_j1_2005_2025_boost_candidates.csv"
HISTORICAL_CSV = BASE_DIR / "j1_j2_elo_input_1993_2025.csv"

OUTPUT_DATASET = BASE_DIR / "manager_boost_condition_dataset.csv"
OUTPUT_SUMMARY_TIMING = BASE_DIR / "manager_boost_condition_summary_by_timing.csv"
OUTPUT_SUMMARY_EXPECTED = BASE_DIR / "manager_boost_condition_summary_by_expected_points.csv"
OUTPUT_SUMMARY_ELO = BASE_DIR / "manager_boost_condition_summary_by_elo.csv"
OUTPUT_SUMMARY_OPP_ELO = BASE_DIR / "manager_boost_condition_summary_by_opponent_elo.csv"
OUTPUT_SUMMARY_LAST5 = BASE_DIR / "manager_boost_condition_summary_by_last5_points.csv"
OUTPUT_SUMMARY_CONTEXT = BASE_DIR / "manager_boost_condition_summary_by_pre_context.csv"
OUTPUT_TOP_POSITIVE = BASE_DIR / "manager_boost_top_positive_events.csv"
OUTPUT_TOP_NEGATIVE = BASE_DIR / "manager_boost_top_negative_events.csv"
OUTPUT_HTML = BASE_DIR / "manager_boost_condition_summary.html"

WINDOWS = ["after_1_3", "after_1_5", "after_1_10", "after_4_6", "after_7_10"]

# 「1イベント=1行」に残すby_event側の集計列
PIVOT_METRICS = [
    "n_matches",
    "mean_expected_points",
    "mean_actual_points",
    "mean_points_residual",
    "sum_expected_points",
    "sum_actual_points",
    "sum_points_residual",
    "points_residual_ci_low",
    "points_residual_ci_high",
    "mean_lambda_for",
    "mean_actual_goals_for",
    "mean_goal_for_residual",
    "mean_lambda_against",
    "mean_actual_goals_against",
    "mean_goals_against_improvement",
    "goals_against_improvement_ci_low",
    "goals_against_improvement_ci_high",
    "mean_goal_diff_residual",
    "mean_team_win_prob",
    "mean_team_draw_prob",
    "mean_team_loss_prob",
    "home_match_rate",
    "mean_team_games_in_train_at_cutoff",
]

EVENT_ID_COLS = [
    "event_id",
    "season",
    "team",
    "old_manager",
    "new_manager",
    "effective_change_date",
]


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"必要なファイルが見つかりません: {path.name}")


def standardize_team_name(name: object) -> str:
    """Jリーグの正式名・略称を分析用の略称に寄せる。"""
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


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for p in [BY_EVENT_CSV, MATCHES_CSV, EVENTS_CSV, HISTORICAL_CSV]:
        require_file(p)

    by_event = pd.read_csv(BY_EVENT_CSV, encoding="utf-8-sig")
    matches = pd.read_csv(MATCHES_CSV, encoding="utf-8-sig")
    events = pd.read_csv(EVENTS_CSV, encoding="utf-8-sig")
    hist = pd.read_csv(HISTORICAL_CSV, encoding="utf-8-sig")

    for df in [by_event, matches, events]:
        if "season" in df.columns:
            df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
        for col in ["effective_change_date", "last_old_manager_match_date", "first_new_manager_match_date", "next_change_date", "date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

    hist = clean_team_cols(hist, ["home", "away"])
    if "division" in hist.columns:
        hist = hist[hist["division"].astype(str) == "J1"].copy()
    hist["year"] = pd.to_numeric(hist["year"], errors="coerce").astype("Int64")
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    hist["home_goal"] = pd.to_numeric(hist["home_goal"], errors="coerce")
    hist["away_goal"] = pd.to_numeric(hist["away_goal"], errors="coerce")
    hist = hist.dropna(subset=["year", "date", "home", "away", "home_goal", "away_goal"]).copy()

    for df in [by_event, matches, events]:
        df = clean_team_cols(df, ["team", "home", "away", "opponent"])

    by_event = clean_team_cols(by_event, ["team"])
    matches = clean_team_cols(matches, ["team", "home", "away", "opponent"])
    events = clean_team_cols(events, ["team"])

    return by_event, matches, events, hist


def make_event_wide(by_event: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in EVENT_ID_COLS + ["window"] if c not in by_event.columns]
    if missing:
        raise ValueError(f"by_event CSVに必要な列がありません: {missing}")

    base = (
        by_event[EVENT_ID_COLS]
        .drop_duplicates(subset=["event_id"])
        .sort_values(["season", "team", "effective_change_date"])
        .reset_index(drop=True)
    )

    use_metrics = [c for c in PIVOT_METRICS if c in by_event.columns]
    pivot = by_event.pivot_table(
        index="event_id",
        columns="window",
        values=use_metrics,
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{window}" for metric, window in pivot.columns]
    pivot = pivot.reset_index()

    wide = base.merge(pivot, on="event_id", how="left")
    return wide


def add_match_level_features(wide: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    df = wide.copy()
    m = matches.copy()
    m["games_after_change"] = pd.to_numeric(m["games_after_change"], errors="coerce")

    # cutoff時点Eloや学習試合数はイベント内で同値のはずなのでfirstを採用
    first_cols = [
        "team_elo_at_cutoff",
        "team_games_in_train_at_cutoff",
        "n_train_matches_at_cutoff",
        "team_prev_games",
        "team_effective_prev_weight",
    ]
    first_cols = [c for c in first_cols if c in m.columns]
    first_features = m.groupby("event_id", as_index=False)[first_cols].first()
    df = df.merge(first_features, on="event_id", how="left")

    windows = {
        "after_1_3": (1, 3),
        "after_1_5": (1, 5),
        "after_1_10": (1, 10),
    }
    for suffix, (lo, hi) in windows.items():
        sub = m[m["games_after_change"].between(lo, hi)].copy()
        if sub.empty:
            continue
        agg_map = {}
        if "opponent_elo_at_cutoff" in sub.columns:
            agg_map["opponent_elo_at_cutoff"] = ["mean", "min", "max"]
        if "expected_points" in sub.columns:
            agg_map["expected_points"] = "sum"
        if "actual_points" in sub.columns:
            agg_map["actual_points"] = "sum"
        if "points_residual" in sub.columns:
            agg_map["points_residual"] = "sum"
        if "is_home" in sub.columns:
            agg_map["is_home"] = "mean"
        if not agg_map:
            continue
        g = sub.groupby("event_id").agg(agg_map)
        g.columns = [
            f"{a}_{b}_{suffix}" if b else f"{a}_{suffix}"
            for a, b in g.columns.to_flat_index()
        ]
        g = g.reset_index()
        df = df.merge(g, on="event_id", how="left")

    # 読みやすい別名
    rename_map = {
        "opponent_elo_at_cutoff_mean_after_1_3": "opponent_elo_mean_after_1_3",
        "opponent_elo_at_cutoff_min_after_1_3": "opponent_elo_min_after_1_3",
        "opponent_elo_at_cutoff_max_after_1_3": "opponent_elo_max_after_1_3",
        "is_home_mean_after_1_3": "home_rate_after_1_3_from_matches",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df


def points_for_team(row: pd.Series, team: str) -> tuple[int, int, int, int]:
    """戻り値: points, gf, ga, gd"""
    home_goal = int(row["home_goal"])
    away_goal = int(row["away_goal"])
    if row["home"] == team:
        gf, ga = home_goal, away_goal
    elif row["away"] == team:
        gf, ga = away_goal, home_goal
    else:
        raise ValueError("row does not contain team")

    if gf > ga:
        pts = 3
    elif gf == ga:
        pts = 1
    else:
        pts = 0
    return pts, gf, ga, gf - ga


def get_pre_form_features(hist: pd.DataFrame, event: pd.Series) -> dict:
    season = int(event["season"])
    team = standardize_team_name(event["team"])
    cutoff = pd.to_datetime(event.get("effective_change_date"), errors="coerce")
    last_old = pd.to_datetime(event.get("last_old_manager_match_date"), errors="coerce")
    # 前任監督の最終戦までを「交代前」とみなす。
    cutoff_limit = last_old if pd.notna(last_old) else cutoff

    team_matches = hist[
        (hist["year"] == season)
        & ((hist["home"] == team) | (hist["away"] == team))
        & (hist["date"] <= cutoff_limit)
    ].sort_values("date")

    out = {
        "pre_matches": len(team_matches),
        "pre_points": np.nan,
        "pre_ppg": np.nan,
        "pre_gf": np.nan,
        "pre_ga": np.nan,
        "pre_gd": np.nan,
        "pre_gf_per_match": np.nan,
        "pre_ga_per_match": np.nan,
        "pre_gd_per_match": np.nan,
        "last3_points": np.nan,
        "last3_ppg": np.nan,
        "last3_gf": np.nan,
        "last3_ga": np.nan,
        "last3_gd": np.nan,
        "last5_points": np.nan,
        "last5_ppg": np.nan,
        "last5_gf": np.nan,
        "last5_ga": np.nan,
        "last5_gd": np.nan,
        "last5_gf_per_match": np.nan,
        "last5_ga_per_match": np.nan,
        "last5_gd_per_match": np.nan,
    }
    if team_matches.empty:
        return out

    records = [points_for_team(r, team) for _, r in team_matches.iterrows()]
    arr = pd.DataFrame(records, columns=["points", "gf", "ga", "gd"])

    out.update({
        "pre_points": arr["points"].sum(),
        "pre_ppg": arr["points"].mean(),
        "pre_gf": arr["gf"].sum(),
        "pre_ga": arr["ga"].sum(),
        "pre_gd": arr["gd"].sum(),
        "pre_gf_per_match": arr["gf"].mean(),
        "pre_ga_per_match": arr["ga"].mean(),
        "pre_gd_per_match": arr["gd"].mean(),
    })

    for n in [3, 5]:
        last = arr.tail(n)
        if last.empty:
            continue
        out.update({
            f"last{n}_points": last["points"].sum(),
            f"last{n}_ppg": last["points"].mean(),
            f"last{n}_gf": last["gf"].sum(),
            f"last{n}_ga": last["ga"].sum(),
            f"last{n}_gd": last["gd"].sum(),
            f"last{n}_gf_per_match": last["gf"].mean(),
            f"last{n}_ga_per_match": last["ga"].mean(),
            f"last{n}_gd_per_match": last["gd"].mean(),
        })
    return out


def add_pre_form_features(wide: pd.DataFrame, events: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    event_cols = [
        "event_id",
        "competition",
        "last_old_manager_match_date",
        "first_new_manager_match_date",
        "matches_played_at_change",
        "remaining_matches_at_change_including_first_new",
        "old_manager_matches_before_change",
        "new_manager_matches_until_next_change_or_season_end",
        "event_kind_auto",
        "bridge_manager",
        "next_manager",
        "next_change_date",
    ]
    event_cols = [c for c in event_cols if c in events.columns]
    e = events[event_cols].drop_duplicates(subset=["event_id"])
    df = wide.merge(e, on="event_id", how="left")

    form_rows = []
    for _, row in df.iterrows():
        features = get_pre_form_features(hist, row)
        features["event_id"] = row["event_id"]
        form_rows.append(features)
    form_df = pd.DataFrame(form_rows)
    df = df.merge(form_df, on="event_id", how="left")
    return df


def qcut_safe(series: pd.Series, q: int, labels: list[str]) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    try:
        return pd.qcut(s, q=q, labels=labels, duplicates="drop")
    except Exception:
        return pd.Series([pd.NA] * len(series), index=series.index, dtype="object")


def add_buckets_and_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 主要目的変数: イベント単位の交代後1〜3試合の勝点残差合計
    if "sum_points_residual_after_1_3" in df.columns:
        residual = pd.to_numeric(df["sum_points_residual_after_1_3"], errors="coerce")
        df["boost_flag_after_1_3"] = residual > 0
        df["strong_boost_flag_after_1_3"] = residual >= 2.0
        df["strong_negative_flag_after_1_3"] = residual <= -2.0

    if "sum_points_residual_after_1_5" in df.columns:
        residual5 = pd.to_numeric(df["sum_points_residual_after_1_5"], errors="coerce")
        df["boost_flag_after_1_5"] = residual5 > 0
        df["strong_boost_flag_after_1_5"] = residual5 >= 3.0

    # 交代時期
    mpg = pd.to_numeric(df.get("matches_played_at_change"), errors="coerce")
    df["timing_bucket"] = pd.cut(
        mpg,
        bins=[-1, 8, 16, 24, 999],
        labels=["00_08", "09_16", "17_24", "25_plus"],
    )

    # Elo・期待勝点・相手強度は四分位
    if "team_elo_at_cutoff" in df.columns:
        df["elo_bucket"] = qcut_safe(
            df["team_elo_at_cutoff"], 4,
            ["Q1_low_elo", "Q2_mid_low_elo", "Q3_mid_high_elo", "Q4_high_elo"],
        )
    if "sum_expected_points_after_1_3" in df.columns:
        df["expected_points_bucket_after_1_3"] = qcut_safe(
            df["sum_expected_points_after_1_3"], 4,
            ["Q1_low_expected", "Q2_mid_low_expected", "Q3_mid_high_expected", "Q4_high_expected"],
        )
    if "opponent_elo_mean_after_1_3" in df.columns:
        df["opponent_elo_bucket_after_1_3"] = qcut_safe(
            df["opponent_elo_mean_after_1_3"], 4,
            ["Q1_easy_schedule", "Q2_mid_easy", "Q3_mid_hard", "Q4_hard_schedule"],
        )

    # 直近5試合の勝点
    if "last5_points" in df.columns:
        l5 = pd.to_numeric(df["last5_points"], errors="coerce")
        df["last5_points_bucket"] = pd.cut(
            l5,
            bins=[-1, 3, 6, 9, 15],
            labels=["0_3_poor", "4_6_bad", "7_9_neutral", "10_15_good"],
        )

    # 交代前の状態分類: ざっくりした自動分類
    def classify_context(row: pd.Series) -> str:
        last5_gf = row.get("last5_gf", np.nan)
        last5_ga = row.get("last5_ga", np.nan)
        last5_points = row.get("last5_points", np.nan)
        pre_ppg = row.get("pre_ppg", np.nan)

        attack_bad = pd.notna(last5_gf) and last5_gf <= 4
        defense_bad = pd.notna(last5_ga) and last5_ga >= 9
        poor_form = pd.notna(last5_points) and last5_points <= 3
        low_base = pd.notna(pre_ppg) and pre_ppg < 1.0

        if attack_bad and defense_bad:
            return "both_attack_defense_bad"
        if defense_bad:
            return "defense_collapse"
        if attack_bad:
            return "attack_stagnation"
        if poor_form and not low_base:
            return "poor_form_possible_regression"
        if low_base:
            return "low_base_strength"
        return "mixed_or_unclear"

    df["pre_context_auto"] = df.apply(classify_context, axis=1)
    return df


def make_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame()

    def mean_or_nan(s):
        return pd.to_numeric(s, errors="coerce").mean()

    residual_col = "sum_points_residual_after_1_3"
    residual5_col = "sum_points_residual_after_1_5"
    gf_col = "mean_goal_for_residual_after_1_3"
    ga_col = "mean_goals_against_improvement_after_1_3"

    agg = df.groupby(group_col, dropna=False).agg(
        n_events=("event_id", "count"),
        mean_sum_points_residual_after_1_3=(residual_col, mean_or_nan),
        median_sum_points_residual_after_1_3=(residual_col, "median"),
        mean_sum_points_residual_after_1_5=(residual5_col, mean_or_nan),
        mean_goal_for_residual_after_1_3=(gf_col, mean_or_nan),
        mean_goals_against_improvement_after_1_3=(ga_col, mean_or_nan),
        boost_rate_after_1_3=("boost_flag_after_1_3", "mean"),
        strong_boost_rate_after_1_3=("strong_boost_flag_after_1_3", "mean"),
        strong_negative_rate_after_1_3=("strong_negative_flag_after_1_3", "mean"),
    ).reset_index()

    # 見やすさのため率は%ではなく0-1のまま。HTMLでは丸める。
    return agg.sort_values("n_events", ascending=False)


def export_html(dataset: pd.DataFrame, summaries: dict[str, pd.DataFrame]) -> None:
    key_cols = [
        "event_id", "season", "team", "old_manager", "new_manager",
        "effective_change_date", "matches_played_at_change",
        "sum_points_residual_after_1_3", "sum_points_residual_after_1_5",
        "team_elo_at_cutoff", "opponent_elo_mean_after_1_3",
        "timing_bucket", "expected_points_bucket_after_1_3", "elo_bucket",
        "last5_points", "pre_context_auto",
    ]
    key_cols = [c for c in key_cols if c in dataset.columns]

    def table_html(df: pd.DataFrame, max_rows: int = 30) -> str:
        if df is None or df.empty:
            return "<p>該当データなし</p>"
        return df.head(max_rows).round(3).to_html(index=False, classes="result-table", border=0)

    top_pos = dataset.sort_values("sum_points_residual_after_1_3", ascending=False)[key_cols].head(15)
    top_neg = dataset.sort_values("sum_points_residual_after_1_3", ascending=True)[key_cols].head(15)

    sections = []
    for title, df in summaries.items():
        sections.append(f"<h2>{title}</h2>{table_html(df)}")

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>監督交代ブースト 条件探索</title>
  <style>
    body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; line-height: 1.7; padding: 24px; background: #f7f7fb; color: #222; }}
    main {{ max-width: 1180px; margin: 0 auto; }}
    section {{ background: #fff; padding: 20px; margin: 18px 0; border-radius: 14px; box-shadow: 0 2px 8px rgba(0,0,0,.05); }}
    .result-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    .result-table th, .result-table td {{ border: 1px solid #ddd; padding: 7px 9px; text-align: center; white-space: nowrap; }}
    .result-table th {{ background: #222; color: #fff; }}
    .scroll {{ overflow-x: auto; }}
  </style>
</head>
<body>
<main>
  <h1>監督交代ブースト 条件探索</h1>
  <section>
    <h2>概要</h2>
    <p>監督交代イベントを1件1行に整形し、交代時期・Elo・期待勝点・相手強度・直近不調度ごとに、交代後1〜3試合の勝点残差を比較する。</p>
    <p>イベント数: <strong>{len(dataset)}</strong></p>
  </section>
  <section class="scroll">
    {''.join(sections)}
  </section>
  <section class="scroll">
    <h2>上振れ上位</h2>
    {table_html(top_pos, 20)}
  </section>
  <section class="scroll">
    <h2>下振れ上位</h2>
    {table_html(top_neg, 20)}
  </section>
</main>
</body>
</html>"""
    OUTPUT_HTML.write_text(html, encoding="utf-8")


def main() -> None:
    by_event, matches, events, hist = load_inputs()
    wide = make_event_wide(by_event)
    wide = add_match_level_features(wide, matches)
    wide = add_pre_form_features(wide, events, hist)
    dataset = add_buckets_and_flags(wide)

    # 日付列をCSVで見やすくする
    for col in dataset.columns:
        if "date" in col:
            dataset[col] = pd.to_datetime(dataset[col], errors="coerce").dt.strftime("%Y-%m-%d")

    summaries = {
        "交代時期別": make_summary(dataset, "timing_bucket"),
        "期待勝点別": make_summary(dataset, "expected_points_bucket_after_1_3"),
        "Elo別": make_summary(dataset, "elo_bucket"),
        "相手Elo別": make_summary(dataset, "opponent_elo_bucket_after_1_3"),
        "直近5試合勝点別": make_summary(dataset, "last5_points_bucket"),
        "交代前状態分類別": make_summary(dataset, "pre_context_auto"),
    }

    dataset.to_csv(OUTPUT_DATASET, index=False, encoding="utf-8-sig")
    summaries["交代時期別"].to_csv(OUTPUT_SUMMARY_TIMING, index=False, encoding="utf-8-sig")
    summaries["期待勝点別"].to_csv(OUTPUT_SUMMARY_EXPECTED, index=False, encoding="utf-8-sig")
    summaries["Elo別"].to_csv(OUTPUT_SUMMARY_ELO, index=False, encoding="utf-8-sig")
    summaries["相手Elo別"].to_csv(OUTPUT_SUMMARY_OPP_ELO, index=False, encoding="utf-8-sig")
    summaries["直近5試合勝点別"].to_csv(OUTPUT_SUMMARY_LAST5, index=False, encoding="utf-8-sig")
    summaries["交代前状態分類別"].to_csv(OUTPUT_SUMMARY_CONTEXT, index=False, encoding="utf-8-sig")

    key_cols = [
        "event_id", "season", "team", "old_manager", "new_manager",
        "effective_change_date", "matches_played_at_change",
        "sum_points_residual_after_1_3", "sum_points_residual_after_1_5",
        "mean_goal_for_residual_after_1_3", "mean_goals_against_improvement_after_1_3",
        "team_elo_at_cutoff", "opponent_elo_mean_after_1_3",
        "last5_points", "last5_gd", "pre_context_auto",
    ]
    key_cols = [c for c in key_cols if c in dataset.columns]
    dataset.sort_values("sum_points_residual_after_1_3", ascending=False)[key_cols].head(30).to_csv(
        OUTPUT_TOP_POSITIVE, index=False, encoding="utf-8-sig"
    )
    dataset.sort_values("sum_points_residual_after_1_3", ascending=True)[key_cols].head(30).to_csv(
        OUTPUT_TOP_NEGATIVE, index=False, encoding="utf-8-sig"
    )

    export_html(dataset, summaries)

    print("=== 出力完了 ===")
    for p in [
        OUTPUT_DATASET,
        OUTPUT_SUMMARY_TIMING,
        OUTPUT_SUMMARY_EXPECTED,
        OUTPUT_SUMMARY_ELO,
        OUTPUT_SUMMARY_OPP_ELO,
        OUTPUT_SUMMARY_LAST5,
        OUTPUT_SUMMARY_CONTEXT,
        OUTPUT_TOP_POSITIVE,
        OUTPUT_TOP_NEGATIVE,
        OUTPUT_HTML,
    ]:
        print(p.name)

    print("\n=== 交代時期別 ===")
    print(summaries["交代時期別"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
