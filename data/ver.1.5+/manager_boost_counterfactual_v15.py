import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# ============================================================
# 監督交代ブースト検証 B案：cutoff反実仮想 ver1.5型
# ------------------------------------------------------------
# 目的:
#   各監督交代イベントについて、前任監督の最終戦までの情報だけで
#   ver1.5型の期待得点・勝敗確率を再計算し、
#   新監督初戦以降の実績と比較する。
#
# 入力:
#   manager_events_j1_for_boost_v2.csv
#   j1_historical_results_1993_2025_table_fixed.csv
#     または j1_j2_elo_input_1993_2025.csv
#
# 出力:
#   manager_boost_counterfactual_matches_v15.csv
#   manager_boost_counterfactual_summary_v15.csv
#   manager_boost_counterfactual_by_event_v15.csv
#   manager_boost_counterfactual_by_year_v15.csv
#   manager_boost_counterfactual_unmatched_v15.csv
#   manager_boost_counterfactual_summary_v15.html
# ============================================================


# =========================
# 1. 設定
# =========================

BASE_DIR = Path(__file__).resolve().parent

EVENTS_CSV_CANDIDATES = [
    "manager_events_j1_for_boost_v2.csv",
    "manager_changes_inferred_v2.csv",
]

HISTORICAL_CSV_CANDIDATES = [
    "j1_historical_results_1993_2025_table_fixed.csv",
    "j1_j2_elo_input_1993_2025.csv",
]

MODEL_VERSION = "v1.5_counterfactual"

# 対象イベントはJ1のみ
TARGET_COMPETITION = "J1"
TARGET_DIVISION = "J1"

# ver1.5 固定設定
PREV_WEIGHT = 0.20
PREV_DECAY = 0.995
ELO_LAMBDA_WEIGHT = 0.20
COMPAT_WEIGHT = 0.20
CURRENT_DECAY = 1.000
SHRINKAGE = 1.00
DRAW_FACTOR = 1.20
MAX_MATCH_DRAW_PROB = 0.33
LAMBDA_CAP = 3.5

# 大勝補正 cap4
GOAL_ADJUST_MODE = "cap"
GOAL_CAP_FOR_STRENGTH = 4

# 昇格組補正は不採用
USE_PROMOTED_PREV_ZERO = True
PROMOTED_PREV_WEIGHT = 0.00
PROMOTED_ATTACK_PRIOR = 1.00
PROMOTED_DEFENSE_PRIOR = 1.00

# 相性Effect設定
MATCHUP_PRIOR_N = 30
MATCHUP_TIME_DECAY = 0.97
COMPAT_FACTOR_MIN = 0.95
COMPAT_FACTOR_MAX = 1.05

# λの基準となるリーグ平均得点はraw得点を使う
USE_RAW_LEAGUE_AVG_FOR_LAMBDA = True

# Elo設定
INITIAL_ELO = 1500
K_FACTOR = 16
HOME_ADV = 0
ELO_FACTOR_SCALE = 4000
ELO_FACTOR_MIN = 0.90
ELO_FACTOR_MAX = 1.10

# スコア確率グリッド
MAX_GOALS_FOR_SCORE_GRID = 10

# 反実仮想の対象範囲
MAX_GAMES_AFTER_CHANGE = 10
STOP_AT_NEXT_MANAGER_CHANGE = True

# 集計ウィンドウ
WINDOWS = [
    ("after_1_3", 1, 3),
    ("after_1_5", 1, 5),
    ("after_1_10", 1, 10),
    ("after_4_6", 4, 6),
    ("after_7_10", 7, 10),
]

# 出力
OUTPUT_MATCHES_CSV = BASE_DIR / "manager_boost_counterfactual_matches_v15.csv"
OUTPUT_SUMMARY_CSV = BASE_DIR / "manager_boost_counterfactual_summary_v15.csv"
OUTPUT_BY_EVENT_CSV = BASE_DIR / "manager_boost_counterfactual_by_event_v15.csv"
OUTPUT_BY_YEAR_CSV = BASE_DIR / "manager_boost_counterfactual_by_year_v15.csv"
OUTPUT_UNMATCHED_CSV = BASE_DIR / "manager_boost_counterfactual_unmatched_v15.csv"
OUTPUT_HTML = BASE_DIR / "manager_boost_counterfactual_summary_v15.html"


# =========================
# 2. 汎用処理
# =========================


def find_file(candidates):
    """候補ファイルを、スクリプト直下・dataフォルダ・作業ディレクトリから探す。"""
    search_dirs = [
        BASE_DIR,
        BASE_DIR / "data",
        BASE_DIR.parent / "data",
        BASE_DIR.parent,
        Path.cwd(),
        Path.cwd() / "data",
    ]
    for filename in candidates:
        for d in search_dirs:
            path = d / filename
            if path.exists():
                return path
    raise FileNotFoundError(f"候補ファイルが見つかりません: {candidates}")


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


def clean_team_names(df):
    df = df.copy()
    for col in ["home", "away", "team"]:
        if col in df.columns:
            df[col] = df[col].apply(standardize_team_name)
    return df


# =========================
# 3. データ読み込み
# =========================


def load_historical_csv(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = clean_team_names(df)

    # j1_j2_elo_input_1993_2025.csv の場合は division でJ1に絞る
    if "division" in df.columns:
        df = df[df["division"].astype(str) == TARGET_DIVISION].copy()

    if "competition" in df.columns:
        df = df[df["competition"].astype(str) == TARGET_COMPETITION].copy()

    if "date" not in df.columns:
        raise ValueError("historical CSVに date 列が必要です。")

    df["date"] = (
        df["date"]
        .astype(str)
        .str.replace(r"\(.*\)", "", regex=True)
        .str.strip()
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "home_goal" in df.columns and "away_goal" in df.columns:
        df["home_goal"] = pd.to_numeric(df["home_goal"], errors="coerce")
        df["away_goal"] = pd.to_numeric(df["away_goal"], errors="coerce")
    elif "home_goals" in df.columns and "away_goals" in df.columns:
        df["home_goal"] = pd.to_numeric(df["home_goals"], errors="coerce")
        df["away_goal"] = pd.to_numeric(df["away_goals"], errors="coerce")
    else:
        raise ValueError("historical CSVに home_goal/away_goal または home_goals/away_goals が必要です。")

    df = df.dropna(subset=["date", "home", "away", "home_goal", "away_goal"]).copy()
    df["home_goal"] = df["home_goal"].astype(int)
    df["away_goal"] = df["away_goal"].astype(int)

    if "year" not in df.columns:
        df["year"] = df["date"].dt.year.astype(int)
    else:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(df["date"].dt.year).astype(int)

    df = df.sort_values(["date", "home", "away"]).reset_index(drop=True)
    return df


def load_events(path):
    events = pd.read_csv(path, encoding="utf-8-sig")
    events = clean_team_names(events)

    required_cols = [
        "season",
        "team",
        "old_manager",
        "new_manager",
        "last_old_manager_match_date",
        "first_new_manager_match_date",
        "effective_change_date",
    ]
    missing = [c for c in required_cols if c not in events.columns]
    if missing:
        raise ValueError(f"events CSVに必要な列がありません: {missing}")

    if "competition" in events.columns:
        events = events[events["competition"].astype(str) == TARGET_COMPETITION].copy()

    if "review_status" in events.columns:
        events = events[events["review_status"].astype(str).isin(["likely_include", "include", "manual_include"])].copy()

    events["season"] = pd.to_numeric(events["season"], errors="coerce").astype(int)
    for col in ["last_old_manager_match_date", "first_new_manager_match_date", "effective_change_date"]:
        events[col] = pd.to_datetime(events[col], errors="coerce")

    events = events.dropna(subset=["season", "team", "last_old_manager_match_date", "effective_change_date"]).copy()
    events = events.sort_values(["season", "team", "effective_change_date"]).reset_index(drop=True)
    events.insert(0, "event_id", [f"E{i:03d}" for i in range(1, len(events) + 1)])

    # 同一チーム・同一シーズンの次の監督交代日を付与する。
    events["next_change_date"] = events.groupby(["season", "team"])["effective_change_date"].shift(-1)
    return events


# =========================
# 4. 順位・勝点
# =========================


def get_points_for_match(home_goal, away_goal):
    if home_goal > away_goal:
        return 3, 0
    if home_goal < away_goal:
        return 0, 3
    return 1, 1


# =========================
# 5. 攻守係数
# =========================


def safe_positive_mean(series, fallback=1.0):
    value = pd.to_numeric(series, errors="coerce").mean()
    if not np.isfinite(value) or value <= 0:
        return fallback
    return float(value)


def safe_strength(value):
    if not np.isfinite(value) or value <= 0:
        return 1.0
    return float(value)


def add_goal_for_strength_columns(df, goal_adjust_mode="raw", goal_cap_for_strength=None):
    df = df.copy()
    if goal_adjust_mode == "raw":
        df["home_goal_strength"] = df["home_goal"].astype(float)
        df["away_goal_strength"] = df["away_goal"].astype(float)
    elif goal_adjust_mode == "cap":
        if goal_cap_for_strength is None:
            raise ValueError("goal_adjust_mode='cap' の場合、goal_cap_for_strength が必要です。")
        cap = float(goal_cap_for_strength)
        df["home_goal_strength"] = df["home_goal"].astype(float).clip(upper=cap)
        df["away_goal_strength"] = df["away_goal"].astype(float).clip(upper=cap)
    else:
        raise ValueError("goal_adjust_mode は 'raw' または 'cap' を指定してください。")
    return df


def calculate_strengths_home_away_goals_only(
    history_df,
    teams,
    decay=1.0,
    goal_adjust_mode="raw",
    goal_cap_for_strength=None,
):
    history_df = history_df.copy().sort_values("date").reset_index(drop=True)
    history_df = add_goal_for_strength_columns(
        history_df,
        goal_adjust_mode=goal_adjust_mode,
        goal_cap_for_strength=goal_cap_for_strength,
    )

    raw_home_avg_goals = safe_positive_mean(history_df["home_goal"], fallback=1.0)
    raw_away_avg_goals = safe_positive_mean(history_df["away_goal"], fallback=1.0)
    strength_home_avg_goals = safe_positive_mean(history_df["home_goal_strength"], fallback=1.0)
    strength_away_avg_goals = safe_positive_mean(history_df["away_goal_strength"], fallback=1.0)

    if USE_RAW_LEAGUE_AVG_FOR_LAMBDA:
        home_avg_goals = raw_home_avg_goals
        away_avg_goals = raw_away_avg_goals
    else:
        home_avg_goals = strength_home_avg_goals
        away_avg_goals = strength_away_avg_goals

    strengths = {}
    for team in teams:
        home_games = history_df[history_df["home"] == team].copy().sort_values("date", ascending=False)
        home_gf = home_ga = home_w = 0.0
        for i, row in enumerate(home_games.itertuples(index=False)):
            rd = row._asdict()
            weight = decay ** i
            home_gf += float(rd["home_goal_strength"]) * weight
            home_ga += float(rd["away_goal_strength"]) * weight
            home_w += weight

        if home_w > 0:
            home_attack = (home_gf / home_w) / strength_home_avg_goals
            home_defense = (home_ga / home_w) / strength_away_avg_goals
        else:
            home_attack = 1.0
            home_defense = 1.0

        away_games = history_df[history_df["away"] == team].copy().sort_values("date", ascending=False)
        away_gf = away_ga = away_w = 0.0
        for i, row in enumerate(away_games.itertuples(index=False)):
            rd = row._asdict()
            weight = decay ** i
            away_gf += float(rd["away_goal_strength"]) * weight
            away_ga += float(rd["home_goal_strength"]) * weight
            away_w += weight

        if away_w > 0:
            away_attack = (away_gf / away_w) / strength_away_avg_goals
            away_defense = (away_ga / away_w) / strength_home_avg_goals
        else:
            away_attack = 1.0
            away_defense = 1.0

        strengths[team] = {
            "home_attack": safe_strength(home_attack),
            "home_defense": safe_strength(home_defense),
            "away_attack": safe_strength(away_attack),
            "away_defense": safe_strength(away_defense),
        }

    return strengths, home_avg_goals, away_avg_goals


def count_team_games(df, team):
    if df.empty:
        return 0
    return int(((df["home"] == team) | (df["away"] == team)).sum())


def blend_with_previous_strengths(
    current_strengths,
    previous_strengths,
    prev_weight,
    previous_df,
    use_promoted_prev_zero=True,
    promoted_prev_weight=0.0,
    promoted_attack_prior=1.0,
    promoted_defense_prior=1.0,
):
    blended = {}
    prev_weight_by_team = {}
    prev_games_by_team = {}

    def safe_blend(current_value, previous_value, weight):
        current_ok = np.isfinite(current_value)
        previous_ok = np.isfinite(previous_value)
        if current_ok and previous_ok:
            return (1 - weight) * current_value + weight * previous_value
        if current_ok:
            return current_value
        if previous_ok:
            return previous_value
        return 1.0

    for team in current_strengths:
        prev_games = count_team_games(previous_df, team)
        effective_prev_weight = prev_weight
        if use_promoted_prev_zero and prev_games == 0:
            effective_prev_weight = promoted_prev_weight

        prev_weight_by_team[team] = effective_prev_weight
        prev_games_by_team[team] = prev_games

        prev = previous_strengths.get(team, {
            "home_attack": 1.0,
            "home_defense": 1.0,
            "away_attack": 1.0,
            "away_defense": 1.0,
        })
        if use_promoted_prev_zero and prev_games == 0:
            prev = {
                "home_attack": float(promoted_attack_prior),
                "home_defense": float(promoted_defense_prior),
                "away_attack": float(promoted_attack_prior),
                "away_defense": float(promoted_defense_prior),
            }

        blended[team] = {
            "home_attack": safe_blend(current_strengths[team]["home_attack"], prev["home_attack"], effective_prev_weight),
            "home_defense": safe_blend(current_strengths[team]["home_defense"], prev["home_defense"], effective_prev_weight),
            "away_attack": safe_blend(current_strengths[team]["away_attack"], prev["away_attack"], effective_prev_weight),
            "away_defense": safe_blend(current_strengths[team]["away_defense"], prev["away_defense"], effective_prev_weight),
        }

    return blended, prev_weight_by_team, prev_games_by_team


def apply_strength_shrinkage(strengths, shrinkage=1.0):
    shrinkage = float(np.clip(float(shrinkage), 0.0, 1.5))
    shrunk = {}
    for team, values in strengths.items():
        shrunk[team] = {}
        for key, value in values.items():
            value = safe_strength(value)
            new_value = 1.0 + (value - 1.0) * shrinkage
            shrunk[team][key] = safe_strength(new_value)
    return shrunk


# =========================
# 6. Elo
# =========================


def get_actual_score_from_goals(home_goal, away_goal):
    if home_goal > away_goal:
        return 1.0
    if home_goal < away_goal:
        return 0.0
    return 0.5


def update_elo_one_match(ratings, home, away, home_goal, away_goal, k_factor=16, home_adv=0):
    if home not in ratings:
        ratings[home] = INITIAL_ELO
    if away not in ratings:
        ratings[away] = INITIAL_ELO

    home_rating = ratings[home]
    away_rating = ratings[away]
    expected_home = 1 / (1 + 10 ** ((away_rating - (home_rating + home_adv)) / 400))
    actual_home = get_actual_score_from_goals(home_goal, away_goal)

    goal_diff = abs(home_goal - away_goal)
    margin_multiplier = 1.0 if goal_diff <= 1 else np.log(goal_diff) + 1.0
    change = k_factor * margin_multiplier * (actual_home - expected_home)

    ratings[home] = home_rating + change
    ratings[away] = away_rating - change
    return expected_home, actual_home


def build_elo_ratings(previous_df, train_df, teams, k_factor=16, home_adv=0):
    all_teams = pd.unique(
        pd.concat([
            previous_df[["home", "away"]],
            train_df[["home", "away"]],
        ], ignore_index=True).values.ravel()
    )
    ratings = {team: INITIAL_ELO for team in all_teams}
    combined = pd.concat([previous_df, train_df], ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)

    for row in combined.itertuples(index=False):
        rd = row._asdict()
        update_elo_one_match(
            ratings,
            rd["home"],
            rd["away"],
            int(rd["home_goal"]),
            int(rd["away_goal"]),
            k_factor=k_factor,
            home_adv=home_adv,
        )
    return {team: ratings.get(team, INITIAL_ELO) for team in teams}


# =========================
# 7. 相性Effect
# =========================


def build_matchup_effects_j1(
    historical_df,
    cutoff_date,
    target_year,
    prior_n=30,
    time_decay=0.97,
    k_factor=16,
    home_adv=0,
):
    df = historical_df[historical_df["date"] <= cutoff_date].copy()
    df = df.sort_values("date").reset_index(drop=True)

    ratings = {}
    weighted_residual_sum = {}
    weighted_match_sum = {}

    def get_rating(team):
        if team not in ratings:
            ratings[team] = INITIAL_ELO
        return ratings[team]

    def add_effect(key, residual, weight):
        weighted_residual_sum[key] = weighted_residual_sum.get(key, 0.0) + residual * weight
        weighted_match_sum[key] = weighted_match_sum.get(key, 0.0) + weight

    for row in df.itertuples(index=False):
        rd = row._asdict()
        home = rd["home"]
        away = rd["away"]
        hg = int(rd["home_goal"])
        ag = int(rd["away_goal"])

        home_rating = get_rating(home)
        away_rating = get_rating(away)
        expected_home = 1 / (1 + 10 ** ((away_rating - (home_rating + home_adv)) / 400))
        actual_home = get_actual_score_from_goals(hg, ag)

        residual_home = actual_home - expected_home
        residual_away = -residual_home

        years_ago = max(0, int(target_year) - int(rd["year"]))
        weight = time_decay ** years_ago

        add_effect((home, away), residual_home, weight)
        add_effect((away, home), residual_away, weight)

        update_elo_one_match(
            ratings,
            home,
            away,
            hg,
            ag,
            k_factor=k_factor,
            home_adv=home_adv,
        )

    effects = {}
    for key in weighted_residual_sum:
        weighted_n = weighted_match_sum[key]
        raw_effect = weighted_residual_sum[key] / weighted_n if weighted_n > 0 else 0.0
        shrink = weighted_n / (weighted_n + prior_n)
        effects[key] = raw_effect * shrink
    return effects


# =========================
# 8. 期待得点・勝敗確率
# =========================


def safe_lambda(lam):
    if lam is None or not np.isfinite(lam) or lam < 0:
        return 0.05
    if LAMBDA_CAP is not None:
        lam = min(lam, LAMBDA_CAP)
    return float(max(lam, 0.05))


def expected_goals_home_away(
    home,
    away,
    strengths,
    home_avg_goals,
    away_avg_goals,
    elo_ratings=None,
    elo_lambda_weight=0.0,
    matchup_effects=None,
    compat_weight=0.0,
):
    lambda_home = strengths[home]["home_attack"] * strengths[away]["away_defense"] * home_avg_goals
    lambda_away = strengths[away]["away_attack"] * strengths[home]["home_defense"] * away_avg_goals

    if elo_ratings is not None and elo_lambda_weight > 0:
        home_elo = elo_ratings.get(home, INITIAL_ELO)
        away_elo = elo_ratings.get(away, INITIAL_ELO)
        elo_diff = home_elo - away_elo
        home_factor = 10 ** ((elo_lambda_weight * elo_diff) / ELO_FACTOR_SCALE)
        away_factor = 10 ** ((-elo_lambda_weight * elo_diff) / ELO_FACTOR_SCALE)
        home_factor = np.clip(home_factor, ELO_FACTOR_MIN, ELO_FACTOR_MAX)
        away_factor = np.clip(away_factor, ELO_FACTOR_MIN, ELO_FACTOR_MAX)
        lambda_home *= home_factor
        lambda_away *= away_factor

    if matchup_effects is not None and compat_weight > 0:
        home_effect = matchup_effects.get((home, away), 0.0)
        away_effect = matchup_effects.get((away, home), 0.0)
        home_factor = np.clip(1 + compat_weight * home_effect, COMPAT_FACTOR_MIN, COMPAT_FACTOR_MAX)
        away_factor = np.clip(1 + compat_weight * away_effect, COMPAT_FACTOR_MIN, COMPAT_FACTOR_MAX)
        lambda_home *= home_factor
        lambda_away *= away_factor

    return safe_lambda(lambda_home), safe_lambda(lambda_away)


def poisson_pmf_array(lam, max_goals):
    probs = np.zeros(max_goals + 1, dtype=float)
    probs[0] = np.exp(-lam)
    for k in range(1, max_goals + 1):
        probs[k] = probs[k - 1] * lam / k
    return probs


def score_probability_matrix_with_draw_factor(lambda_home, lambda_away, draw_factor, max_goals=10, max_match_draw_prob=None):
    home_probs = poisson_pmf_array(lambda_home, max_goals)
    away_probs = poisson_pmf_array(lambda_away, max_goals)
    score_matrix = np.outer(home_probs, away_probs)

    total_prob = score_matrix.sum()
    if not np.isfinite(total_prob) or total_prob <= 0:
        score_matrix = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
        score_matrix[0, 0] = 1.0
        return score_matrix

    score_matrix = score_matrix / total_prob
    diag_idx = np.arange(max_goals + 1)
    draw_mask = np.zeros_like(score_matrix, dtype=bool)
    draw_mask[diag_idx, diag_idx] = True

    base_draw_prob = float(score_matrix[draw_mask].sum())
    base_non_draw_prob = 1.0 - base_draw_prob
    if base_draw_prob <= 0 or base_non_draw_prob <= 0:
        return score_matrix / score_matrix.sum()

    target_draw_prob = base_draw_prob * draw_factor
    if max_match_draw_prob is not None:
        target_draw_prob = min(target_draw_prob, float(max_match_draw_prob))

    target_draw_prob = float(np.clip(target_draw_prob, 0.0, 0.95))
    target_non_draw_prob = 1.0 - target_draw_prob

    draw_scale = target_draw_prob / base_draw_prob
    non_draw_scale = target_non_draw_prob / base_non_draw_prob
    score_matrix[draw_mask] *= draw_scale
    score_matrix[~draw_mask] *= non_draw_scale
    score_matrix = score_matrix / score_matrix.sum()
    return score_matrix


def get_probs_from_score_matrix(score_matrix):
    max_goals = score_matrix.shape[0] - 1
    goals = np.arange(max_goals + 1)

    home_win_prob = float(np.tril(score_matrix, k=-1).sum())
    draw_prob = float(np.trace(score_matrix))
    away_win_prob = float(np.triu(score_matrix, k=1).sum())

    expected_home_goals = float((score_matrix * goals[:, None]).sum())
    expected_away_goals = float((score_matrix * goals[None, :]).sum())

    home_expected_points = 3 * home_win_prob + draw_prob
    away_expected_points = 3 * away_win_prob + draw_prob

    return {
        "home_win_prob": home_win_prob,
        "draw_prob": draw_prob,
        "away_win_prob": away_win_prob,
        "home_expected_points": home_expected_points,
        "away_expected_points": away_expected_points,
        "score_grid_expected_home_goals": expected_home_goals,
        "score_grid_expected_away_goals": expected_away_goals,
    }


# =========================
# 9. 反実仮想予測
# =========================


def build_model_state_at_cutoff(historical_df, season, cutoff_date):
    previous_year = int(season) - 1
    previous_df = historical_df[historical_df["year"] == previous_year].copy()
    target_df = historical_df[historical_df["year"] == int(season)].copy()

    if previous_df.empty:
        raise ValueError(f"{previous_year}年のJ1データがありません。")
    if target_df.empty:
        raise ValueError(f"{season}年のJ1データがありません。")

    previous_df = previous_df.sort_values("date").reset_index(drop=True)
    target_df = target_df.sort_values("date").reset_index(drop=True)
    teams = list(pd.unique(target_df[["home", "away"]].values.ravel()))

    # 反実仮想の根幹：cutoff以前の同年試合だけを学習に使う
    train_df = target_df[target_df["date"] <= cutoff_date].copy()

    current_strengths, home_avg_goals, away_avg_goals = calculate_strengths_home_away_goals_only(
        train_df,
        teams=teams,
        decay=CURRENT_DECAY,
        goal_adjust_mode=GOAL_ADJUST_MODE,
        goal_cap_for_strength=GOAL_CAP_FOR_STRENGTH,
    )
    previous_strengths, _, _ = calculate_strengths_home_away_goals_only(
        previous_df,
        teams=teams,
        decay=PREV_DECAY,
        goal_adjust_mode=GOAL_ADJUST_MODE,
        goal_cap_for_strength=GOAL_CAP_FOR_STRENGTH,
    )
    strengths, prev_weight_by_team, prev_games_by_team = blend_with_previous_strengths(
        current_strengths=current_strengths,
        previous_strengths=previous_strengths,
        prev_weight=PREV_WEIGHT,
        previous_df=previous_df,
        use_promoted_prev_zero=USE_PROMOTED_PREV_ZERO,
        promoted_prev_weight=PROMOTED_PREV_WEIGHT,
        promoted_attack_prior=PROMOTED_ATTACK_PRIOR,
        promoted_defense_prior=PROMOTED_DEFENSE_PRIOR,
    )
    strengths = apply_strength_shrinkage(strengths, shrinkage=SHRINKAGE)

    elo_ratings = build_elo_ratings(
        previous_df=previous_df,
        train_df=train_df,
        teams=teams,
        k_factor=K_FACTOR,
        home_adv=HOME_ADV,
    )

    matchup_effects = None
    if COMPAT_WEIGHT > 0:
        matchup_effects = build_matchup_effects_j1(
            historical_df=historical_df,
            cutoff_date=cutoff_date,
            target_year=int(season),
            prior_n=MATCHUP_PRIOR_N,
            time_decay=MATCHUP_TIME_DECAY,
            k_factor=K_FACTOR,
            home_adv=HOME_ADV,
        )

    return {
        "previous_year": previous_year,
        "previous_df": previous_df,
        "target_df": target_df,
        "train_df": train_df,
        "teams": teams,
        "strengths": strengths,
        "home_avg_goals": home_avg_goals,
        "away_avg_goals": away_avg_goals,
        "elo_ratings": elo_ratings,
        "matchup_effects": matchup_effects,
        "prev_weight_by_team": prev_weight_by_team,
        "prev_games_by_team": prev_games_by_team,
    }


def predict_one_match(row_dict, state):
    home = row_dict["home"]
    away = row_dict["away"]

    lambda_home, lambda_away = expected_goals_home_away(
        home=home,
        away=away,
        strengths=state["strengths"],
        home_avg_goals=state["home_avg_goals"],
        away_avg_goals=state["away_avg_goals"],
        elo_ratings=state["elo_ratings"],
        elo_lambda_weight=ELO_LAMBDA_WEIGHT,
        matchup_effects=state["matchup_effects"],
        compat_weight=COMPAT_WEIGHT,
    )

    score_matrix = score_probability_matrix_with_draw_factor(
        lambda_home=lambda_home,
        lambda_away=lambda_away,
        draw_factor=DRAW_FACTOR,
        max_goals=MAX_GOALS_FOR_SCORE_GRID,
        max_match_draw_prob=MAX_MATCH_DRAW_PROB,
    )
    probs = get_probs_from_score_matrix(score_matrix)
    return lambda_home, lambda_away, probs


def get_future_matches_for_event(target_df, event):
    season = int(event["season"])
    team = event["team"]
    effective_date = event["effective_change_date"]
    next_change_date = event.get("next_change_date", pd.NaT)

    mask = (
        (target_df["date"] >= effective_date)
        & ((target_df["home"] == team) | (target_df["away"] == team))
    )

    if STOP_AT_NEXT_MANAGER_CHANGE and pd.notna(next_change_date):
        mask &= target_df["date"] < next_change_date

    future = target_df[mask].copy().sort_values("date").reset_index(drop=True)
    future = future.head(MAX_GAMES_AFTER_CHANGE).copy()
    return future


def run_counterfactual_event_study(historical_df, events_df):
    match_rows = []
    unmatched_rows = []

    for _, event in events_df.iterrows():
        event_id = event["event_id"]
        season = int(event["season"])
        team = event["team"]
        cutoff_date = event["last_old_manager_match_date"]
        effective_date = event["effective_change_date"]

        print(f"{event_id} {season} {team}: cutoff={cutoff_date.date()} effective={effective_date.date()}")

        try:
            state = build_model_state_at_cutoff(
                historical_df=historical_df,
                season=season,
                cutoff_date=cutoff_date,
            )
        except Exception as e:
            unmatched_rows.append({
                "event_id": event_id,
                "season": season,
                "team": team,
                "reason": f"model_state_error: {e}",
            })
            print(f"  [WARN] モデル状態作成失敗: {e}")
            continue

        if team not in state["teams"]:
            unmatched_rows.append({
                "event_id": event_id,
                "season": season,
                "team": team,
                "reason": "team_not_in_target_year",
            })
            print("  [WARN] target_yearにチームがありません")
            continue

        future = get_future_matches_for_event(state["target_df"], event)
        if future.empty:
            unmatched_rows.append({
                "event_id": event_id,
                "season": season,
                "team": team,
                "reason": "no_future_matches_after_effective_date",
            })
            print("  [WARN] 新監督初戦以降の対象試合がありません")
            continue

        print(f"  対象試合数: {len(future)}")

        for games_after, row in enumerate(future.itertuples(index=False), start=1):
            rd = row._asdict()
            home = rd["home"]
            away = rd["away"]
            actual_hg = int(rd["home_goal"])
            actual_ag = int(rd["away_goal"])

            lambda_home, lambda_away, probs = predict_one_match(rd, state)
            home_actual_points, away_actual_points = get_points_for_match(actual_hg, actual_ag)

            if team == home:
                opponent = away
                is_home = True
                lambda_for = lambda_home
                lambda_against = lambda_away
                expected_goals_for = probs["score_grid_expected_home_goals"]
                expected_goals_against = probs["score_grid_expected_away_goals"]
                expected_points = probs["home_expected_points"]
                actual_points = home_actual_points
                actual_goals_for = actual_hg
                actual_goals_against = actual_ag
                team_win_prob = probs["home_win_prob"]
                team_draw_prob = probs["draw_prob"]
                team_loss_prob = probs["away_win_prob"]
                team_elo = state["elo_ratings"].get(home, INITIAL_ELO)
                opponent_elo = state["elo_ratings"].get(away, INITIAL_ELO)
                matchup_effect = 0.0 if state["matchup_effects"] is None else state["matchup_effects"].get((home, away), 0.0)
            else:
                opponent = home
                is_home = False
                lambda_for = lambda_away
                lambda_against = lambda_home
                expected_goals_for = probs["score_grid_expected_away_goals"]
                expected_goals_against = probs["score_grid_expected_home_goals"]
                expected_points = probs["away_expected_points"]
                actual_points = away_actual_points
                actual_goals_for = actual_ag
                actual_goals_against = actual_hg
                team_win_prob = probs["away_win_prob"]
                team_draw_prob = probs["draw_prob"]
                team_loss_prob = probs["home_win_prob"]
                team_elo = state["elo_ratings"].get(away, INITIAL_ELO)
                opponent_elo = state["elo_ratings"].get(home, INITIAL_ELO)
                matchup_effect = 0.0 if state["matchup_effects"] is None else state["matchup_effects"].get((away, home), 0.0)

            match_rows.append({
                "model_version": MODEL_VERSION,
                "event_id": event_id,
                "season": season,
                "team": team,
                "old_manager": event["old_manager"],
                "new_manager": event["new_manager"],
                "last_old_manager_match_date": cutoff_date,
                "effective_change_date": effective_date,
                "next_change_date": event.get("next_change_date", pd.NaT),
                "stop_at_next_manager_change": STOP_AT_NEXT_MANAGER_CHANGE,
                "games_after_change": games_after,
                "date": rd["date"],
                "home": home,
                "away": away,
                "opponent": opponent,
                "is_home": is_home,
                "actual_home_goal": actual_hg,
                "actual_away_goal": actual_ag,
                "actual_goals_for": actual_goals_for,
                "actual_goals_against": actual_goals_against,
                "lambda_home": lambda_home,
                "lambda_away": lambda_away,
                "lambda_for": lambda_for,
                "lambda_against": lambda_against,
                "expected_goals_for_score_grid": expected_goals_for,
                "expected_goals_against_score_grid": expected_goals_against,
                "home_win_prob": probs["home_win_prob"],
                "draw_prob": probs["draw_prob"],
                "away_win_prob": probs["away_win_prob"],
                "team_win_prob": team_win_prob,
                "team_draw_prob": team_draw_prob,
                "team_loss_prob": team_loss_prob,
                "expected_points": expected_points,
                "actual_points": actual_points,
                "points_residual": actual_points - expected_points,
                "goal_for_residual": actual_goals_for - lambda_for,
                "goal_against_residual": actual_goals_against - lambda_against,
                "goals_against_improvement": lambda_against - actual_goals_against,
                "goal_diff_residual": (actual_goals_for - actual_goals_against) - (lambda_for - lambda_against),
                "team_elo_at_cutoff": team_elo,
                "opponent_elo_at_cutoff": opponent_elo,
                "matchup_effect_at_cutoff": matchup_effect,
                "n_train_matches_at_cutoff": len(state["train_df"]),
                "team_games_in_train_at_cutoff": count_team_games(state["train_df"], team),
                "team_prev_games": state["prev_games_by_team"].get(team, 0),
                "team_effective_prev_weight": state["prev_weight_by_team"].get(team, PREV_WEIGHT),
            })

    matches_df = pd.DataFrame(match_rows)
    unmatched_df = pd.DataFrame(unmatched_rows)
    return matches_df, unmatched_df


# =========================
# 10. 集計
# =========================


def ci95(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    n = len(s)
    if n == 0:
        return np.nan, np.nan
    if n == 1:
        return float(s.iloc[0]), float(s.iloc[0])
    mean = float(s.mean())
    se = float(s.std(ddof=1) / math.sqrt(n))
    return mean - 1.96 * se, mean + 1.96 * se


def summarize_group(df, group_cols):
    if df.empty:
        return pd.DataFrame()

    rows = []
    grouped = df.groupby(group_cols, dropna=False)
    for key, g in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))

        ci_low, ci_high = ci95(g["points_residual"])
        ga_ci_low, ga_ci_high = ci95(g["goals_against_improvement"])

        row.update({
            "n_matches": int(len(g)),
            "n_events": int(g["event_id"].nunique()),
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
            "goals_against_improvement_ci_low": ga_ci_low,
            "goals_against_improvement_ci_high": ga_ci_high,
            "mean_goal_diff_residual": float(g["goal_diff_residual"].mean()),
            "mean_team_win_prob": float(g["team_win_prob"].mean()),
            "mean_team_draw_prob": float(g["team_draw_prob"].mean()),
            "mean_team_loss_prob": float(g["team_loss_prob"].mean()),
            "home_match_rate": float(g["is_home"].mean()),
            "mean_team_games_in_train_at_cutoff": float(g["team_games_in_train_at_cutoff"].mean()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def add_window_labels(matches_df):
    parts = []
    for name, start, end in WINDOWS:
        g = matches_df[
            (matches_df["games_after_change"] >= start)
            & (matches_df["games_after_change"] <= end)
        ].copy()
        if g.empty:
            continue
        g["window"] = name
        g["window_start"] = start
        g["window_end"] = end
        parts.append(g)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def make_summaries(matches_df):
    window_df = add_window_labels(matches_df)
    if window_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    summary = summarize_group(window_df, ["window", "window_start", "window_end"])
    summary = summary.sort_values(["window_start", "window_end"]).reset_index(drop=True)

    by_year = summarize_group(window_df, ["season", "window", "window_start", "window_end"])
    by_year = by_year.sort_values(["season", "window_start", "window_end"]).reset_index(drop=True)

    by_event_cols = [
        "event_id", "season", "team", "old_manager", "new_manager", "effective_change_date",
        "window", "window_start", "window_end",
    ]
    by_event = summarize_group(window_df, by_event_cols)
    by_event = by_event.sort_values(["season", "event_id", "window_start", "window_end"]).reset_index(drop=True)

    return summary, by_year, by_event


# =========================
# 11. HTML出力
# =========================


def format_percent_columns(df):
    df = df.copy()
    for col in ["mean_team_win_prob", "mean_team_draw_prob", "mean_team_loss_prob", "home_match_rate"]:
        if col in df.columns:
            df[col] = df[col].map(lambda x: f"{x:.1%}" if pd.notna(x) else "")
    return df


def round_numeric(df, digits=3):
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and col not in ["season", "window_start", "window_end", "n_matches", "n_events"]:
            df[col] = df[col].round(digits)
    return df


def export_html(summary_df, by_year_df, by_event_df, unmatched_df, output_path):
    summary_display = format_percent_columns(round_numeric(summary_df))
    by_year_display = format_percent_columns(round_numeric(by_year_df))
    by_event_display = format_percent_columns(round_numeric(by_event_df))

    def table_html(df):
        if df is None or df.empty:
            return "<p>データなし</p>"
        return df.to_html(index=False, classes="result-table", escape=False)

    unmatched_count = 0 if unmatched_df is None or unmatched_df.empty else len(unmatched_df)

    html = f"""<!DOCTYPE html>
<html lang=\"ja\">
<head>
  <meta charset=\"UTF-8\">
  <title>監督交代ブースト cutoff反実仮想 | Football Prediction Lab</title>
  <link rel=\"stylesheet\" href=\"style.css\">
  <style>
    .result-table {{ width: 100%; border-collapse: collapse; font-size: 13px; background: white; }}
    .result-table th, .result-table td {{ border: 1px solid #ddd; padding: 8px 10px; text-align: center; white-space: nowrap; }}
    .result-table th {{ background: #222; color: white; }}
    .result-table tbody tr:nth-child(even) {{ background: #f5f5f5; }}
    .table-wrap {{ overflow-x: auto; }}
    .note-box {{ background: #f8fafc; border-left: 5px solid #9400d3; padding: 16px 18px; border-radius: 10px; line-height: 1.8; }}
  </style>
</head>
<body>
  <header>
    <h1>監督交代ブースト cutoff反実仮想</h1>
    <p>前任監督の最終戦までの情報だけで、その後の試合を予測し、実績と比較</p>
  </header>

  <main>
    <section>
      <h2>概要</h2>
      <div class=\"note-box\">
        このページでは、各監督交代イベントについて、前任監督の最後の試合日をcutoffとし、
        その時点までの試合結果のみでver1.5型の期待勝点・期待得点・期待失点を再計算する。
        そのうえで、新監督初戦以降の実績と比較する。
        <br><br>
        これは厳密な因果推論ではなく、
        「前任監督時点のチーム力が続いた場合のモデル予測」と実際の比較である。
        <br><br>
        STOP_AT_NEXT_MANAGER_CHANGE = {STOP_AT_NEXT_MANAGER_CHANGE}<br>
        結合・予測できなかったイベント数: {unmatched_count}
      </div>
    </section>

    <section>
      <h2>ウィンドウ別集計</h2>
      <div class=\"table-wrap\">{table_html(summary_display)}</div>
    </section>

    <section>
      <h2>年度別集計</h2>
      <div class=\"table-wrap\">{table_html(by_year_display)}</div>
    </section>

    <section>
      <h2>監督交代イベント別集計</h2>
      <div class=\"table-wrap\">{table_html(by_event_display)}</div>
    </section>
  </main>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


# =========================
# 12. main
# =========================


def main():
    events_path = find_file(EVENTS_CSV_CANDIDATES)
    historical_path = find_file(HISTORICAL_CSV_CANDIDATES)

    print("events:", events_path)
    print("historical:", historical_path)

    events_df = load_events(events_path)
    historical_df = load_historical_csv(historical_path)

    print(f"events: {len(events_df)}")
    print(f"historical J1 matches: {len(historical_df)}")
    print("event teams:", sorted(events_df["team"].unique()))

    matches_df, unmatched_df = run_counterfactual_event_study(historical_df, events_df)

    if matches_df.empty:
        raise ValueError("反実仮想予測対象の試合が0件でした。events CSVとhistorical CSVを確認してください。")

    summary_df, by_year_df, by_event_df = make_summaries(matches_df)

    # 日付列をCSVで見やすくする
    for df in [matches_df, unmatched_df, summary_df, by_year_df, by_event_df]:
        if df is None or df.empty:
            continue
        for col in df.columns:
            if "date" in col:
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")
                except Exception:
                    pass

    matches_df.to_csv(OUTPUT_MATCHES_CSV, index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    by_event_df.to_csv(OUTPUT_BY_EVENT_CSV, index=False, encoding="utf-8-sig")
    by_year_df.to_csv(OUTPUT_BY_YEAR_CSV, index=False, encoding="utf-8-sig")

    if unmatched_df is None or unmatched_df.empty:
        pd.DataFrame(columns=["event_id", "season", "team", "reason"]).to_csv(
            OUTPUT_UNMATCHED_CSV, index=False, encoding="utf-8-sig"
        )
    else:
        unmatched_df.to_csv(OUTPUT_UNMATCHED_CSV, index=False, encoding="utf-8-sig")

    export_html(summary_df, by_year_df, by_event_df, unmatched_df, OUTPUT_HTML)

    print("\n=== 出力完了 ===")
    for path in [
        OUTPUT_MATCHES_CSV,
        OUTPUT_SUMMARY_CSV,
        OUTPUT_BY_EVENT_CSV,
        OUTPUT_BY_YEAR_CSV,
        OUTPUT_UNMATCHED_CSV,
        OUTPUT_HTML,
    ]:
        print(path.name)

    print("\n=== ウィンドウ別集計 ===")
    show_cols = [
        "window", "n_matches", "n_events",
        "mean_expected_points", "mean_actual_points", "mean_points_residual",
        "mean_goal_for_residual", "mean_goals_against_improvement",
    ]
    print(summary_df[show_cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
