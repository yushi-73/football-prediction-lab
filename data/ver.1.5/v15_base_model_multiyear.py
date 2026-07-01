import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path

# ============================================================
# J1 ver1.5 基準モデル 3年検証版
# ------------------------------------------------------------
# 目的:
#   ・終盤補正、監督解任ブーストなどの追加補正を入れない
#   ・2023, 2024, 2025 の3年分で基準モデルを検証する
#   ・順位予測結果に加えて、監督解任ブースト検証用の試合単位ログを出力する
#
# ver1.5 固定設定:
#   PREV_WEIGHT=0.20 / PREV_DECAY=0.995
#   ELO_LAMBDA_WEIGHT=0.20
#   GOAL_ADJUST=cap4
#   COMPAT_WEIGHT=0.20
#   DRAW_FACTOR=1.20 / MAX_MATCH_DRAW_PROB=0.33
#   CURRENT_DECAY=1.000 / SHRINKAGE=1.00
#   LAMBDA_CAP=3.5
#   昇格組補正なし
#   λには得点のみを使用し、シュート数・枠内シュート数は使わない
# ============================================================


# =========================
# 1. 設定
# =========================

BASE_DIR = Path(__file__).resolve().parent


def find_file(filename):
    candidates = [
        BASE_DIR / filename,
        BASE_DIR / "data" / filename,
        BASE_DIR.parent / "data" / filename,
        BASE_DIR.parent / filename,
        Path.cwd() / filename,
        Path.cwd() / "data" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"{filename} が見つかりません。スクリプトと同じフォルダ、または data フォルダに置いてください。"
    )


HISTORICAL_CSV = find_file("j1_historical_results_1993_2025_table_fixed.csv")

MODEL_VERSION = "v1.5"
MODEL_NAME = "v15_base"

TARGET_YEARS = [2023, 2024, 2025]

# 固定パラメータ
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
GOAL_ADJUST_NAME = "cap4"
GOAL_ADJUST_MODE = "cap"
GOAL_CAP_FOR_STRENGTH = 4

# 昇格組補正は不採用
PROMOTED_PRIOR_NAME = "none"
PROMOTED_PREV_WEIGHT = 0.00
PROMOTED_ATTACK_PRIOR = 1.00
PROMOTED_DEFENSE_PRIOR = 1.00
USE_PROMOTED_PREV_ZERO = True

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

# シミュレーション設定
N_SIM = 3000
RANDOM_SEED = 42
MAX_GOALS_FOR_SCORE_GRID = 10

# 出力
OUTPUT_DETAIL_CSV = BASE_DIR / "v15_base_multiyear_detail.csv"
OUTPUT_SUMMARY_CSV = BASE_DIR / "v15_base_multiyear_summary.csv"
OUTPUT_PREDICTIONS_CSV = BASE_DIR / "v15_base_multiyear_predictions.csv"
OUTPUT_MATCH_LOG_CSV = BASE_DIR / "v15_base_multiyear_match_log.csv"
OUTPUT_SUMMARY_HTML = BASE_DIR / "v15_base_multiyear_summary.html"


# =========================
# 2. チーム名処理
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
    for col in ["home", "away"]:
        if col in df.columns:
            df[col] = df[col].apply(standardize_team_name)
    return df


# =========================
# 3. データ読み込み
# =========================


def load_historical_j1_csv(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = clean_team_names(df)

    if "date" not in df.columns:
        raise ValueError("CSVに date 列が必要です。")

    df["date"] = (
        df["date"]
        .astype(str)
        .str.replace(r"\(.*\)", "", regex=True)
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "home_goal" in df.columns and "away_goal" in df.columns:
        df["home_goal"] = pd.to_numeric(df["home_goal"], errors="coerce")
        df["away_goal"] = pd.to_numeric(df["away_goal"], errors="coerce")
    elif "home_goals" in df.columns and "away_goals" in df.columns:
        df["home_goal"] = pd.to_numeric(df["home_goals"], errors="coerce")
        df["away_goal"] = pd.to_numeric(df["away_goals"], errors="coerce")
    else:
        raise ValueError("CSVに home_goal/away_goal または home_goals/away_goals が必要です。")

    df = df.dropna(subset=["date", "home", "away", "home_goal", "away_goal"]).copy()
    df["home_goal"] = df["home_goal"].astype(int)
    df["away_goal"] = df["away_goal"].astype(int)
    df["year"] = df["date"].dt.year.astype(int)
    df = df.sort_values("date").reset_index(drop=True)

    return df


# =========================
# 4. 順位表
# =========================


def get_points_for_match(home_goal, away_goal):
    if home_goal > away_goal:
        return 3, 0
    if home_goal < away_goal:
        return 0, 3
    return 1, 1


def calculate_table(match_df, teams):
    table = {
        team: {"points": 0, "gf": 0, "ga": 0, "gd": 0}
        for team in teams
    }

    for row in match_df.itertuples(index=False):
        row_dict = row._asdict()
        home = row_dict["home"]
        away = row_dict["away"]
        hg = int(row_dict["home_goal"])
        ag = int(row_dict["away_goal"])

        if home not in table:
            table[home] = {"points": 0, "gf": 0, "ga": 0, "gd": 0}
        if away not in table:
            table[away] = {"points": 0, "gf": 0, "ga": 0, "gd": 0}

        table[home]["gf"] += hg
        table[home]["ga"] += ag
        table[away]["gf"] += ag
        table[away]["ga"] += hg

        hp, ap = get_points_for_match(hg, ag)
        table[home]["points"] += hp
        table[away]["points"] += ap

    for team in table:
        table[team]["gd"] = table[team]["gf"] - table[team]["ga"]

    return table


def update_table_one_match(table, home, away, hg, ag):
    table[home]["gf"] += hg
    table[home]["ga"] += ag
    table[away]["gf"] += ag
    table[away]["ga"] += hg

    hp, ap = get_points_for_match(hg, ag)
    table[home]["points"] += hp
    table[away]["points"] += ap

    table[home]["gd"] = table[home]["gf"] - table[home]["ga"]
    table[away]["gd"] = table[away]["gf"] - table[away]["ga"]


def make_ranking(table):
    return sorted(
        table.items(),
        key=lambda x: (
            -x[1]["points"],
            -x[1]["gd"],
            -x[1]["gf"],
        )
    )


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
        home_gf = 0.0
        home_ga = 0.0
        home_w = 0.0

        for i, row in enumerate(home_games.itertuples(index=False)):
            weight = decay ** i
            row_dict = row._asdict()
            home_gf += float(row_dict["home_goal_strength"]) * weight
            home_ga += float(row_dict["away_goal_strength"]) * weight
            home_w += weight

        if home_w > 0:
            home_attack = (home_gf / home_w) / strength_home_avg_goals
            home_defense = (home_ga / home_w) / strength_away_avg_goals
        else:
            home_attack = 1.0
            home_defense = 1.0

        away_games = history_df[history_df["away"] == team].copy().sort_values("date", ascending=False)
        away_gf = 0.0
        away_ga = 0.0
        away_w = 0.0

        for i, row in enumerate(away_games.itertuples(index=False)):
            weight = decay ** i
            row_dict = row._asdict()
            away_gf += float(row_dict["away_goal_strength"]) * weight
            away_ga += float(row_dict["home_goal_strength"]) * weight
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
    if goal_diff <= 1:
        margin_multiplier = 1.0
    else:
        margin_multiplier = np.log(goal_diff) + 1.0

    change = k_factor * margin_multiplier * (actual_home - expected_home)

    ratings[home] = home_rating + change
    ratings[away] = away_rating - change

    return expected_home, actual_home


def build_elo_ratings(previous_df, train_df, teams, k_factor=16, home_adv=0):
    all_teams = pd.unique(
        pd.concat([
            previous_df[["home", "away"]],
            train_df[["home", "away"]],
        ]).values.ravel()
    )

    ratings = {team: INITIAL_ELO for team in all_teams}
    combined = pd.concat([previous_df, train_df], ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)

    for row in combined.itertuples(index=False):
        row_dict = row._asdict()
        update_elo_one_match(
            ratings=ratings,
            home=row_dict["home"],
            away=row_dict["away"],
            home_goal=int(row_dict["home_goal"]),
            away_goal=int(row_dict["away_goal"]),
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
        row_dict = row._asdict()
        home = row_dict["home"]
        away = row_dict["away"]
        hg = int(row_dict["home_goal"])
        ag = int(row_dict["away_goal"])

        home_rating = get_rating(home)
        away_rating = get_rating(away)

        expected_home = 1 / (1 + 10 ** ((away_rating - (home_rating + home_adv)) / 400))
        actual_home = get_actual_score_from_goals(hg, ag)

        residual_home = actual_home - expected_home
        residual_away = -residual_home

        years_ago = max(0, int(target_year) - int(row_dict["year"]))
        weight = time_decay ** years_ago

        add_effect((home, away), residual_home, weight)
        add_effect((away, home), residual_away, weight)

        update_elo_one_match(
            ratings=ratings,
            home=home,
            away=away,
            home_goal=hg,
            away_goal=ag,
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
    lambda_home = (
        strengths[home]["home_attack"]
        * strengths[away]["away_defense"]
        * home_avg_goals
    )
    lambda_away = (
        strengths[away]["away_attack"]
        * strengths[home]["home_defense"]
        * away_avg_goals
    )

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

        home_factor = np.clip(
            1 + compat_weight * home_effect,
            COMPAT_FACTOR_MIN,
            COMPAT_FACTOR_MAX,
        )
        away_factor = np.clip(
            1 + compat_weight * away_effect,
            COMPAT_FACTOR_MIN,
            COMPAT_FACTOR_MAX,
        )

        lambda_home *= home_factor
        lambda_away *= away_factor

    return safe_lambda(lambda_home), safe_lambda(lambda_away)


def poisson_pmf_array(lam, max_goals):
    probs = np.zeros(max_goals + 1, dtype=float)
    probs[0] = np.exp(-lam)
    for k in range(1, max_goals + 1):
        probs[k] = probs[k - 1] * lam / k
    return probs


def score_probability_matrix_with_draw_factor(
    lambda_home,
    lambda_away,
    draw_factor,
    max_goals=MAX_GOALS_FOR_SCORE_GRID,
    max_match_draw_prob=None,
):
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


def make_score_sampler(score_matrix):
    flat_probs = score_matrix.ravel()
    flat_probs = flat_probs / flat_probs.sum()
    cum_probs = np.cumsum(flat_probs)
    score_pairs = np.array(np.unravel_index(np.arange(flat_probs.size), score_matrix.shape)).T
    return score_pairs, cum_probs


def sample_score_from_precomputed(score_pairs, cum_probs):
    idx = int(np.searchsorted(cum_probs, np.random.random(), side="right"))
    if idx >= len(score_pairs):
        idx = len(score_pairs) - 1
    hg, ag = score_pairs[idx]
    return int(hg), int(ag)


# =========================
# 9. 1年分シミュレーション
# =========================


def simulate_target_year(historical_df, target_year, n_sim=N_SIM, seed=42):
    previous_year = target_year - 1

    previous_df = historical_df[historical_df["year"] == previous_year].copy()
    target_df = historical_df[historical_df["year"] == target_year].copy()

    if previous_df.empty:
        raise ValueError(f"{previous_year}年のデータがありません。")
    if target_df.empty:
        raise ValueError(f"{target_year}年のデータがありません。")

    previous_df = previous_df.sort_values("date").reset_index(drop=True)
    target_df = target_df.sort_values("date").reset_index(drop=True)

    teams = list(pd.unique(target_df[["home", "away"]].values.ravel()))

    # 元の終盤補正コードと同じく、試合数の50%で前半・後半を分割する
    split = int(len(target_df) * 0.5)
    train_df = target_df.iloc[:split].copy()
    test_df = target_df.iloc[split:].copy()

    actual_table = calculate_table(target_df, teams)
    actual_ranking = make_ranking(actual_table)
    actual_position = {team: pos + 1 for pos, (team, _) in enumerate(actual_ranking)}

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

    cutoff_date = train_df["date"].max()
    matchup_effects = None
    if COMPAT_WEIGHT > 0:
        matchup_effects = build_matchup_effects_j1(
            historical_df=historical_df,
            cutoff_date=cutoff_date,
            target_year=target_year,
            prior_n=MATCHUP_PRIOR_N,
            time_decay=MATCHUP_TIME_DECAY,
            k_factor=K_FACTOR,
            home_adv=HOME_ADV,
        )

    match_items = []
    match_log_rows = []

    for match_index, row in enumerate(test_df.itertuples(index=False), start=1):
        row_dict = row._asdict()
        home = row_dict["home"]
        away = row_dict["away"]
        actual_hg = int(row_dict["home_goal"])
        actual_ag = int(row_dict["away_goal"])

        lambda_home, lambda_away = expected_goals_home_away(
            home=home,
            away=away,
            strengths=strengths,
            home_avg_goals=home_avg_goals,
            away_avg_goals=away_avg_goals,
            elo_ratings=elo_ratings,
            elo_lambda_weight=ELO_LAMBDA_WEIGHT,
            matchup_effects=matchup_effects,
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
        score_pairs, cum_probs = make_score_sampler(score_matrix)

        actual_home_points, actual_away_points = get_points_for_match(actual_hg, actual_ag)

        match_log_rows.append({
            "model_version": MODEL_VERSION,
            "target_year": target_year,
            "previous_year": previous_year,
            "match_index_in_test": match_index,
            "date": row_dict["date"],
            "home": home,
            "away": away,
            "actual_home_goal": actual_hg,
            "actual_away_goal": actual_ag,
            "lambda_home": lambda_home,
            "lambda_away": lambda_away,
            "home_win_prob": probs["home_win_prob"],
            "draw_prob": probs["draw_prob"],
            "away_win_prob": probs["away_win_prob"],
            "home_expected_points": probs["home_expected_points"],
            "away_expected_points": probs["away_expected_points"],
            "home_actual_points": actual_home_points,
            "away_actual_points": actual_away_points,
            "home_points_residual": actual_home_points - probs["home_expected_points"],
            "away_points_residual": actual_away_points - probs["away_expected_points"],
            "home_goal_residual": actual_hg - lambda_home,
            "away_goal_residual": actual_ag - lambda_away,
            "score_grid_expected_home_goals": probs["score_grid_expected_home_goals"],
            "score_grid_expected_away_goals": probs["score_grid_expected_away_goals"],
            "home_elo": elo_ratings.get(home, INITIAL_ELO),
            "away_elo": elo_ratings.get(away, INITIAL_ELO),
            "home_matchup_effect": 0.0 if matchup_effects is None else matchup_effects.get((home, away), 0.0),
            "away_matchup_effect": 0.0 if matchup_effects is None else matchup_effects.get((away, home), 0.0),
        })

        match_items.append({
            "home": home,
            "away": away,
            "score_pairs": score_pairs,
            "cum_probs": cum_probs,
        })

    if seed is not None:
        np.random.seed(seed)

    position_counts = {team: Counter() for team in teams}
    points_sum = {team: 0.0 for team in teams}
    gf_sum = {team: 0.0 for team in teams}
    ga_sum = {team: 0.0 for team in teams}
    gd_sum = {team: 0.0 for team in teams}
    draw_count = 0
    simulated_match_count = 0

    for sim in range(1, n_sim + 1):
        table = calculate_table(train_df, teams)

        for item in match_items:
            home = item["home"]
            away = item["away"]
            hg, ag = sample_score_from_precomputed(item["score_pairs"], item["cum_probs"])

            simulated_match_count += 1
            if hg == ag:
                draw_count += 1

            update_table_one_match(table, home, away, hg, ag)

        ranking = make_ranking(table)

        for pos, (team, stats) in enumerate(ranking):
            position_counts[team][pos + 1] += 1

        for team in teams:
            points_sum[team] += table[team]["points"]
            gf_sum[team] += table[team]["gf"]
            ga_sum[team] += table[team]["ga"]
            gd_sum[team] += table[team]["gd"]

    rows = []
    n_teams = len(teams)

    for team in teams:
        avg_pred_pos = sum(
            pos * (position_counts[team][pos] / n_sim)
            for pos in range(1, n_teams + 1)
        )
        actual_pos = actual_position[team]

        rows.append({
            "model_version": MODEL_VERSION,
            "target_year": target_year,
            "previous_year": previous_year,
            "team": team,
            "actual_position": actual_pos,
            "avg_pred_position": avg_pred_pos,
            "position_error": abs(avg_pred_pos - actual_pos),
            "prob_actual_position": position_counts[team][actual_pos] / n_sim,
            "prev_weight": PREV_WEIGHT,
            "prev_decay": PREV_DECAY,
            "elo_lambda_weight": ELO_LAMBDA_WEIGHT,
            "compat_weight": COMPAT_WEIGHT,
            "current_decay": CURRENT_DECAY,
            "shrinkage": SHRINKAGE,
            "draw_factor": DRAW_FACTOR,
            "max_match_draw_prob": MAX_MATCH_DRAW_PROB,
            "lambda_cap": LAMBDA_CAP,
            "promoted_prior_name": PROMOTED_PRIOR_NAME,
            "promoted_prev_weight": PROMOTED_PREV_WEIGHT,
            "promoted_attack_prior": PROMOTED_ATTACK_PRIOR,
            "promoted_defense_prior": PROMOTED_DEFENSE_PRIOR,
            "matchup_prior_n": MATCHUP_PRIOR_N,
            "matchup_time_decay": MATCHUP_TIME_DECAY,
            "compat_factor_min": COMPAT_FACTOR_MIN,
            "compat_factor_max": COMPAT_FACTOR_MAX,
            "goal_adjust_name": GOAL_ADJUST_NAME,
            "goal_adjust_mode": GOAL_ADJUST_MODE,
            "goal_cap_for_strength": GOAL_CAP_FOR_STRENGTH,
            "elo": elo_ratings.get(team, INITIAL_ELO),
            "prev_games": prev_games_by_team.get(team, 0),
            "is_promoted": bool(prev_games_by_team.get(team, 0) == 0),
            "effective_prev_weight": prev_weight_by_team.get(team, PREV_WEIGHT),
            "most_likely_position": position_counts[team].most_common(1)[0][0],
            "champion_prob": position_counts[team][1] / n_sim,
            "top3_prob": sum(position_counts[team][p] for p in range(1, min(3, n_teams) + 1)) / n_sim,
            "bottom3_prob": sum(position_counts[team][p] for p in range(max(1, n_teams - 2), n_teams + 1)) / n_sim,
            "avg_points": points_sum[team] / n_sim,
            "avg_gf": gf_sum[team] / n_sim,
            "avg_ga": ga_sum[team] / n_sim,
            "avg_gd": gd_sum[team] / n_sim,
        })

    prediction_df = pd.DataFrame(rows)
    prediction_df = prediction_df.sort_values("avg_pred_position").reset_index(drop=True)
    prediction_df.insert(0, "pred_rank", prediction_df.index + 1)

    mae = float(prediction_df["position_error"].mean())
    mean_prob_actual_position = float(prediction_df["prob_actual_position"].mean())
    sim_draw_rate = float(draw_count / simulated_match_count) if simulated_match_count > 0 else np.nan

    match_log_df = pd.DataFrame(match_log_rows)

    summary = {
        "model_version": MODEL_VERSION,
        "target_year": target_year,
        "previous_year": previous_year,
        "prev_weight": PREV_WEIGHT,
        "prev_decay": PREV_DECAY,
        "elo_lambda_weight": ELO_LAMBDA_WEIGHT,
        "compat_weight": COMPAT_WEIGHT,
        "current_decay": CURRENT_DECAY,
        "shrinkage": SHRINKAGE,
        "draw_factor": DRAW_FACTOR,
        "max_match_draw_prob": MAX_MATCH_DRAW_PROB,
        "lambda_cap": LAMBDA_CAP,
        "promoted_prior_name": PROMOTED_PRIOR_NAME,
        "promoted_prev_weight": PROMOTED_PREV_WEIGHT,
        "promoted_attack_prior": PROMOTED_ATTACK_PRIOR,
        "promoted_defense_prior": PROMOTED_DEFENSE_PRIOR,
        "matchup_prior_n": MATCHUP_PRIOR_N,
        "matchup_time_decay": MATCHUP_TIME_DECAY,
        "compat_factor_min": COMPAT_FACTOR_MIN,
        "compat_factor_max": COMPAT_FACTOR_MAX,
        "goal_adjust_name": GOAL_ADJUST_NAME,
        "goal_adjust_mode": GOAL_ADJUST_MODE,
        "goal_cap_for_strength": GOAL_CAP_FOR_STRENGTH,
        "n_sim": n_sim,
        "n_teams": n_teams,
        "n_previous_matches": len(previous_df),
        "n_target_matches": len(target_df),
        "n_train_matches": len(train_df),
        "n_test_matches": len(test_df),
        "mae": mae,
        "mean_prob_actual_position": mean_prob_actual_position,
        "sim_draw_rate": sim_draw_rate,
        "match_log_mean_home_points_residual": float(match_log_df["home_points_residual"].mean()),
        "match_log_mean_away_points_residual": float(match_log_df["away_points_residual"].mean()),
        "match_log_mean_home_goal_residual": float(match_log_df["home_goal_residual"].mean()),
        "match_log_mean_away_goal_residual": float(match_log_df["away_goal_residual"].mean()),
    }

    promoted_mask = prediction_df["is_promoted"].astype(bool)
    summary["promoted_team_count"] = int(promoted_mask.sum())
    summary["promoted_mean_error"] = (
        float(prediction_df.loc[promoted_mask, "position_error"].mean())
        if promoted_mask.any()
        else np.nan
    )

    title_mask = prediction_df["actual_position"] <= min(3, n_teams)
    bottom3_mask = prediction_df["actual_position"] >= max(1, n_teams - 2)
    summary["title_contender_mean_error"] = (
        float(prediction_df.loc[title_mask, "position_error"].mean())
        if title_mask.any()
        else np.nan
    )
    summary["bottom3_mean_error"] = (
        float(prediction_df.loc[bottom3_mask, "position_error"].mean())
        if bottom3_mask.any()
        else np.nan
    )

    return summary, prediction_df, match_log_df


# =========================
# 10. HTML出力
# =========================


def export_summary_html(summary_df, output_path):
    display_df = summary_df.copy()

    round_cols = [
        "mean_mae", "std_mae", "mean_prob_actual_position", "mean_sim_draw_rate",
        "mae_2023", "mae_2024", "mae_2025",
        "mean_title_contender_error", "mean_bottom3_error",
        "prev_weight", "prev_decay", "elo_lambda_weight", "compat_weight",
        "draw_factor", "max_match_draw_prob", "lambda_cap",
    ]
    for col in round_cols:
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(display_df[col], errors="coerce").round(4)

    table_html = display_df.to_html(index=False, classes="prediction-table")

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>ver1.5 基準モデル 3年検証</title>
  <style>
    body {{
      font-family: Arial, "Hiragino Kaku Gothic ProN", "Yu Gothic", sans-serif;
      margin: 40px;
      background: #f7f7f7;
      color: #222;
    }}
    h1 {{ margin-bottom: 8px; }}
    .note {{ line-height: 1.8; color: #555; margin-bottom: 24px; }}
    .table-wrap {{
      overflow-x: auto;
      background: white;
      padding: 16px;
      border-radius: 12px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{
      border: 1px solid #ddd;
      padding: 8px 10px;
      text-align: center;
      white-space: nowrap;
    }}
    th {{ background: #222; color: white; }}
    tr:nth-child(even) {{ background: #f5f5f5; }}
  </style>
</head>
<body>
  <h1>ver1.5 基準モデル 3年検証</h1>
  <div class="note">
    <p>
      終盤補正・監督解任ブーストを入れない基準モデルです。
      2022→2023、2023→2024、2024→2025を対象に、3年分の順位予測精度を検証しています。
    </p>
    <p>
      PREV_WEIGHT={PREV_WEIGHT}, PREV_DECAY={PREV_DECAY}, ELO_LAMBDA_WEIGHT={ELO_LAMBDA_WEIGHT},
      COMPAT_WEIGHT={COMPAT_WEIGHT}, DRAW_FACTOR={DRAW_FACTOR}, MAX_MATCH_DRAW_PROB={MAX_MATCH_DRAW_PROB},
      LAMBDA_CAP={LAMBDA_CAP}, N_SIM={N_SIM}。
    </p>
  </div>
  <div class="table-wrap">
    {table_html}
  </div>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


# =========================
# 11. main
# =========================


def main():
    historical_df = load_historical_j1_csv(HISTORICAL_CSV)

    print("\n==============================")
    print("ver1.5 基準モデル 3年検証")
    print("==============================")
    print("HISTORICAL_CSV:", HISTORICAL_CSV)
    print("TARGET_YEARS:", TARGET_YEARS)
    print("PREV_WEIGHT:", PREV_WEIGHT)
    print("PREV_DECAY:", PREV_DECAY)
    print("ELO_LAMBDA_WEIGHT:", ELO_LAMBDA_WEIGHT)
    print("COMPAT_WEIGHT:", COMPAT_WEIGHT)
    print("DRAW_FACTOR:", DRAW_FACTOR)
    print("MAX_MATCH_DRAW_PROB:", MAX_MATCH_DRAW_PROB)
    print("LAMBDA_CAP:", LAMBDA_CAP)
    print("N_SIM:", N_SIM)

    detail_rows = []
    prediction_dfs = []
    match_log_dfs = []

    for i, target_year in enumerate(TARGET_YEARS, start=1):
        print(f"\n[{i}/{len(TARGET_YEARS)}] target_year={target_year}")

        summary, prediction_df, match_log_df = simulate_target_year(
            historical_df=historical_df,
            target_year=target_year,
            n_sim=N_SIM,
            seed=RANDOM_SEED + target_year,
        )

        detail_rows.append(summary)
        prediction_dfs.append(prediction_df)
        match_log_dfs.append(match_log_df)

        print(
            f"  MAE={summary['mae']:.4f}, "
            f"実順位確率平均={summary['mean_prob_actual_position']:.4f}, "
            f"引分率={summary['sim_draw_rate']:.4f}"
        )

    detail_df = pd.DataFrame(detail_rows)
    predictions_df = pd.concat(prediction_dfs, ignore_index=True)
    match_log_df = pd.concat(match_log_dfs, ignore_index=True)

    detail_df.to_csv(OUTPUT_DETAIL_CSV, index=False, encoding="utf-8-sig")
    predictions_df.to_csv(OUTPUT_PREDICTIONS_CSV, index=False, encoding="utf-8-sig")
    match_log_df.to_csv(OUTPUT_MATCH_LOG_CSV, index=False, encoding="utf-8-sig")

    summary = {
        "model_version": MODEL_VERSION,
        "target_years": ",".join(str(y) for y in TARGET_YEARS),
        "prev_weight": PREV_WEIGHT,
        "prev_decay": PREV_DECAY,
        "elo_lambda_weight": ELO_LAMBDA_WEIGHT,
        "compat_weight": COMPAT_WEIGHT,
        "current_decay": CURRENT_DECAY,
        "shrinkage": SHRINKAGE,
        "draw_factor": DRAW_FACTOR,
        "max_match_draw_prob": MAX_MATCH_DRAW_PROB,
        "lambda_cap": LAMBDA_CAP,
        "goal_adjust_name": GOAL_ADJUST_NAME,
        "goal_adjust_mode": GOAL_ADJUST_MODE,
        "goal_cap_for_strength": GOAL_CAP_FOR_STRENGTH,
        "promoted_prior_name": PROMOTED_PRIOR_NAME,
        "n_sim": N_SIM,
        "mean_mae": float(detail_df["mae"].mean()),
        "std_mae": float(detail_df["mae"].std()),
        "mean_prob_actual_position": float(detail_df["mean_prob_actual_position"].mean()),
        "mean_sim_draw_rate": float(detail_df["sim_draw_rate"].mean()),
        "mean_title_contender_error": float(detail_df["title_contender_mean_error"].mean()),
        "mean_bottom3_error": float(detail_df["bottom3_mean_error"].mean()),
    }

    for year in TARGET_YEARS:
        year_row = detail_df[detail_df["target_year"] == year].iloc[0]
        summary[f"mae_{year}"] = float(year_row["mae"])
        summary[f"prob_actual_position_{year}"] = float(year_row["mean_prob_actual_position"])
        summary[f"sim_draw_rate_{year}"] = float(year_row["sim_draw_rate"])

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    export_summary_html(summary_df, OUTPUT_SUMMARY_HTML)

    print("\n==============================")
    print("3年集計")
    print("==============================")
    print(summary_df[[
        "mean_mae", "std_mae", "mean_prob_actual_position", "mean_sim_draw_rate",
        "mae_2023", "mae_2024", "mae_2025",
    ]].to_string(index=False))

    print("\n出力ファイル:")
    print("DETAIL CSV:", OUTPUT_DETAIL_CSV)
    print("SUMMARY CSV:", OUTPUT_SUMMARY_CSV)
    print("PREDICTIONS CSV:", OUTPUT_PREDICTIONS_CSV)
    print("MATCH LOG CSV:", OUTPUT_MATCH_LOG_CSV)
    print("SUMMARY HTML:", OUTPUT_SUMMARY_HTML)


if __name__ == "__main__":
    main()
