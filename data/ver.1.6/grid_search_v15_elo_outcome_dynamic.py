import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path

# ============================================================
# J1 2025 順位予測 ver.1.4 大勝補正版
# ------------------------------------------------------------
# 採用モデル:
#   ・2025前半戦: 得点 + 枠内シュート
#   ・2024前年: 得点 + 総シュート
#   ・前年レーティング重み: 0.4
#   ・昇格組補正: 前年J1試合数0のチームは前年重み0.0
#   ・Elo補正: K=16, HOME_ADV=0, ELO_LAMBDA_WEIGHT=0.20
#   ・相性Effect: ON
#
# ver.1.4 追加点:
#   ・攻守係数を作るときだけ、大勝試合の得点をcapする
#   ・実際の順位表・初期勝点・実順位計算には生の得点を使う
#   ・シミュレーションの得点生成も通常のポアソンで行う
# ============================================================


# =========================
# 1. 設定
# =========================

BASE_DIR = Path(__file__).resolve().parent

# スクリプトを data フォルダに置いても、プロジェクト直下に置いても動くようにする
def find_file(filename):
    candidates = [
        BASE_DIR / filename,
        BASE_DIR / "data" / filename,
        BASE_DIR.parent / "data" / filename,
        BASE_DIR.parent / filename,
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"{filename} が見つかりません。スクリプトと同じフォルダ、または data フォルダに置いてください。"
    )


CURRENT_CSV = find_file("j1_2025_match_stats_merged_fixed.csv")
PREVIOUS_CSV = find_file("j1_2024_match_stats_merged_fixed.csv")

# 相性Effectを使う場合だけ必要
HISTORICAL_J1_FILENAME = find_file("j1_historical_results_1993_2025_table_fixed.csv")

# ver.1.4 大勝補正
# raw  : 得点をそのまま攻守係数に使う
# cap4 : 攻守係数計算用の得点を最大4点にする
# cap3 : 攻守係数計算用の得点を最大3点にする
GOAL_ADJUST_MODE = "cap4"
GOAL_CAP_FOR_STRENGTH = 4

# True : リーグ平均得点は生の得点を使い、チーム攻守係数だけ補正得点で作る
# False: リーグ平均得点も補正得点から作る
USE_RAW_LEAGUE_AVG_FOR_LAMBDA = True

OUTPUT_TAG = f"v14_{GOAL_ADJUST_MODE}"
OUTPUT_CSV = BASE_DIR / f"j1_2025_prediction_{OUTPUT_TAG}.csv"
OUTPUT_HTML = BASE_DIR / f"j1_2025_prediction_{OUTPUT_TAG}.html"

# 基本モデル
SOT_WEIGHT = 0.1
PREV_SHOT_WEIGHT = 0.1
PREV_WEIGHT = 0.4

# 昇格組補正
# 前年J1にいないチームは、前年由来の中立値1.0を混ぜない
USE_PROMOTED_PREV_ZERO = True
PROMOTED_PREV_WEIGHT = 0.0

# Elo補正
INITIAL_ELO = 1500
K_FACTOR = 16
HOME_ADV = 0
ELO_LAMBDA_WEIGHT = 0.20

# 相性Effect
USE_MATCHUP_EFFECT = True
COMPAT_WEIGHT = 0.20
MATCHUP_PRIOR_N = 30
MATCHUP_TIME_DECAY = 0.97
COMPAT_FACTOR_MIN = 0.95
COMPAT_FACTOR_MAX = 1.05

# シミュレーション
N_SIM = 10000
DECAY = 1.0
LAMBDA_CAP = 3.5
RANDOM_SEED = 42

# Elo lambda補正の強さ
ELO_FACTOR_SCALE = 4000
ELO_FACTOR_MIN = 0.90
ELO_FACTOR_MAX = 1.10


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

def load_match_stats_csv(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = clean_team_names(df)

    df["date"] = (
        df["date"]
        .astype(str)
        .str.replace(r"\(.*\)", "", regex=True)
    )
    df["date"] = pd.to_datetime(df["date"])

    df["home_goal"] = pd.to_numeric(df["home_goals"], errors="coerce")
    df["away_goal"] = pd.to_numeric(df["away_goals"], errors="coerce")

    if "home_shots_official" in df.columns:
        df["home_shots"] = pd.to_numeric(df["home_shots_official"], errors="coerce")
        df["away_shots"] = pd.to_numeric(df["away_shots_official"], errors="coerce")
    elif "home_shots" in df.columns:
        df["home_shots"] = pd.to_numeric(df["home_shots"], errors="coerce")
        df["away_shots"] = pd.to_numeric(df["away_shots"], errors="coerce")
    else:
        df["home_shots"] = np.nan
        df["away_shots"] = np.nan

    if "home_shots_on_target" in df.columns:
        df["home_shots_on_target"] = pd.to_numeric(df["home_shots_on_target"], errors="coerce")
        df["away_shots_on_target"] = pd.to_numeric(df["away_shots_on_target"], errors="coerce")
    else:
        df["home_shots_on_target"] = np.nan
        df["away_shots_on_target"] = np.nan

    df = df.dropna(subset=["home_goal", "away_goal"]).copy()
    df = df.sort_values("date").reset_index(drop=True)

    return df


def load_historical_j1_csv(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = clean_team_names(df)

    df["date"] = pd.to_datetime(df["date"])
    df["home_goal"] = pd.to_numeric(df["home_goal"], errors="coerce")
    df["away_goal"] = pd.to_numeric(df["away_goal"], errors="coerce")

    df = df.dropna(subset=["home_goal", "away_goal"]).copy()
    df = df.sort_values("date").reset_index(drop=True)

    return df


# =========================
# 4. 順位表
# =========================

def calculate_table(match_df, teams):
    table = {
        team: {"points": 0, "gf": 0, "ga": 0, "gd": 0}
        for team in teams
    }

    for _, row in match_df.iterrows():
        home = row["home"]
        away = row["away"]
        hg = int(row["home_goal"])
        ag = int(row["away_goal"])

        if home not in table:
            table[home] = {"points": 0, "gf": 0, "ga": 0, "gd": 0}
        if away not in table:
            table[away] = {"points": 0, "gf": 0, "ga": 0, "gd": 0}

        table[home]["gf"] += hg
        table[home]["ga"] += ag
        table[away]["gf"] += ag
        table[away]["ga"] += hg

        if hg > ag:
            table[home]["points"] += 3
        elif hg < ag:
            table[away]["points"] += 3
        else:
            table[home]["points"] += 1
            table[away]["points"] += 1

    for team in table:
        table[team]["gd"] = table[team]["gf"] - table[team]["ga"]

    return table


def make_ranking(table):
    return sorted(
        table.items(),
        key=lambda x: (
            -x[1]["points"],
            -x[1]["gd"],
            -x[1]["gf"]
        )
    )


# =========================
# 5. 大勝補正
# =========================

def add_goal_for_strength_columns(df):
    """
    攻守係数計算用の得点列を追加する。

    注意:
        ・順位表計算、実順位、初期勝点には生の home_goal / away_goal を使う。
        ・この列は、チームの攻撃力・守備力を推定するときだけ使う。
    """
    df = df.copy()

    if GOAL_ADJUST_MODE == "raw":
        df["home_goal_strength"] = df["home_goal"].astype(float)
        df["away_goal_strength"] = df["away_goal"].astype(float)

    elif GOAL_ADJUST_MODE in ["cap3", "cap4"]:
        cap = float(GOAL_CAP_FOR_STRENGTH)
        df["home_goal_strength"] = df["home_goal"].astype(float).clip(upper=cap)
        df["away_goal_strength"] = df["away_goal"].astype(float).clip(upper=cap)

    else:
        raise ValueError("GOAL_ADJUST_MODE は 'raw', 'cap3', 'cap4' のどれかにしてください。")

    return df


def safe_positive_mean(series, fallback=1.0):
    value = pd.to_numeric(series, errors="coerce").mean()
    if not np.isfinite(value) or value <= 0:
        return fallback
    return float(value)


# =========================
# 6. 攻守係数
# =========================

def calculate_strengths_home_away(
    history_df,
    teams,
    decay=1.0,
    feature_weight=0.1,
    feature_type="sot"
):
    history_df = history_df.copy()
    history_df = add_goal_for_strength_columns(history_df)

    if feature_type == "sot":
        home_feature_col = "home_shots_on_target"
        away_feature_col = "away_shots_on_target"
    elif feature_type == "shots":
        home_feature_col = "home_shots"
        away_feature_col = "away_shots"
    elif feature_type == "none":
        home_feature_col = None
        away_feature_col = None
        feature_weight = 0.0
    else:
        raise ValueError("feature_type は 'sot', 'shots', 'none' のどれかにしてください。")

    use_feature = feature_weight > 0 and home_feature_col is not None
    goal_weight = 1.0 - feature_weight

    # 生のリーグ平均得点: 最終的なlambdaの基準値として使う
    raw_home_avg_goals = safe_positive_mean(history_df["home_goal"], fallback=1.0)
    raw_away_avg_goals = safe_positive_mean(history_df["away_goal"], fallback=1.0)

    # 補正後のリーグ平均得点: 攻守係数の比率計算に使う
    strength_home_avg_goals = safe_positive_mean(history_df["home_goal_strength"], fallback=1.0)
    strength_away_avg_goals = safe_positive_mean(history_df["away_goal_strength"], fallback=1.0)

    if USE_RAW_LEAGUE_AVG_FOR_LAMBDA:
        home_avg_goals = raw_home_avg_goals
        away_avg_goals = raw_away_avg_goals
    else:
        home_avg_goals = strength_home_avg_goals
        away_avg_goals = strength_away_avg_goals

    if use_feature:
        history_df[home_feature_col] = pd.to_numeric(history_df[home_feature_col], errors="coerce")
        history_df[away_feature_col] = pd.to_numeric(history_df[away_feature_col], errors="coerce")

        home_avg_feature = history_df[home_feature_col].mean()
        away_avg_feature = history_df[away_feature_col].mean()

        if not np.isfinite(home_avg_feature) or home_avg_feature <= 0:
            home_avg_feature = 1.0
        if not np.isfinite(away_avg_feature) or away_avg_feature <= 0:
            away_avg_feature = 1.0

        history_df[home_feature_col] = history_df[home_feature_col].fillna(home_avg_feature)
        history_df[away_feature_col] = history_df[away_feature_col].fillna(away_avg_feature)
    else:
        home_avg_feature = 1.0
        away_avg_feature = 1.0

    def safe_strength(value):
        if not np.isfinite(value) or value <= 0:
            return 1.0
        return float(value)

    strengths = {}

    for team in teams:
        home_games = history_df[history_df["home"] == team].copy()
        home_games = home_games.sort_values("date", ascending=False)

        home_gf = 0.0
        home_ga = 0.0
        home_feature_for = 0.0
        home_feature_against = 0.0
        home_w = 0.0

        for i, row in enumerate(home_games.itertuples(index=False)):
            row_dict = row._asdict()
            weight = decay ** i

            home_gf += row_dict["home_goal_strength"] * weight
            home_ga += row_dict["away_goal_strength"] * weight

            if use_feature:
                home_feature_for += row_dict[home_feature_col] * weight
                home_feature_against += row_dict[away_feature_col] * weight

            home_w += weight

        if home_w == 0:
            home_goal_attack = 1.0
            home_goal_defense = 1.0
            home_feature_attack = 1.0
            home_feature_defense = 1.0
        else:
            home_goal_attack = (home_gf / home_w) / strength_home_avg_goals
            home_goal_defense = (home_ga / home_w) / strength_away_avg_goals

            if use_feature:
                home_feature_attack = (home_feature_for / home_w) / home_avg_feature
                home_feature_defense = (home_feature_against / home_w) / away_avg_feature
            else:
                home_feature_attack = 1.0
                home_feature_defense = 1.0

        away_games = history_df[history_df["away"] == team].copy()
        away_games = away_games.sort_values("date", ascending=False)

        away_gf = 0.0
        away_ga = 0.0
        away_feature_for = 0.0
        away_feature_against = 0.0
        away_w = 0.0

        for i, row in enumerate(away_games.itertuples(index=False)):
            row_dict = row._asdict()
            weight = decay ** i

            away_gf += row_dict["away_goal_strength"] * weight
            away_ga += row_dict["home_goal_strength"] * weight

            if use_feature:
                away_feature_for += row_dict[away_feature_col] * weight
                away_feature_against += row_dict[home_feature_col] * weight

            away_w += weight

        if away_w == 0:
            away_goal_attack = 1.0
            away_goal_defense = 1.0
            away_feature_attack = 1.0
            away_feature_defense = 1.0
        else:
            away_goal_attack = (away_gf / away_w) / strength_away_avg_goals
            away_goal_defense = (away_ga / away_w) / strength_home_avg_goals

            if use_feature:
                away_feature_attack = (away_feature_for / away_w) / away_avg_feature
                away_feature_defense = (away_feature_against / away_w) / home_avg_feature
            else:
                away_feature_attack = 1.0
                away_feature_defense = 1.0

        strengths[team] = {
            "home_attack": safe_strength(goal_weight * home_goal_attack + feature_weight * home_feature_attack),
            "home_defense": safe_strength(goal_weight * home_goal_defense + feature_weight * home_feature_defense),
            "away_attack": safe_strength(goal_weight * away_goal_attack + feature_weight * away_feature_attack),
            "away_defense": safe_strength(goal_weight * away_goal_defense + feature_weight * away_feature_defense),
        }

    return strengths, home_avg_goals, away_avg_goals


def count_team_games(df, team):
    """指定チームがdf内で何試合出場しているかを数える"""
    return int(((df["home"] == team) | (df["away"] == team)).sum())


def blend_with_previous_strengths(
    current_strengths,
    previous_strengths,
    prev_weight,
    previous_df=None,
    use_promoted_prev_zero=True,
    promoted_prev_weight=0.0
):
    """
    2025前半の攻守係数と2024前年の攻守係数を混ぜる。

    通常:
        blended = 2025前半 * (1 - prev_weight) + 2024前年 * prev_weight

    昇格組補正ONの場合:
        2024年J1に試合がないチームは、前年重みを promoted_prev_weight にする。
        例: promoted_prev_weight = 0.0 なら、2025前半の係数だけを使う。
    """
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
        effective_prev_weight = prev_weight

        if previous_df is not None:
            prev_games = count_team_games(previous_df, team)
        else:
            prev_games = None

        # 前年J1にいないチームは、前年由来の中立値1.0を混ぜない
        if (
            use_promoted_prev_zero
            and previous_df is not None
            and prev_games == 0
        ):
            effective_prev_weight = promoted_prev_weight

        prev_weight_by_team[team] = effective_prev_weight
        prev_games_by_team[team] = prev_games

        if team in previous_strengths:
            blended[team] = {
                "home_attack": safe_blend(
                    current_strengths[team]["home_attack"],
                    previous_strengths[team]["home_attack"],
                    effective_prev_weight
                ),
                "home_defense": safe_blend(
                    current_strengths[team]["home_defense"],
                    previous_strengths[team]["home_defense"],
                    effective_prev_weight
                ),
                "away_attack": safe_blend(
                    current_strengths[team]["away_attack"],
                    previous_strengths[team]["away_attack"],
                    effective_prev_weight
                ),
                "away_defense": safe_blend(
                    current_strengths[team]["away_defense"],
                    previous_strengths[team]["away_defense"],
                    effective_prev_weight
                ),
            }
        else:
            blended[team] = current_strengths[team]

    return blended, prev_weight_by_team, prev_games_by_team


# =========================
# 7. Elo
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
            train_df[["home", "away"]]
        ]).values.ravel()
    )

    ratings = {team: INITIAL_ELO for team in all_teams}

    combined = pd.concat([previous_df, train_df], ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)

    for _, row in combined.iterrows():
        update_elo_one_match(
            ratings=ratings,
            home=row["home"],
            away=row["away"],
            home_goal=int(row["home_goal"]),
            away_goal=int(row["away_goal"]),
            k_factor=k_factor,
            home_adv=home_adv
        )

    return {team: ratings.get(team, INITIAL_ELO) for team in teams}


# =========================
# 8. 相性Effect
# =========================

def build_matchup_effects_j1(
    historical_j1_df,
    cutoff_date,
    target_year,
    prior_n=20,
    time_decay=0.94,
    k_factor=16,
    home_adv=0
):
    df = historical_j1_df.copy()
    df = df[df["date"] <= cutoff_date].copy()
    df = df.sort_values("date").reset_index(drop=True)

    ratings = {}

    weighted_residual_sum = {}
    weighted_match_sum = {}
    raw_match_count = {}

    def get_rating(team):
        if team not in ratings:
            ratings[team] = INITIAL_ELO
        return ratings[team]

    def add_effect(key, residual, weight):
        weighted_residual_sum[key] = weighted_residual_sum.get(key, 0.0) + residual * weight
        weighted_match_sum[key] = weighted_match_sum.get(key, 0.0) + weight
        raw_match_count[key] = raw_match_count.get(key, 0) + 1

    for _, row in df.iterrows():
        home = row["home"]
        away = row["away"]
        hg = int(row["home_goal"])
        ag = int(row["away_goal"])

        home_rating = get_rating(home)
        away_rating = get_rating(away)

        expected_home = 1 / (1 + 10 ** ((away_rating - (home_rating + home_adv)) / 400))
        actual_home = get_actual_score_from_goals(hg, ag)

        residual_home = actual_home - expected_home
        residual_away = -residual_home

        years_ago = max(0, target_year - int(row["date"].year))
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
            home_adv=home_adv
        )

    effects = {}

    for key in weighted_residual_sum:
        weighted_n = weighted_match_sum[key]
        raw_effect = weighted_residual_sum[key] / weighted_n if weighted_n > 0 else 0.0

        shrink = weighted_n / (weighted_n + prior_n)
        effects[key] = raw_effect * shrink

    return effects


# =========================
# 9. 期待得点
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
    compat_weight=0.0
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
            COMPAT_FACTOR_MAX
        )
        away_factor = np.clip(
            1 + compat_weight * away_effect,
            COMPAT_FACTOR_MIN,
            COMPAT_FACTOR_MAX
        )

        lambda_home *= home_factor
        lambda_away *= away_factor

    return safe_lambda(lambda_home), safe_lambda(lambda_away)


# =========================
# 10. HTML出力
# =========================

def export_prediction_html(df, output_path):
    display_df = df.copy()

    percent_cols = ["champion_prob", "top3_prob", "top5_prob", "bottom3_prob"]
    for col in percent_cols:
        if col in display_df.columns:
            display_df[col] = (display_df[col] * 100).round(1).astype(str) + "%"

    round_cols = [
        "avg_pred_position",
        "avg_points",
        "avg_gf",
        "avg_ga",
        "avg_gd",
        "position_error",
        "elo"
    ]
    for col in round_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(2)

    column_names = {
        "pred_rank": "予測順位",
        "team": "チーム",
        "actual_position": "実順位",
        "avg_pred_position": "平均予測順位",
        "position_error": "順位誤差",
        "most_likely_position": "最頻順位",
        "champion_prob": "優勝確率",
        "top3_prob": "上位3確率",
        "top5_prob": "上位5確率",
        "bottom3_prob": "下位3確率",
        "avg_points": "平均勝点",
        "avg_gf": "平均得点",
        "avg_ga": "平均失点",
        "avg_gd": "平均得失点差",
        "elo": "Elo",
    }

    display_df = display_df.rename(columns=column_names)

    table_html = display_df.to_html(index=False, classes="prediction-table")

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>J1 2025 順位予測 | ver.1.4 大勝補正</title>
  <style>
    body {{
      font-family: Arial, "Hiragino Kaku Gothic ProN", "Yu Gothic", sans-serif;
      margin: 40px;
      background: #f7f7f7;
      color: #222;
    }}
    h1 {{
      margin-bottom: 8px;
    }}
    .note {{
      line-height: 1.8;
      color: #555;
      margin-bottom: 24px;
    }}
    .table-wrap {{
      overflow-x: auto;
      background: white;
      padding: 16px;
      border-radius: 12px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      border: 1px solid #ddd;
      padding: 8px 10px;
      text-align: center;
      white-space: nowrap;
    }}
    th {{
      background: #222;
      color: white;
    }}
    tr:nth-child(even) {{
      background: #f5f5f5;
    }}
  </style>
</head>
<body>
  <h1>J1 2025 順位予測 ver.1.4 大勝補正</h1>
  <div class="note">
    <p>
      採用設定: 前年レーティング重み {PREV_WEIGHT} /
      2025枠内シュート重み {SOT_WEIGHT} /
      2024総シュート重み {PREV_SHOT_WEIGHT} /
      昇格組前年重み {PROMOTED_PREV_WEIGHT if USE_PROMOTED_PREV_ZERO else PREV_WEIGHT} /
      大勝補正 {GOAL_ADJUST_MODE}, cap={GOAL_CAP_FOR_STRENGTH} /
      Elo補正 K={K_FACTOR}, HOME_ADV={HOME_ADV}, ELO_LAMBDA_WEIGHT={ELO_LAMBDA_WEIGHT}
    </p>
    <p>
      N_SIM={N_SIM}。相性Effect: {"ON" if USE_MATCHUP_EFFECT else "OFF"}。
      攻守係数計算時のみ大勝補正を適用し、実順位・初期勝点には実際の得点を使用しています。
      予測順位はシミュレーション上の平均順位で並べています。
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
# 11. v1.5候補: Eloをλではなく勝敗確率に使う検証
# =========================

# 探索中は2000回、最良候補だけ10000回で再実行します。
# 重い場合は N_SIM_SEARCH を 1000 に下げてください。
N_SIM_SEARCH = 2000
N_SIM_FINAL = 10000
RUN_FINAL_BEST = True

# 今回は以下2候補を残しつつ、Eloを勝敗確率側に混ぜる重みを検証します。
PREV_SETTING_CANDIDATES = [
    {"name": "prev060_decay0985", "prev_weight": 0.60, "prev_decay": 0.985},
    {"name": "prev040_decay0995", "prev_weight": 0.40, "prev_decay": 0.995},
]

# Eloをλに掛ける補正は使わず、勝敗確率側にだけ使う。
# 0.00 は「Elo勝敗補正なし」の比較用です。
ELO_OUTCOME_WEIGHT_CANDIDATES = [
    0.00, 0.05, 0.10, 0.15, 0.20, 0.25,
]

# シミュレーション内で仮想試合結果をEloへ反映する間隔。
# 0 は固定Elo、1 は毎試合更新、3 は3試合おき更新。
# 案Aの本命は 3 ですが、比較のため固定・毎試合・5試合おきも残しています。
ELO_UPDATE_EVERY_CANDIDATES = [0, 1, 3, 5]

# Elo差を勝敗優勢度へ変換するときのスケール。
# 通常のElo期待値に近い形で、400を基本にする。
ELO_OUTCOME_SCALE = 400

# 勝敗確率補正用に条件付きスコアを抽選するときの最大得点。
# λは3.5でcapされるため、10点までで実用上ほぼ十分。
MAX_SCORE_FOR_OUTCOME = 10

OUTPUT_GRID_CSV = BASE_DIR / "v15_elo_outcome_dynamic_grid_results.csv"
OUTPUT_BEST_CSV = BASE_DIR / "v15_elo_outcome_dynamic_best_prediction.csv"
OUTPUT_BEST_HTML = BASE_DIR / "v15_elo_outcome_dynamic_best_prediction.html"


def poisson_pmf_array(lam, max_goals=10):
    """scipyなしでPoisson PMFを0〜max_goalsまで作る。"""
    lam = safe_lambda(lam)
    probs = np.zeros(max_goals + 1, dtype=float)
    probs[0] = np.exp(-lam)
    for k in range(1, max_goals + 1):
        probs[k] = probs[k - 1] * lam / k

    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        probs[:] = 0.0
        probs[0] = 1.0
    else:
        probs = probs / total
    return probs


def elo_home_win_share_non_draw(home, away, elo_ratings, scale=400, home_adv=0):
    """
    引き分け以外になったと仮定したとき、ホーム側が勝つ寄りかをEloで出す。
    0.5より大きいほどホーム優勢。
    """
    if elo_ratings is None:
        return 0.5

    home_elo = elo_ratings.get(home, INITIAL_ELO)
    away_elo = elo_ratings.get(away, INITIAL_ELO)
    elo_diff = (home_elo + home_adv) - away_elo

    share = 1 / (1 + 10 ** (-elo_diff / scale))
    return float(np.clip(share, 0.05, 0.95))


def sample_score_with_elo_outcome_blend(
    lambda_home,
    lambda_away,
    home,
    away,
    elo_ratings,
    elo_outcome_weight=0.0,
    max_goals=10,
):
    """
    Eloをλではなく勝敗確率に混ぜてから、条件付きでスコアを抽選する。

    手順:
      1. λからPoissonのスコア確率表を作る
      2. Poisson上のホーム勝ち/分け/アウェイ勝ち確率を集計
      3. Elo差から「引き分け以外ならホームが勝つ割合」を作る
      4. 引き分け率はPoissonのまま、非引き分け部分をEloで分配
      5. Poisson勝敗確率とElo勝敗確率をブレンド
      6. 勝敗結果を先に抽選し、その勝敗に合うスコアをPoisson条件付き分布から抽選

    elo_outcome_weight=0.0 の場合は、従来通りPoissonから直接スコアを抽選する。
    """
    if elo_outcome_weight <= 0 or elo_ratings is None:
        return int(np.random.poisson(lambda_home)), int(np.random.poisson(lambda_away))

    home_probs = poisson_pmf_array(lambda_home, max_goals=max_goals)
    away_probs = poisson_pmf_array(lambda_away, max_goals=max_goals)
    score_matrix = np.outer(home_probs, away_probs)

    total = score_matrix.sum()
    if not np.isfinite(total) or total <= 0:
        return int(np.random.poisson(lambda_home)), int(np.random.poisson(lambda_away))
    score_matrix = score_matrix / total

    goals = np.arange(max_goals + 1)
    hg_grid, ag_grid = np.meshgrid(goals, goals, indexing="ij")

    home_win_mask = hg_grid > ag_grid
    draw_mask = hg_grid == ag_grid
    away_win_mask = hg_grid < ag_grid

    p_home_poisson = float(score_matrix[home_win_mask].sum())
    p_draw_poisson = float(score_matrix[draw_mask].sum())
    p_away_poisson = float(score_matrix[away_win_mask].sum())

    non_draw = max(1.0 - p_draw_poisson, 1e-12)
    home_share_elo = elo_home_win_share_non_draw(
        home=home,
        away=away,
        elo_ratings=elo_ratings,
        scale=ELO_OUTCOME_SCALE,
        home_adv=HOME_ADV,
    )

    # Elo側の勝敗確率。引き分け率はPoissonに任せる。
    p_home_elo = non_draw * home_share_elo
    p_draw_elo = p_draw_poisson
    p_away_elo = non_draw * (1.0 - home_share_elo)

    w = float(np.clip(elo_outcome_weight, 0.0, 1.0))
    outcome_probs = np.array([
        (1.0 - w) * p_home_poisson + w * p_home_elo,
        (1.0 - w) * p_draw_poisson + w * p_draw_elo,
        (1.0 - w) * p_away_poisson + w * p_away_elo,
    ], dtype=float)

    outcome_total = outcome_probs.sum()
    if not np.isfinite(outcome_total) or outcome_total <= 0:
        return int(np.random.poisson(lambda_home)), int(np.random.poisson(lambda_away))
    outcome_probs = outcome_probs / outcome_total

    # 0: home win, 1: draw, 2: away win
    outcome = int(np.random.choice(3, p=outcome_probs))
    if outcome == 0:
        mask = home_win_mask
    elif outcome == 1:
        mask = draw_mask
    else:
        mask = away_win_mask

    conditional = score_matrix * mask
    conditional_total = conditional.sum()

    # 極端なλで条件付き分布が作れない場合の保険
    if not np.isfinite(conditional_total) or conditional_total <= 0:
        return int(np.random.poisson(lambda_home)), int(np.random.poisson(lambda_away))

    flat_probs = (conditional / conditional_total).ravel()
    idx = int(np.random.choice(flat_probs.size, p=flat_probs))
    hg = idx // (max_goals + 1)
    ag = idx % (max_goals + 1)

    return int(hg), int(ag)


def simulate_with_prev_and_elo_outcome(
    prev_setting,
    elo_outcome_weight,
    elo_update_every,
    n_sim,
    prepared,
    seed=42,
    verbose=False,
):
    """
    v1.4構造を保ったまま、Eloをλではなく勝敗確率側に使う検証。

    固定点:
      ・Eloのλ補正は使わない: ELO_LAMBDA_WEIGHT = 0.0相当
      ・Eloは勝敗確率側にだけ混ぜる
      ・シミュレーション内の仮想試合結果でEloを段階更新する
      ・λと攻守係数は後半戦中に更新しない
      ・相性Effectは従来通りλに掛ける
      ・大勝補正cap4、lambda capなどはv1.4設定を維持
      ・2025年前半係数はDECAY=1.0
      ・前年係数は候補ごとのPREV_WEIGHT/PREV_DECAYで作る
    """
    if seed is not None:
        np.random.seed(seed)

    prev_name = prev_setting["name"]
    prev_weight = float(prev_setting["prev_weight"])
    prev_decay = float(prev_setting["prev_decay"])
    elo_update_every = int(elo_update_every) if elo_update_every is not None else 0

    current_df = prepared["current_df"]
    previous_df = prepared["previous_df"]
    train_df = prepared["train_df"]
    test_df = prepared["test_df"]
    teams = prepared["teams"]
    current_strengths = prepared["current_strengths"]
    home_avg_goals = prepared["home_avg_goals"]
    away_avg_goals = prepared["away_avg_goals"]
    elo_ratings = prepared["elo_ratings"]
    matchup_effects = prepared["matchup_effects"]
    actual_position = prepared["actual_position"]

    previous_strengths, _, _ = calculate_strengths_home_away(
        previous_df,
        teams=teams,
        decay=prev_decay,
        feature_weight=PREV_SHOT_WEIGHT,
        feature_type="shots",
    )

    strengths, prev_weight_by_team, prev_games_by_team = blend_with_previous_strengths(
        current_strengths=current_strengths,
        previous_strengths=previous_strengths,
        prev_weight=prev_weight,
        previous_df=previous_df,
        use_promoted_prev_zero=USE_PROMOTED_PREV_ZERO,
        promoted_prev_weight=PROMOTED_PREV_WEIGHT,
    )

    position_counts = {team: Counter() for team in teams}
    points_sum = {team: 0.0 for team in teams}
    gf_sum = {team: 0.0 for team in teams}
    ga_sum = {team: 0.0 for team in teams}
    gd_sum = {team: 0.0 for team in teams}

    for sim in range(1, n_sim + 1):
        table = calculate_table(train_df, teams)

        # 各シミュレーションは、前半終了時点のEloから開始する。
        # elo_update_every=0 の場合は固定Eloとして使う。
        sim_elo_ratings = dict(elo_ratings)
        pending_elo_updates = []

        for row in test_df.itertuples(index=False):
            row_dict = row._asdict()
            home = row_dict["home"]
            away = row_dict["away"]

            # Eloはλに掛けない。相性Effectだけ従来通りλに反映する。
            lambda_home, lambda_away = expected_goals_home_away(
                home=home,
                away=away,
                strengths=strengths,
                home_avg_goals=home_avg_goals,
                away_avg_goals=away_avg_goals,
                elo_ratings=None,
                elo_lambda_weight=0.0,
                matchup_effects=matchup_effects,
                compat_weight=COMPAT_WEIGHT,
            )

            hg, ag = sample_score_with_elo_outcome_blend(
                lambda_home=lambda_home,
                lambda_away=lambda_away,
                home=home,
                away=away,
                elo_ratings=sim_elo_ratings,
                elo_outcome_weight=elo_outcome_weight,
                max_goals=MAX_SCORE_FOR_OUTCOME,
            )

            table[home]["gf"] += hg
            table[home]["ga"] += ag
            table[away]["gf"] += ag
            table[away]["ga"] += hg

            if hg > ag:
                table[home]["points"] += 3
            elif hg < ag:
                table[away]["points"] += 3
            else:
                table[home]["points"] += 1
                table[away]["points"] += 1

            table[home]["gd"] = table[home]["gf"] - table[home]["ga"]
            table[away]["gd"] = table[away]["gf"] - table[away]["ga"]

            # 案A: λは固定し、Eloだけを仮想試合結果で段階更新する。
            # 3試合おき更新の場合、直近3試合分をまとめて次の試合以降へ反映する。
            if elo_update_every > 0 and elo_outcome_weight > 0:
                pending_elo_updates.append((home, away, hg, ag))
                if len(pending_elo_updates) >= elo_update_every:
                    for uh, ua, uhg, uag in pending_elo_updates:
                        update_elo_one_match(
                            ratings=sim_elo_ratings,
                            home=uh,
                            away=ua,
                            home_goal=uhg,
                            away_goal=uag,
                            k_factor=K_FACTOR,
                            home_adv=HOME_ADV,
                        )
                    pending_elo_updates = []

        ranking = make_ranking(table)

        for pos, (team, stats) in enumerate(ranking):
            position_counts[team][pos + 1] += 1

        for team in teams:
            points_sum[team] += table[team]["points"]
            gf_sum[team] += table[team]["gf"]
            ga_sum[team] += table[team]["ga"]
            gd_sum[team] += table[team]["gd"]

        if verbose and sim % 1000 == 0:
            print(
                f"  {prev_name}, ELO_OUTCOME_WEIGHT={elo_outcome_weight:.2f}: "
                f"{sim}/{n_sim} 回終了"
            )

    rows = []
    n_teams = len(teams)

    for team in teams:
        avg_pred_pos = sum(
            pos * (position_counts[team][pos] / n_sim)
            for pos in range(1, n_teams + 1)
        )

        rows.append({
            "team": team,
            "actual_position": actual_position[team],
            "avg_pred_position": avg_pred_pos,
            "position_error": abs(avg_pred_pos - actual_position[team]),
            "prev_setting_name": prev_name,
            "prev_weight_candidate": prev_weight,
            "prev_decay_candidate": prev_decay,
            "elo_mode": "outcome_blend_dynamic_elo",
            "elo_lambda_weight_candidate": 0.0,
            "elo_outcome_weight_candidate": elo_outcome_weight,
            "elo_update_every_candidate": elo_update_every,
            "elo_outcome_scale": ELO_OUTCOME_SCALE,
            "prev_games_2024": prev_games_by_team.get(team),
            "effective_prev_weight": prev_weight_by_team.get(team),
            "goal_adjust_mode": GOAL_ADJUST_MODE,
            "goal_cap_for_strength": GOAL_CAP_FOR_STRENGTH,
            "most_likely_position": position_counts[team].most_common(1)[0][0],
            "champion_prob": position_counts[team][1] / n_sim,
            "top3_prob": sum(position_counts[team][p] for p in range(1, 4)) / n_sim,
            "top5_prob": sum(position_counts[team][p] for p in range(1, 6)) / n_sim,
            "bottom3_prob": sum(position_counts[team][p] for p in range(n_teams - 2, n_teams + 1)) / n_sim,
            "avg_points": points_sum[team] / n_sim,
            "avg_gf": gf_sum[team] / n_sim,
            "avg_ga": ga_sum[team] / n_sim,
            "avg_gd": gd_sum[team] / n_sim,
            "elo": elo_ratings.get(team, INITIAL_ELO),
        })

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values("avg_pred_position").reset_index(drop=True)
    result_df.insert(0, "pred_rank", result_df.index + 1)

    mae = float(result_df["position_error"].mean())
    max_error = float(result_df["position_error"].max())

    team_lookup = result_df.set_index("team")

    def team_value(team, col, default=np.nan):
        if team in team_lookup.index and col in team_lookup.columns:
            return team_lookup.loc[team, col]
        return default

    summary = {
        "prev_setting_name": prev_name,
        "prev_weight": prev_weight,
        "prev_decay": prev_decay,
        "elo_mode": "outcome_blend_dynamic_elo",
        "elo_lambda_weight": 0.0,
        "elo_outcome_weight": elo_outcome_weight,
        "elo_update_every": elo_update_every,
        "elo_outcome_scale": ELO_OUTCOME_SCALE,
        "n_sim": n_sim,
        "mae": mae,
        "max_error": max_error,
        "kawasaki_actual_position": team_value("川崎F", "actual_position"),
        "kawasaki_avg_pred_position": team_value("川崎F", "avg_pred_position"),
        "kawasaki_error": team_value("川崎F", "position_error"),
        "kashiwa_actual_position": team_value("柏", "actual_position"),
        "kashiwa_avg_pred_position": team_value("柏", "avg_pred_position"),
        "kashiwa_error": team_value("柏", "position_error"),
        "kyoto_actual_position": team_value("京都", "actual_position"),
        "kyoto_avg_pred_position": team_value("京都", "avg_pred_position"),
        "kyoto_error": team_value("京都", "position_error"),
        "shimizu_actual_position": team_value("清水", "actual_position"),
        "shimizu_avg_pred_position": team_value("清水", "avg_pred_position"),
        "shimizu_error": team_value("清水", "position_error"),
        "machida_actual_position": team_value("町田", "actual_position"),
        "machida_avg_pred_position": team_value("町田", "avg_pred_position"),
        "machida_error": team_value("町田", "position_error"),
        "nagoya_actual_position": team_value("名古屋", "actual_position"),
        "nagoya_avg_pred_position": team_value("名古屋", "avg_pred_position"),
        "nagoya_error": team_value("名古屋", "position_error"),
        "yokohamafm_actual_position": team_value("横浜FM", "actual_position"),
        "yokohamafm_avg_pred_position": team_value("横浜FM", "avg_pred_position"),
        "yokohamafm_error": team_value("横浜FM", "position_error"),
        "fctokyo_actual_position": team_value("FC東京", "actual_position"),
        "fctokyo_avg_pred_position": team_value("FC東京", "avg_pred_position"),
        "fctokyo_error": team_value("FC東京", "position_error"),
    }

    return summary, result_df


def export_grid_html(df, output_path, title):
    display_df = df.copy()

    percent_cols = ["champion_prob", "top3_prob", "top5_prob", "bottom3_prob"]
    for col in percent_cols:
        if col in display_df.columns:
            display_df[col] = (display_df[col] * 100).round(1).astype(str) + "%"

    round_cols = [
        "avg_pred_position", "avg_points", "avg_gf", "avg_ga", "avg_gd",
        "position_error", "elo", "prev_weight_candidate", "prev_decay_candidate",
        "elo_lambda_weight_candidate", "elo_outcome_weight_candidate",
        "elo_update_every_candidate",
    ]
    for col in round_cols:
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(display_df[col], errors="coerce").round(3)

    column_names = {
        "pred_rank": "予測順位",
        "team": "チーム",
        "actual_position": "実順位",
        "avg_pred_position": "平均予測順位",
        "position_error": "順位誤差",
        "prev_setting_name": "前年設定",
        "prev_weight_candidate": "PREV_WEIGHT",
        "prev_decay_candidate": "PREV_DECAY",
        "elo_mode": "Elo方式",
        "elo_lambda_weight_candidate": "Elo λ重み",
        "elo_outcome_weight_candidate": "Elo勝敗重み",
        "elo_update_every_candidate": "Elo更新間隔",
        "effective_prev_weight": "実効前年重み",
        "most_likely_position": "最頻順位",
        "champion_prob": "優勝確率",
        "top3_prob": "上位3確率",
        "top5_prob": "上位5確率",
        "bottom3_prob": "下位3確率",
        "avg_points": "平均勝点",
        "avg_gf": "平均得点",
        "avg_ga": "平均失点",
        "avg_gd": "平均得失点差",
        "elo": "Elo",
    }
    display_df = display_df.rename(columns=column_names)

    table_html = display_df.to_html(index=False, classes="prediction-table")

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
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
      overflow-x: auto; background: white; padding: 16px;
      border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
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
  <h1>{title}</h1>
  <div class="note">
    <p>
      v1.4をベースに、Eloを期待得点λではなく勝敗確率側に混ぜる方式を検証した結果です。
      大勝補正={GOAL_ADJUST_MODE}, cap={GOAL_CAP_FOR_STRENGTH},
      2025枠内重み={SOT_WEIGHT}, 2024総シュート重み={PREV_SHOT_WEIGHT},
      相性Effect={"ON" if USE_MATCHUP_EFFECT else "OFF"}。
    </p>
  </div>
  <div class="table-wrap">
    {table_html}
  </div>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def prepare_common_data():
    current_df = load_match_stats_csv(CURRENT_CSV)
    previous_df = load_match_stats_csv(PREVIOUS_CSV)

    teams = list(pd.unique(current_df[["home", "away"]].values.ravel()))

    split = int(len(current_df) * 0.5)
    train_df = current_df.iloc[:split].copy()
    test_df = current_df.iloc[split:].copy()

    print("\n==============================")
    print("データ確認")
    print("==============================")
    print("CURRENT_CSV:", CURRENT_CSV)
    print("PREVIOUS_CSV:", PREVIOUS_CSV)
    print("GOAL_ADJUST_MODE:", GOAL_ADJUST_MODE)
    print("GOAL_CAP_FOR_STRENGTH:", GOAL_CAP_FOR_STRENGTH)
    print("USE_RAW_LEAGUE_AVG_FOR_LAMBDA:", USE_RAW_LEAGUE_AVG_FOR_LAMBDA)
    print("2025試合数:", len(current_df))
    print("train試合数:", len(train_df))
    print("test試合数:", len(test_df))
    print("チーム数:", len(teams))
    print("N_SIM_SEARCH:", N_SIM_SEARCH)
    print("N_SIM_FINAL:", N_SIM_FINAL)
    print("PREV_SETTING_CANDIDATES:", PREV_SETTING_CANDIDATES)
    print("ELO_OUTCOME_WEIGHT_CANDIDATES:", ELO_OUTCOME_WEIGHT_CANDIDATES)
    print("ELO_UPDATE_EVERY_CANDIDATES:", ELO_UPDATE_EVERY_CANDIDATES)
    print("ELO_OUTCOME_SCALE:", ELO_OUTCOME_SCALE)
    print("MAX_SCORE_FOR_OUTCOME:", MAX_SCORE_FOR_OUTCOME)

    current_strengths, home_avg_goals, away_avg_goals = calculate_strengths_home_away(
        train_df,
        teams=teams,
        decay=DECAY,
        feature_weight=SOT_WEIGHT,
        feature_type="sot",
    )

    elo_ratings = build_elo_ratings(
        previous_df=previous_df,
        train_df=train_df,
        teams=teams,
        k_factor=K_FACTOR,
        home_adv=HOME_ADV,
    )

    matchup_effects = None
    if USE_MATCHUP_EFFECT:
        historical_path = find_file(HISTORICAL_J1_FILENAME)
        historical_j1_df = load_historical_j1_csv(historical_path)

        cutoff_date = train_df["date"].max()
        target_year = int(current_df["date"].dt.year.max())

        matchup_effects = build_matchup_effects_j1(
            historical_j1_df=historical_j1_df,
            cutoff_date=cutoff_date,
            target_year=target_year,
            prior_n=MATCHUP_PRIOR_N,
            time_decay=MATCHUP_TIME_DECAY,
            k_factor=K_FACTOR,
            home_adv=HOME_ADV,
        )

        print("HISTORICAL_J1_CSV:", historical_path)
        print("相性Effect件数:", len(matchup_effects))

    actual_table = calculate_table(current_df, teams)
    actual_ranking = make_ranking(actual_table)
    actual_position = {
        team: pos + 1
        for pos, (team, stats) in enumerate(actual_ranking)
    }

    return {
        "current_df": current_df,
        "previous_df": previous_df,
        "train_df": train_df,
        "test_df": test_df,
        "teams": teams,
        "current_strengths": current_strengths,
        "home_avg_goals": home_avg_goals,
        "away_avg_goals": away_avg_goals,
        "elo_ratings": elo_ratings,
        "matchup_effects": matchup_effects,
        "actual_position": actual_position,
    }


def main():
    prepared = prepare_common_data()

    grid_rows = []
    best_summary = None
    best_prediction = None

    total = (
        len(PREV_SETTING_CANDIDATES)
        * len(ELO_OUTCOME_WEIGHT_CANDIDATES)
        * len(ELO_UPDATE_EVERY_CANDIDATES)
    )
    done = 0

    print("\n==============================")
    print("PREV設定候補 × Elo勝敗ブレンド＋段階更新 × Elo段階更新 グリッドサーチ開始")
    print("==============================")

    for prev_setting in PREV_SETTING_CANDIDATES:
        for elo_outcome_weight in ELO_OUTCOME_WEIGHT_CANDIDATES:
            for elo_update_every in ELO_UPDATE_EVERY_CANDIDATES:
                # Elo勝敗重み0では更新しても結果が同じなので、固定Eloだけ残して重複を避ける。
                if np.isclose(elo_outcome_weight, 0.0) and elo_update_every != 0:
                    continue

                done += 1
                print(
                    f"\n[{done}/{total}] "
                    f"{prev_setting['name']} "
                    f"PREV_WEIGHT={prev_setting['prev_weight']:.2f}, "
                    f"PREV_DECAY={prev_setting['prev_decay']:.3f}, "
                    f"ELO_OUTCOME_WEIGHT={elo_outcome_weight:.2f}, "
                    f"ELO_UPDATE_EVERY={elo_update_every}"
                )
                summary, prediction_df = simulate_with_prev_and_elo_outcome(
                    prev_setting=prev_setting,
                    elo_outcome_weight=elo_outcome_weight,
                    elo_update_every=elo_update_every,
                    n_sim=N_SIM_SEARCH,
                    prepared=prepared,
                    seed=RANDOM_SEED,
                    verbose=False,
                )
                grid_rows.append(summary)
                print(
                    f"  MAE={summary['mae']:.4f} / "
                    f"川崎F予測={summary['kawasaki_avg_pred_position']:.2f} "
                    f"誤差={summary['kawasaki_error']:.2f} / "
                    f"京都誤差={summary['kyoto_error']:.2f} / "
                    f"柏誤差={summary['kashiwa_error']:.2f}"
                )

                if best_summary is None or summary["mae"] < best_summary["mae"]:
                    best_summary = summary
                    best_prediction = prediction_df

    grid_df = pd.DataFrame(grid_rows)

    # 各PREV設定内で、Elo勝敗補正なしとの差分を見る
    no_elo_mae_by_prev = (
        grid_df[np.isclose(grid_df["elo_outcome_weight"], 0.0)]
        .set_index("prev_setting_name")["mae"]
        .to_dict()
    )
    grid_df["delta_vs_no_elo_same_prev"] = grid_df.apply(
        lambda r: r["mae"] - no_elo_mae_by_prev.get(r["prev_setting_name"], np.nan),
        axis=1,
    )

    # 第一候補のEloなしとの差分
    baseline_rows = grid_df[
        (grid_df["prev_setting_name"] == "prev060_decay0985")
        & np.isclose(grid_df["elo_outcome_weight"], 0.0)
    ]
    if len(baseline_rows) > 0:
        baseline_mae = float(baseline_rows.iloc[0]["mae"])
        grid_df["delta_vs_prev060_decay0985_no_elo"] = grid_df["mae"] - baseline_mae
    else:
        grid_df["delta_vs_prev060_decay0985_no_elo"] = np.nan

    grid_df["rank_within_prev_setting"] = (
        grid_df.groupby("prev_setting_name")["mae"]
        .rank(method="min", ascending=True)
        .astype(int)
    )
    grid_df["rank_within_prev_and_update"] = (
        grid_df.groupby(["prev_setting_name", "elo_update_every"])["mae"]
        .rank(method="min", ascending=True)
        .astype(int)
    )

    grid_df = grid_df.sort_values("mae").reset_index(drop=True)
    grid_df.insert(0, "rank", grid_df.index + 1)
    grid_df.to_csv(OUTPUT_GRID_CSV, index=False, encoding="utf-8-sig")

    print("\n==============================")
    print("探索結果 上位")
    print("==============================")
    show_cols = [
        "rank", "prev_setting_name", "prev_weight", "prev_decay",
        "elo_mode", "elo_outcome_weight", "elo_update_every", "mae",
        "delta_vs_no_elo_same_prev", "delta_vs_prev060_decay0985_no_elo",
        "kawasaki_avg_pred_position", "kawasaki_error",
        "kyoto_avg_pred_position", "kyoto_error",
        "kashiwa_avg_pred_position", "kashiwa_error",
        "machida_error", "nagoya_error", "shimizu_error", "fctokyo_error",
    ]
    print(grid_df[show_cols].head(15).to_string(index=False))
    print("\nGRID CSV:", OUTPUT_GRID_CSV)

    if RUN_FINAL_BEST and best_summary is not None:
        best_prev_setting_name = str(best_summary["prev_setting_name"])
        best_prev_weight = float(best_summary["prev_weight"])
        best_prev_decay = float(best_summary["prev_decay"])
        best_elo_outcome_weight = float(best_summary["elo_outcome_weight"])
        best_elo_update_every = int(best_summary["elo_update_every"])

        best_prev_setting = {
            "name": best_prev_setting_name,
            "prev_weight": best_prev_weight,
            "prev_decay": best_prev_decay,
        }

        print("\n==============================")
        print("最良候補をN_SIM_FINALで再実行")
        print("==============================")
        print("best PREV_SETTING:", best_prev_setting_name)
        print("best PREV_WEIGHT:", best_prev_weight)
        print("best PREV_DECAY:", best_prev_decay)
        print("best ELO_OUTCOME_WEIGHT:", best_elo_outcome_weight)
        print("best ELO_UPDATE_EVERY:", best_elo_update_every)

        final_summary, final_prediction = simulate_with_prev_and_elo_outcome(
            prev_setting=best_prev_setting,
            elo_outcome_weight=best_elo_outcome_weight,
            elo_update_every=best_elo_update_every,
            n_sim=N_SIM_FINAL,
            prepared=prepared,
            seed=RANDOM_SEED,
            verbose=True,
        )

        final_prediction.to_csv(OUTPUT_BEST_CSV, index=False, encoding="utf-8-sig")
        export_grid_html(
            final_prediction,
            OUTPUT_BEST_HTML,
            title=(
                "J1 2025 順位予測 v1.5候補 Elo勝敗ブレンド＋段階更新 "
                f"{best_prev_setting_name}, outcome={best_elo_outcome_weight:.2f}, update={best_elo_update_every}"
            ),
        )

        print("\n==============================")
        print("最終確認")
        print("==============================")
        print("PREV_SETTING:", best_prev_setting_name)
        print("PREV_WEIGHT:", best_prev_weight)
        print("PREV_DECAY:", best_prev_decay)
        print("ELO_MODE: outcome_blend_dynamic_elo")
        print("ELO_LAMBDA_WEIGHT: 0.0")
        print("ELO_OUTCOME_WEIGHT:", best_elo_outcome_weight)
        print("ELO_UPDATE_EVERY:", best_elo_update_every)
        print("MAE:", round(final_summary["mae"], 4))
        print("川崎F 平均予測順位:", round(final_summary["kawasaki_avg_pred_position"], 2))
        print("川崎F 誤差:", round(final_summary["kawasaki_error"], 2))
        print("京都 平均予測順位:", round(final_summary["kyoto_avg_pred_position"], 2))
        print("京都 誤差:", round(final_summary["kyoto_error"], 2))
        print("柏 平均予測順位:", round(final_summary["kashiwa_avg_pred_position"], 2))
        print("柏 誤差:", round(final_summary["kashiwa_error"], 2))
        print("BEST CSV:", OUTPUT_BEST_CSV)
        print("BEST HTML:", OUTPUT_BEST_HTML)
        print("\n予測順位表:")
        print(final_prediction[[
            "pred_rank", "team", "actual_position", "avg_pred_position",
            "position_error", "prev_setting_name", "prev_weight_candidate", "prev_decay_candidate",
            "elo_mode", "elo_outcome_weight_candidate", "elo_update_every_candidate", "effective_prev_weight",
            "champion_prob", "top3_prob", "bottom3_prob", "avg_points", "avg_gd", "elo",
        ]].to_string(index=False))


if __name__ == "__main__":
    main()
