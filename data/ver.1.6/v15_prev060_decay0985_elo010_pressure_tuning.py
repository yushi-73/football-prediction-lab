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
# 10. 終盤プレッシャー補正
# =========================

def calculate_games_played(match_df, teams):
    """各チームの消化試合数を数える。"""
    played = {team: 0 for team in teams}
    for row in match_df.itertuples(index=False):
        row_dict = row._asdict()
        home = row_dict["home"]
        away = row_dict["away"]
        if home not in played:
            played[home] = 0
        if away not in played:
            played[away] = 0
        played[home] += 1
        played[away] += 1
    return played


def calculate_total_games(current_df, teams):
    """既知のシーズン日程から、各チームの総試合数を数える。"""
    return calculate_games_played(current_df, teams)


def late_season_factor(played, total_games):
    """
    シーズン終盤度を0〜1で返す。
    LATE_SEASON_START_RATIOより前は0、最終節に近づくほど1に近づく。
    """
    total_games = max(int(total_games), 1)
    ratio = played / total_games

    if ratio <= LATE_SEASON_START_RATIO:
        return 0.0

    denom = max(1e-9, 1.0 - LATE_SEASON_START_RATIO)
    return float(np.clip((ratio - LATE_SEASON_START_RATIO) / denom, 0.0, 1.0))


def calculate_pressure_context(table, played, total_games_by_team, teams):
    """
    試合直前の順位表だけを使って、残留圧力・優勝圧力を計算する。

    未来情報を避けるため、ここで使うのはシミュレーション中に更新されている
    table / played / 既知の日程上の総試合数だけ。
    """
    ranking = make_ranking(table)
    n_teams = len(ranking)

    rank_by_team = {team: pos + 1 for pos, (team, _) in enumerate(ranking)}
    points_by_team = {team: stats["points"] for team, stats in table.items()}

    # 下位3クラブが降格圏なら、残留ラインは17位の勝点。
    # チーム数が変わっても動くように安全側で処理する。
    safety_rank = max(1, n_teams - RELEGATION_SPOTS)
    safety_index = min(max(safety_rank - 1, 0), n_teams - 1)
    safety_line_points = ranking[safety_index][1]["points"]
    top_points = ranking[0][1]["points"]

    context = {}

    for team in teams:
        team_points = points_by_team.get(team, 0)
        team_rank = rank_by_team.get(team, n_teams)
        team_played = played.get(team, 0)
        team_total_games = total_games_by_team.get(team, max(team_played, 1))
        late = late_season_factor(team_played, team_total_games)

        # 残留ラインに近いほど高い。上でも下でも、6勝点以内なら効く。
        rel_distance = abs(team_points - safety_line_points)
        rel_pressure = max(0.0, 1.0 - rel_distance / RELEGATION_POINTS_WINDOW)

        # あまりに上位のチームに偶然効かないよう、下位寄りだけに制限する。
        lower_table_gate = 1.0 if team_rank >= n_teams - RELEGATION_SPOTS - 3 else 0.0
        rel_pressure *= lower_table_gate * late

        # 首位に近いほど高い。首位も追う側も対象。
        title_distance = max(0.0, top_points - team_points)
        title_pressure = max(0.0, 1.0 - title_distance / TITLE_POINTS_WINDOW)

        # あまりに中位以下のチームに偶然効かないよう、上位寄りだけに制限する。
        upper_table_gate = 1.0 if team_rank <= 5 else 0.0
        title_pressure *= upper_table_gate * late

        context[team] = {
            "rank": team_rank,
            "points": team_points,
            "late": late,
            "relegation_pressure": float(np.clip(rel_pressure, 0.0, 1.0)),
            "title_pressure": float(np.clip(title_pressure, 0.0, 1.0)),
        }

    return context


def apply_pressure_effects_to_lambdas(lambda_home, lambda_away, home, away, pressure_context):
    """
    残留ブースト・優勝デバフをλに反映する。

    残留争い:
      ・自チームの得点λを上げる
      ・相手の得点λを下げる
    優勝争い:
      ・自チームの得点λを下げる
    """
    if not USE_PRESSURE_EFFECTS:
        return safe_lambda(lambda_home), safe_lambda(lambda_away), {
            "home_relegation_pressure": 0.0,
            "away_relegation_pressure": 0.0,
            "home_title_pressure": 0.0,
            "away_title_pressure": 0.0,
        }

    home_ctx = pressure_context.get(home, {})
    away_ctx = pressure_context.get(away, {})

    home_rel = float(home_ctx.get("relegation_pressure", 0.0))
    away_rel = float(away_ctx.get("relegation_pressure", 0.0))
    home_title = float(home_ctx.get("title_pressure", 0.0))
    away_title = float(away_ctx.get("title_pressure", 0.0))

    home_multiplier = np.exp(
        RELEGATION_ATTACK_LAMBDA_BOOST * home_rel
        - RELEGATION_DEFENSE_LAMBDA_BOOST * away_rel
        - TITLE_ATTACK_LAMBDA_DEBUFF * home_title
    )
    away_multiplier = np.exp(
        RELEGATION_ATTACK_LAMBDA_BOOST * away_rel
        - RELEGATION_DEFENSE_LAMBDA_BOOST * home_rel
        - TITLE_ATTACK_LAMBDA_DEBUFF * away_title
    )

    adjusted_home = safe_lambda(lambda_home * home_multiplier)
    adjusted_away = safe_lambda(lambda_away * away_multiplier)

    pressure_info = {
        "home_relegation_pressure": home_rel,
        "away_relegation_pressure": away_rel,
        "home_title_pressure": home_title,
        "away_title_pressure": away_title,
    }

    return adjusted_home, adjusted_away, pressure_info


def apply_pressure_draw_shift(home_goal, away_goal, pressure_info):
    """
    1点差決着だけ、終盤プレッシャーに応じて引き分けへ寄せる。

    例:
      1-0 → 1-1
      2-1 → 2-2

    大差試合を無理に引き分け化すると不自然なので、1点差限定にしている。
    """
    if not USE_PRESSURE_EFFECTS:
        return home_goal, away_goal

    if abs(home_goal - away_goal) != 1:
        return home_goal, away_goal

    rel_pressure = max(
        pressure_info.get("home_relegation_pressure", 0.0),
        pressure_info.get("away_relegation_pressure", 0.0),
    )
    title_pressure = max(
        pressure_info.get("home_title_pressure", 0.0),
        pressure_info.get("away_title_pressure", 0.0),
    )

    draw_shift_prob = (
        DRAW_SHIFT_RELEGATION * rel_pressure
        + DRAW_SHIFT_TITLE * title_pressure
    )
    draw_shift_prob = float(np.clip(draw_shift_prob, 0.0, MAX_DRAW_SHIFT))

    if np.random.random() < draw_shift_prob:
        if home_goal > away_goal:
            away_goal = home_goal
        else:
            home_goal = away_goal

    return home_goal, away_goal


# =========================
# 11. HTML出力
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
# 11. v1.5候補: PREV設定固定候補 × Elo補正検証
# =========================

# 探索中は2000回、最良候補だけ10000回で再実行します。
# 重い場合は N_SIM_SEARCH を 1000 に下げてください。
N_SIM_SEARCH = 1000
N_SIM_FINAL = 10000
RUN_FINAL_BEST = True

# PREV設定は 0.60 / 0.985、Elo補正重みは0.10に固定する。
PREV_SETTING_FIXED = {"name": "prev060_decay0985", "prev_weight": 0.60, "prev_decay": 0.985}
ELO_LAMBDA_WEIGHT_FIXED = 0.10

# 既存の表示関数・prepare_common_dataが参照するため、単一候補としても保持する。
PREV_SETTING_CANDIDATES = [PREV_SETTING_FIXED]
ELO_LAMBDA_WEIGHT_CANDIDATES = [ELO_LAMBDA_WEIGHT_FIXED]

OUTPUT_GRID_CSV = BASE_DIR / "v15_prev060_decay0985_elo010_pressure_tuning_results.csv"
OUTPUT_GRID_HTML = BASE_DIR / "v15_prev060_decay0985_elo010_pressure_tuning_results.html"
OUTPUT_BEST_CSV = BASE_DIR / "v15_prev060_decay0985_elo010_pressure_tuning_best_prediction.csv"
OUTPUT_BEST_HTML = BASE_DIR / "v15_prev060_decay0985_elo010_pressure_tuning_best_prediction.html"

# =========================
# 12. 終盤プレッシャー補正 チューニング設定
# =========================

PRESSURE_SETTING_NAME = "base"
USE_PRESSURE_EFFECTS = True
RELEGATION_SPOTS = 3
RELEGATION_POINTS_WINDOW = 6.0
TITLE_POINTS_WINDOW = 6.0
LATE_SEASON_START_RATIO = 0.65
RELEGATION_ATTACK_LAMBDA_BOOST = 0.04
RELEGATION_DEFENSE_LAMBDA_BOOST = 0.04
TITLE_ATTACK_LAMBDA_DEBUFF = 0.05
DRAW_SHIFT_RELEGATION = 0.035
DRAW_SHIFT_TITLE = 0.040
MAX_DRAW_SHIFT = 0.08

# 候補はここを増減させれば調整できます。
# まずは、補正なし・残留のみ・優勝のみ・両方弱/中/強を比較する構成です。
PRESSURE_SETTING_CANDIDATES = [
    {
        "name": "pressure_off",
        "use_pressure_effects": False,
        "late_season_start_ratio": 0.65,
        "relegation_points_window": 6.0,
        "title_points_window": 6.0,
        "relegation_attack_lambda_boost": 0.00,
        "relegation_defense_lambda_boost": 0.00,
        "title_attack_lambda_debuff": 0.00,
        "draw_shift_relegation": 0.000,
        "draw_shift_title": 0.000,
        "max_draw_shift": 0.00,
    },
    {
        "name": "relegation_weak_title_off",
        "use_pressure_effects": True,
        "late_season_start_ratio": 0.65,
        "relegation_points_window": 6.0,
        "title_points_window": 6.0,
        "relegation_attack_lambda_boost": 0.02,
        "relegation_defense_lambda_boost": 0.02,
        "title_attack_lambda_debuff": 0.00,
        "draw_shift_relegation": 0.020,
        "draw_shift_title": 0.000,
        "max_draw_shift": 0.06,
    },
    {
        "name": "relegation_mid_title_off",
        "use_pressure_effects": True,
        "late_season_start_ratio": 0.65,
        "relegation_points_window": 6.0,
        "title_points_window": 6.0,
        "relegation_attack_lambda_boost": 0.04,
        "relegation_defense_lambda_boost": 0.04,
        "title_attack_lambda_debuff": 0.00,
        "draw_shift_relegation": 0.035,
        "draw_shift_title": 0.000,
        "max_draw_shift": 0.08,
    },
    {
        "name": "title_weak_relegation_off",
        "use_pressure_effects": True,
        "late_season_start_ratio": 0.65,
        "relegation_points_window": 6.0,
        "title_points_window": 6.0,
        "relegation_attack_lambda_boost": 0.00,
        "relegation_defense_lambda_boost": 0.00,
        "title_attack_lambda_debuff": 0.03,
        "draw_shift_relegation": 0.000,
        "draw_shift_title": 0.025,
        "max_draw_shift": 0.06,
    },
    {
        "name": "title_mid_relegation_off",
        "use_pressure_effects": True,
        "late_season_start_ratio": 0.65,
        "relegation_points_window": 6.0,
        "title_points_window": 6.0,
        "relegation_attack_lambda_boost": 0.00,
        "relegation_defense_lambda_boost": 0.00,
        "title_attack_lambda_debuff": 0.05,
        "draw_shift_relegation": 0.000,
        "draw_shift_title": 0.040,
        "max_draw_shift": 0.08,
    },
    {
        "name": "both_weak",
        "use_pressure_effects": True,
        "late_season_start_ratio": 0.65,
        "relegation_points_window": 6.0,
        "title_points_window": 6.0,
        "relegation_attack_lambda_boost": 0.02,
        "relegation_defense_lambda_boost": 0.02,
        "title_attack_lambda_debuff": 0.03,
        "draw_shift_relegation": 0.020,
        "draw_shift_title": 0.025,
        "max_draw_shift": 0.06,
    },
    {
        "name": "both_mid",
        "use_pressure_effects": True,
        "late_season_start_ratio": 0.65,
        "relegation_points_window": 6.0,
        "title_points_window": 6.0,
        "relegation_attack_lambda_boost": 0.04,
        "relegation_defense_lambda_boost": 0.04,
        "title_attack_lambda_debuff": 0.05,
        "draw_shift_relegation": 0.035,
        "draw_shift_title": 0.040,
        "max_draw_shift": 0.08,
    },
    {
        "name": "both_strong",
        "use_pressure_effects": True,
        "late_season_start_ratio": 0.65,
        "relegation_points_window": 6.0,
        "title_points_window": 6.0,
        "relegation_attack_lambda_boost": 0.06,
        "relegation_defense_lambda_boost": 0.06,
        "title_attack_lambda_debuff": 0.07,
        "draw_shift_relegation": 0.050,
        "draw_shift_title": 0.055,
        "max_draw_shift": 0.10,
    },
    {
        "name": "both_mid_late070",
        "use_pressure_effects": True,
        "late_season_start_ratio": 0.70,
        "relegation_points_window": 6.0,
        "title_points_window": 6.0,
        "relegation_attack_lambda_boost": 0.04,
        "relegation_defense_lambda_boost": 0.04,
        "title_attack_lambda_debuff": 0.05,
        "draw_shift_relegation": 0.035,
        "draw_shift_title": 0.040,
        "max_draw_shift": 0.08,
    },
    {
        "name": "both_mid_window4",
        "use_pressure_effects": True,
        "late_season_start_ratio": 0.65,
        "relegation_points_window": 4.0,
        "title_points_window": 4.0,
        "relegation_attack_lambda_boost": 0.04,
        "relegation_defense_lambda_boost": 0.04,
        "title_attack_lambda_debuff": 0.05,
        "draw_shift_relegation": 0.035,
        "draw_shift_title": 0.040,
        "max_draw_shift": 0.08,
    },
]


def apply_pressure_setting(setting):
    """PRESSURE_SETTING_CANDIDATESの1候補をグローバル設定に反映する。"""
    global PRESSURE_SETTING_NAME
    global USE_PRESSURE_EFFECTS
    global LATE_SEASON_START_RATIO
    global RELEGATION_POINTS_WINDOW
    global TITLE_POINTS_WINDOW
    global RELEGATION_ATTACK_LAMBDA_BOOST
    global RELEGATION_DEFENSE_LAMBDA_BOOST
    global TITLE_ATTACK_LAMBDA_DEBUFF
    global DRAW_SHIFT_RELEGATION
    global DRAW_SHIFT_TITLE
    global MAX_DRAW_SHIFT

    PRESSURE_SETTING_NAME = setting["name"]
    USE_PRESSURE_EFFECTS = bool(setting["use_pressure_effects"])
    LATE_SEASON_START_RATIO = float(setting["late_season_start_ratio"])
    RELEGATION_POINTS_WINDOW = float(setting["relegation_points_window"])
    TITLE_POINTS_WINDOW = float(setting["title_points_window"])
    RELEGATION_ATTACK_LAMBDA_BOOST = float(setting["relegation_attack_lambda_boost"])
    RELEGATION_DEFENSE_LAMBDA_BOOST = float(setting["relegation_defense_lambda_boost"])
    TITLE_ATTACK_LAMBDA_DEBUFF = float(setting["title_attack_lambda_debuff"])
    DRAW_SHIFT_RELEGATION = float(setting["draw_shift_relegation"])
    DRAW_SHIFT_TITLE = float(setting["draw_shift_title"])
    MAX_DRAW_SHIFT = float(setting["max_draw_shift"])


def export_pressure_grid_html(df, output_path, title):
    """圧力チューニング結果の一覧HTMLを出力する。"""
    display_df = df.copy()

    round_cols = [
        "mae", "max_error", "sim_draw_rate", "delta_vs_pressure_off",
        "late_season_start_ratio", "relegation_points_window", "title_points_window",
        "relegation_attack_lambda_boost", "relegation_defense_lambda_boost",
        "title_attack_lambda_debuff", "draw_shift_relegation", "draw_shift_title",
        "max_draw_shift", "kawasaki_error", "kyoto_error", "kashiwa_error",
        "machida_error", "nagoya_error", "shimizu_error", "fctokyo_error",
    ]
    for col in round_cols:
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(display_df[col], errors="coerce").round(4)

    column_names = {
        "rank": "順位",
        "pressure_setting_name": "圧力設定",
        "use_pressure_effects": "補正ON",
        "late_season_start_ratio": "発動開始消化率",
        "relegation_points_window": "残留勝点窓",
        "title_points_window": "優勝勝点窓",
        "relegation_attack_lambda_boost": "残留攻撃λ+",
        "relegation_defense_lambda_boost": "残留守備λ-",
        "title_attack_lambda_debuff": "優勝攻撃λ-",
        "draw_shift_relegation": "残留引分寄せ",
        "draw_shift_title": "優勝引分寄せ",
        "max_draw_shift": "引分寄せ上限",
        "sim_draw_rate": "シミュ引分率",
        "mae": "MAE",
        "delta_vs_pressure_off": "補正OFF比",
        "kawasaki_error": "川崎F誤差",
        "kyoto_error": "京都誤差",
        "kashiwa_error": "柏誤差",
        "machida_error": "町田誤差",
        "nagoya_error": "名古屋誤差",
        "shimizu_error": "清水誤差",
        "fctokyo_error": "FC東京誤差",
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
  <h1>{title}</h1>
  <div class="note">
    <p>
      PREV_WEIGHT=0.60, PREV_DECAY=0.985, ELO_LAMBDA_WEIGHT=0.10 は固定。
      残留ブースト・優勝デバフの強さだけを比較しています。
    </p>
    <p>
      N_SIM_SEARCH={N_SIM_SEARCH}。上位候補は必要に応じて N_SIM_FINAL={N_SIM_FINAL} で再実行します。
    </p>
  </div>
  <div class="table-wrap">
    {table_html}
  </div>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")

def simulate_with_prev_and_elo(
    prev_setting,
    elo_lambda_weight,
    n_sim,
    prepared,
    seed=42,
    verbose=False,
):
    """
    v1.4構造を保ったまま、PREV設定候補とELO_LAMBDA_WEIGHTを検証する。

    変更点:
      ・PREV_WEIGHT / PREV_DECAY は PREV_SETTING_CANDIDATES から選ぶ
      ・Elo補正の強さ ELO_LAMBDA_WEIGHT を差し替える

    固定点:
      ・2025年前半係数はDECAY=1.0のまま
      ・2024前年係数は得点+総シュートで作る
      ・相性Effect、大勝補正cap4、lambda capなどはv1.4設定を維持
      ・昇格組はUSE_PROMOTED_PREV_ZERO=Trueなら前年重み0.0
    """
    if seed is not None:
        np.random.seed(seed)

    prev_name = prev_setting["name"]
    prev_weight = float(prev_setting["prev_weight"])
    prev_decay = float(prev_setting["prev_decay"])

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
    rel_pressure_sum = {team: 0.0 for team in teams}
    title_pressure_sum = {team: 0.0 for team in teams}
    pressure_sample_count = {team: 0 for team in teams}
    simulated_match_count = 0
    draw_count = 0

    total_games_by_team = calculate_total_games(current_df, teams)

    for sim in range(1, n_sim + 1):
        table = calculate_table(train_df, teams)
        played = calculate_games_played(train_df, teams)

        for row in test_df.itertuples(index=False):
            row_dict = row._asdict()
            home = row_dict["home"]
            away = row_dict["away"]

            lambda_home, lambda_away = expected_goals_home_away(
                home=home,
                away=away,
                strengths=strengths,
                home_avg_goals=home_avg_goals,
                away_avg_goals=away_avg_goals,
                elo_ratings=elo_ratings,
                elo_lambda_weight=elo_lambda_weight,
                matchup_effects=matchup_effects,
                compat_weight=COMPAT_WEIGHT,
            )

            pressure_context = calculate_pressure_context(
                table=table,
                played=played,
                total_games_by_team=total_games_by_team,
                teams=teams,
            )

            lambda_home, lambda_away, pressure_info = apply_pressure_effects_to_lambdas(
                lambda_home=lambda_home,
                lambda_away=lambda_away,
                home=home,
                away=away,
                pressure_context=pressure_context,
            )

            rel_pressure_sum[home] += pressure_info["home_relegation_pressure"]
            rel_pressure_sum[away] += pressure_info["away_relegation_pressure"]
            title_pressure_sum[home] += pressure_info["home_title_pressure"]
            title_pressure_sum[away] += pressure_info["away_title_pressure"]
            pressure_sample_count[home] += 1
            pressure_sample_count[away] += 1

            hg = np.random.poisson(lambda_home)
            ag = np.random.poisson(lambda_away)
            hg, ag = apply_pressure_draw_shift(hg, ag, pressure_info)

            simulated_match_count += 1
            if hg == ag:
                draw_count += 1

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

            played[home] = played.get(home, 0) + 1
            played[away] = played.get(away, 0) + 1

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
                f"  {prev_name}, ELO_LAMBDA_WEIGHT={elo_lambda_weight:.2f}: "
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
            "elo_lambda_weight_candidate": elo_lambda_weight,
            "prev_games_2024": prev_games_by_team.get(team),
            "effective_prev_weight": prev_weight_by_team.get(team),
            "use_pressure_effects": USE_PRESSURE_EFFECTS,
            "avg_relegation_pressure": (
                rel_pressure_sum[team] / pressure_sample_count[team]
                if pressure_sample_count[team] > 0 else 0.0
            ),
            "avg_title_pressure": (
                title_pressure_sum[team] / pressure_sample_count[team]
                if pressure_sample_count[team] > 0 else 0.0
            ),
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
        "elo_lambda_weight": elo_lambda_weight,
        "n_sim": n_sim,
        "pressure_setting_name": globals().get("PRESSURE_SETTING_NAME", "manual"),
        "use_pressure_effects": USE_PRESSURE_EFFECTS,
        "late_season_start_ratio": LATE_SEASON_START_RATIO,
        "relegation_points_window": RELEGATION_POINTS_WINDOW,
        "title_points_window": TITLE_POINTS_WINDOW,
        "relegation_attack_lambda_boost": RELEGATION_ATTACK_LAMBDA_BOOST,
        "relegation_defense_lambda_boost": RELEGATION_DEFENSE_LAMBDA_BOOST,
        "title_attack_lambda_debuff": TITLE_ATTACK_LAMBDA_DEBUFF,
        "draw_shift_relegation": DRAW_SHIFT_RELEGATION,
        "draw_shift_title": DRAW_SHIFT_TITLE,
        "max_draw_shift": MAX_DRAW_SHIFT,
        "sim_draw_rate": (draw_count / simulated_match_count if simulated_match_count > 0 else np.nan),
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
        "elo_lambda_weight_candidate", "effective_prev_weight",
        "avg_relegation_pressure", "avg_title_pressure"
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
        "prev_setting_name": "前年設定名",
        "prev_weight_candidate": "候補PREV_WEIGHT",
        "prev_decay_candidate": "候補PREV_DECAY",
        "elo_lambda_weight_candidate": "候補ELO_LAMBDA_WEIGHT",
        "prev_games_2024": "2024J1試合数",
        "effective_prev_weight": "実効前年重み",
        "use_pressure_effects": "終盤補正",
        "avg_relegation_pressure": "平均残留圧力",
        "avg_title_pressure": "平均優勝圧力",
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
      overflow-x: auto;
      background: white;
      padding: 16px;
      border-radius: 12px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.06);
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
      v1.4をベースに、PREV=0.60/0.985、Elo補正重み=0.10を固定し、
      残留ブースト・優勝デバフを追加した結果です。
      大勝補正={GOAL_ADJUST_MODE}, cap={GOAL_CAP_FOR_STRENGTH},
      2025枠内重み={SOT_WEIGHT}, 2024総シュート重み={PREV_SHOT_WEIGHT},
      相性Effect={"ON" if USE_MATCHUP_EFFECT else "OFF"},
      終盤補正={"ON" if USE_PRESSURE_EFFECTS else "OFF"}。
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
    print("ELO_LAMBDA_WEIGHT_CANDIDATES:", ELO_LAMBDA_WEIGHT_CANDIDATES)
    print("USE_PRESSURE_EFFECTS:", USE_PRESSURE_EFFECTS)
    print("LATE_SEASON_START_RATIO:", LATE_SEASON_START_RATIO)
    print("RELEGATION_POINTS_WINDOW:", RELEGATION_POINTS_WINDOW)
    print("TITLE_POINTS_WINDOW:", TITLE_POINTS_WINDOW)
    print("RELEGATION_ATTACK_LAMBDA_BOOST:", RELEGATION_ATTACK_LAMBDA_BOOST)
    print("RELEGATION_DEFENSE_LAMBDA_BOOST:", RELEGATION_DEFENSE_LAMBDA_BOOST)
    print("TITLE_ATTACK_LAMBDA_DEBUFF:", TITLE_ATTACK_LAMBDA_DEBUFF)
    print("DRAW_SHIFT_RELEGATION:", DRAW_SHIFT_RELEGATION)
    print("DRAW_SHIFT_TITLE:", DRAW_SHIFT_TITLE)

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
    best_setting = None

    total = len(PRESSURE_SETTING_CANDIDATES)

    print("\n==============================")
    print("残留ブースト・優勝デバフ チューニング開始")
    print("==============================")

    for i, pressure_setting in enumerate(PRESSURE_SETTING_CANDIDATES, start=1):
        apply_pressure_setting(pressure_setting)

        print(
            f"\n[{i}/{total}] {PRESSURE_SETTING_NAME} | "
            f"USE={USE_PRESSURE_EFFECTS}, "
            f"REL=({RELEGATION_ATTACK_LAMBDA_BOOST:.3f}, {RELEGATION_DEFENSE_LAMBDA_BOOST:.3f}, draw={DRAW_SHIFT_RELEGATION:.3f}), "
            f"TITLE=({TITLE_ATTACK_LAMBDA_DEBUFF:.3f}, draw={DRAW_SHIFT_TITLE:.3f}), "
            f"late={LATE_SEASON_START_RATIO:.2f}, window=({RELEGATION_POINTS_WINDOW:.1f}, {TITLE_POINTS_WINDOW:.1f})"
        )

        summary, prediction_df = simulate_with_prev_and_elo(
            prev_setting=PREV_SETTING_FIXED,
            elo_lambda_weight=ELO_LAMBDA_WEIGHT_FIXED,
            n_sim=N_SIM_SEARCH,
            prepared=prepared,
            seed=RANDOM_SEED,
            verbose=False,
        )
        grid_rows.append(summary)

        print(
            f"  MAE={summary['mae']:.4f} / "
            f"引分率={summary['sim_draw_rate']:.3f} / "
            f"京都誤差={summary['kyoto_error']:.2f} / "
            f"柏誤差={summary['kashiwa_error']:.2f} / "
            f"川崎F誤差={summary['kawasaki_error']:.2f}"
        )

        if best_summary is None or summary["mae"] < best_summary["mae"]:
            best_summary = summary
            best_prediction = prediction_df
            best_setting = dict(pressure_setting)

    grid_df = pd.DataFrame(grid_rows)

    off_rows = grid_df[grid_df["pressure_setting_name"] == "pressure_off"]
    if len(off_rows) > 0:
        off_mae = float(off_rows.iloc[0]["mae"])
        grid_df["delta_vs_pressure_off"] = grid_df["mae"] - off_mae
    else:
        grid_df["delta_vs_pressure_off"] = np.nan

    grid_df = grid_df.sort_values("mae").reset_index(drop=True)
    grid_df.insert(0, "rank", grid_df.index + 1)
    grid_df.to_csv(OUTPUT_GRID_CSV, index=False, encoding="utf-8-sig")
    export_pressure_grid_html(
        grid_df,
        OUTPUT_GRID_HTML,
        title="J1 2025 終盤プレッシャー補正 チューニング結果",
    )

    print("\n==============================")
    print("探索結果 上位")
    print("==============================")
    show_cols = [
        "rank", "pressure_setting_name", "use_pressure_effects",
        "late_season_start_ratio", "relegation_points_window", "title_points_window",
        "relegation_attack_lambda_boost", "relegation_defense_lambda_boost",
        "title_attack_lambda_debuff", "draw_shift_relegation", "draw_shift_title",
        "sim_draw_rate", "mae", "delta_vs_pressure_off",
        "kyoto_error", "kashiwa_error", "kawasaki_error",
        "machida_error", "nagoya_error", "shimizu_error", "fctokyo_error",
    ]
    existing_show_cols = [col for col in show_cols if col in grid_df.columns]
    print(grid_df[existing_show_cols].head(20).to_string(index=False))
    print("\nGRID CSV:", OUTPUT_GRID_CSV)
    print("GRID HTML:", OUTPUT_GRID_HTML)

    if RUN_FINAL_BEST and best_summary is not None and best_setting is not None:
        apply_pressure_setting(best_setting)

        print("\n==============================")
        print("最良候補をN_SIM_FINALで再実行")
        print("==============================")
        print("best pressure setting:", PRESSURE_SETTING_NAME)
        print("PREV_SETTING:", PREV_SETTING_FIXED)
        print("ELO_LAMBDA_WEIGHT:", ELO_LAMBDA_WEIGHT_FIXED)
        print("RELEGATION_ATTACK_LAMBDA_BOOST:", RELEGATION_ATTACK_LAMBDA_BOOST)
        print("RELEGATION_DEFENSE_LAMBDA_BOOST:", RELEGATION_DEFENSE_LAMBDA_BOOST)
        print("TITLE_ATTACK_LAMBDA_DEBUFF:", TITLE_ATTACK_LAMBDA_DEBUFF)
        print("DRAW_SHIFT_RELEGATION:", DRAW_SHIFT_RELEGATION)
        print("DRAW_SHIFT_TITLE:", DRAW_SHIFT_TITLE)
        print("MAX_DRAW_SHIFT:", MAX_DRAW_SHIFT)

        final_summary, final_prediction = simulate_with_prev_and_elo(
            prev_setting=PREV_SETTING_FIXED,
            elo_lambda_weight=ELO_LAMBDA_WEIGHT_FIXED,
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
                "J1 2025 順位予測 v1.5 終盤プレッシャー補正 最良候補 "
                f"{PRESSURE_SETTING_NAME}"
            ),
        )

        print("\n==============================")
        print("最終確認")
        print("==============================")
        print("PRESSURE_SETTING:", PRESSURE_SETTING_NAME)
        print("MAE:", round(final_summary["mae"], 4))
        print("シミュレーション引分率:", round(final_summary["sim_draw_rate"], 4))
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
            "elo_lambda_weight_candidate", "effective_prev_weight",
            "avg_relegation_pressure", "avg_title_pressure",
            "champion_prob", "top3_prob", "bottom3_prob", "avg_points", "avg_gd", "elo",
        ]].to_string(index=False))


if __name__ == "__main__":
    main()
