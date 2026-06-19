import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path

# ============================================================
# J1 2025 順位予測 更新版
# ------------------------------------------------------------
# 採用モデル:
#   ・2025前半戦: 得点 + 枠内シュート
#   ・2024前年: 得点 + 総シュート
#   ・前年レーティング重み: 0.4
#   ・Elo補正: K=16, HOME_ADV=0, ELO_LAMBDA_WEIGHT=0.20
#
# 相性Effect:
#   まだ検証中のため、標準ではOFF。
#   使う場合は USE_MATCHUP_EFFECT = True にして、
#   j1_historical_results_1993_2025_table_fixed.csv を同じフォルダに置く。
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
HISTORICAL_J1_FILENAME = "j1_historical_results_1993_2025_table_fixed.csv"

OUTPUT_CSV = BASE_DIR / "j1_2025_prediction_current_model.csv"
OUTPUT_HTML = BASE_DIR / "j1_2025_prediction_current_model.html"

# 基本モデル
SOT_WEIGHT = 0.1
PREV_SHOT_WEIGHT = 0.1
PREV_WEIGHT = 0.4

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
# 5. 攻守係数
# =========================

def calculate_strengths_home_away(
    history_df,
    teams,
    decay=1.0,
    feature_weight=0.1,
    feature_type="sot"
):
    history_df = history_df.copy()

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

    home_avg_goals = history_df["home_goal"].mean()
    away_avg_goals = history_df["away_goal"].mean()

    if not np.isfinite(home_avg_goals) or home_avg_goals <= 0:
        home_avg_goals = 1.0
    if not np.isfinite(away_avg_goals) or away_avg_goals <= 0:
        away_avg_goals = 1.0

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

            home_gf += row_dict["home_goal"] * weight
            home_ga += row_dict["away_goal"] * weight

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
            home_goal_attack = (home_gf / home_w) / home_avg_goals
            home_goal_defense = (home_ga / home_w) / away_avg_goals

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

            away_gf += row_dict["away_goal"] * weight
            away_ga += row_dict["home_goal"] * weight

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
            away_goal_attack = (away_gf / away_w) / away_avg_goals
            away_goal_defense = (away_ga / away_w) / home_avg_goals

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


def blend_with_previous_strengths(current_strengths, previous_strengths, prev_weight):
    blended = {}

    def safe_blend(current_value, previous_value, prev_weight):
        current_ok = np.isfinite(current_value)
        previous_ok = np.isfinite(previous_value)

        if current_ok and previous_ok:
            return (1 - prev_weight) * current_value + prev_weight * previous_value
        if current_ok:
            return current_value
        if previous_ok:
            return previous_value
        return 1.0

    for team in current_strengths:
        if team in previous_strengths:
            blended[team] = {
                "home_attack": safe_blend(
                    current_strengths[team]["home_attack"],
                    previous_strengths[team]["home_attack"],
                    prev_weight
                ),
                "home_defense": safe_blend(
                    current_strengths[team]["home_defense"],
                    previous_strengths[team]["home_defense"],
                    prev_weight
                ),
                "away_attack": safe_blend(
                    current_strengths[team]["away_attack"],
                    previous_strengths[team]["away_attack"],
                    prev_weight
                ),
                "away_defense": safe_blend(
                    current_strengths[team]["away_defense"],
                    previous_strengths[team]["away_defense"],
                    prev_weight
                ),
            }
        else:
            blended[team] = current_strengths[team]

    return blended


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
# 7. 相性Effect
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
# 8. 期待得点
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
# 9. HTML出力
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
  <title>J1 2025 順位予測 | Current Model</title>
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
  <h1>J1 2025 順位予測</h1>
  <div class="note">
    <p>
      採用設定: 前年レーティング重み {PREV_WEIGHT} /
      2025枠内シュート重み {SOT_WEIGHT} /
      2024総シュート重み {PREV_SHOT_WEIGHT} /
      Elo補正 K={K_FACTOR}, HOME_ADV={HOME_ADV}, ELO_LAMBDA_WEIGHT={ELO_LAMBDA_WEIGHT}
    </p>
    <p>
      N_SIM={N_SIM}。相性Effect: {"ON" if USE_MATCHUP_EFFECT else "OFF"}。
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
# 10. メイン
# =========================

def main():
    if RANDOM_SEED is not None:
        np.random.seed(RANDOM_SEED)

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
    print("2025試合数:", len(current_df))
    print("train試合数:", len(train_df))
    print("test試合数:", len(test_df))
    print("チーム数:", len(teams))

    current_strengths, home_avg_goals, away_avg_goals = calculate_strengths_home_away(
        train_df,
        teams=teams,
        decay=DECAY,
        feature_weight=SOT_WEIGHT,
        feature_type="sot"
    )

    previous_strengths, _, _ = calculate_strengths_home_away(
        previous_df,
        teams=teams,
        decay=DECAY,
        feature_weight=PREV_SHOT_WEIGHT,
        feature_type="shots"
    )

    strengths = blend_with_previous_strengths(
        current_strengths,
        previous_strengths,
        PREV_WEIGHT
    )

    elo_ratings = build_elo_ratings(
        previous_df=previous_df,
        train_df=train_df,
        teams=teams,
        k_factor=K_FACTOR,
        home_adv=HOME_ADV
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
            home_adv=HOME_ADV
        )

        print("HISTORICAL_J1_CSV:", historical_path)
        print("相性Effect件数:", len(matchup_effects))

    actual_table = calculate_table(current_df, teams)
    actual_ranking = make_ranking(actual_table)

    actual_position = {
        team: pos + 1
        for pos, (team, stats) in enumerate(actual_ranking)
    }

    position_counts = {team: Counter() for team in teams}
    points_sum = {team: 0.0 for team in teams}
    gf_sum = {team: 0.0 for team in teams}
    ga_sum = {team: 0.0 for team in teams}
    gd_sum = {team: 0.0 for team in teams}

    for sim in range(1, N_SIM + 1):
        table = calculate_table(train_df, teams)

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
                elo_lambda_weight=ELO_LAMBDA_WEIGHT,
                matchup_effects=matchup_effects,
                compat_weight=COMPAT_WEIGHT
            )

            hg = np.random.poisson(lambda_home)
            ag = np.random.poisson(lambda_away)

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

        ranking = make_ranking(table)

        for pos, (team, stats) in enumerate(ranking):
            position_counts[team][pos + 1] += 1

        for team in teams:
            points_sum[team] += table[team]["points"]
            gf_sum[team] += table[team]["gf"]
            ga_sum[team] += table[team]["ga"]
            gd_sum[team] += table[team]["gd"]

        if sim % 500 == 0:
            print(f"{sim}/{N_SIM} 回終了")

    rows = []

    for team in teams:
        avg_pred_pos = sum(
            pos * (position_counts[team][pos] / N_SIM)
            for pos in range(1, len(teams) + 1)
        )

        rows.append({
            "team": team,
            "actual_position": actual_position[team],
            "avg_pred_position": avg_pred_pos,
            "position_error": abs(avg_pred_pos - actual_position[team]),
            "most_likely_position": position_counts[team].most_common(1)[0][0],
            "champion_prob": position_counts[team][1] / N_SIM,
            "top3_prob": sum(position_counts[team][p] for p in range(1, 4)) / N_SIM,
            "top5_prob": sum(position_counts[team][p] for p in range(1, 6)) / N_SIM,
            "bottom3_prob": sum(position_counts[team][p] for p in range(len(teams) - 2, len(teams) + 1)) / N_SIM,
            "avg_points": points_sum[team] / N_SIM,
            "avg_gf": gf_sum[team] / N_SIM,
            "avg_ga": ga_sum[team] / N_SIM,
            "avg_gd": gd_sum[team] / N_SIM,
            "elo": elo_ratings.get(team, INITIAL_ELO),
        })

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values("avg_pred_position").reset_index(drop=True)
    result_df.insert(0, "pred_rank", result_df.index + 1)

    mae = result_df["position_error"].mean()

    result_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    export_prediction_html(result_df, OUTPUT_HTML)

    print("\n==============================")
    print("更新完了")
    print("==============================")
    print("MAE:", round(mae, 4))
    print("CSV:", OUTPUT_CSV)
    print("HTML:", OUTPUT_HTML)
    print("\n予測順位表:")
    print(result_df[[
        "pred_rank",
        "team",
        "actual_position",
        "avg_pred_position",
        "position_error",
        "champion_prob",
        "top3_prob",
        "bottom3_prob",
        "avg_points",
        "avg_gd",
        "elo",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
