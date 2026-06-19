import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path

# ============================================================
# J1 Current Decay 3年比較用 簡易版
# ------------------------------------------------------------
# 目的:
#   ・前年補正 PREV_WEIGHT=0.20 / PREV_DECAY=0.995 を固定する
#   ・Elo補正 ELO_LAMBDA_WEIGHT=0.20 を固定する
#   ・大勝補正 cap4 を固定する
#   ・相性Effect COMPAT_WEIGHT=0.20 を固定する
#   ・Draw Factor=1.20 / MAX_MATCH_DRAW_PROB=0.33 を固定する
#   ・現年前半戦成績にかけるCURRENT_DECAYを3年版で比較する
#   ・検証年: 2022→2023, 2023→2024, 2024→2025
#   ・1993-2025のJ1履歴CSVから、対象年と前年を抽出する
#   ・λには得点のみを使用し、枠内シュート数・シュート数は使わない
#   ・優勝ブースト、残留ブーストは使わない
#
# 検証方法:
#   1. target_yearの前半戦を学習データにする
#   2. target_yearの後半戦をシミュレーション対象にする
#   3. previous_yearの全試合から前年攻守係数を作る
#   4. 攻守係数計算時だけ大勝補正cap4を使う
#   5. 現年前半の攻守係数と前年攻守係数をPREV_WEIGHTで混合する
#   6. 現年前半の攻守係数をCURRENT_DECAYで重み付けする
#      decay<1.0なら、前半戦の中でも新しい試合をやや重視する
#   7. previous_year + target_year前半から固定Eloを作る
#   8. cutoff_date以前の全J1履歴から相性Effectを作る
#   9. Draw Factorと引き分け確率上限を適用する
#  10. target_year最終順位に対するMAEを計算する
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

# 2022→2023, 2023→2024, 2024→2025 を検証する
TARGET_YEARS = [2023, 2024, 2025]

# 固定する前年補正
FIXED_PREV_SETTING = {
    "name": "prev020_decay0995",
    "prev_weight": 0.20,
    "prev_decay": 0.995,
}

# 固定するElo補正
ELO_LAMBDA_WEIGHT_FIXED = 0.20

# 大勝補正は前回の検証結果に基づきcap4で固定する
GOAL_ADJUST_FIXED = {"name": "cap4", "goal_adjust_mode": "cap", "goal_cap_for_strength": 4}

# 相性Effectは前回の検証結果に基づき0.20で固定する
COMPAT_WEIGHT_FIXED = 0.20

# Draw Factor は前回検証で採用した 1.20 に固定する。
# 1.00 が通常の独立ポアソン。
# 1.20 は 0-0, 1-1, 2-2 などの引き分けスコアを増やす補正。
DRAW_FACTOR_FIXED = 1.20

# Draw Factor適用後の「1試合ごとの引き分け確率」上限。
# 前回検証で cap0.33 を採用。
MAX_MATCH_DRAW_PROB_FIXED = 0.33

# 現年前半戦のDecay候補。
# 1.000 = 前半戦の全試合を同じ重みで使う。
# 0.995以下 = 前半戦の中でも新しい試合をやや重視する。
CURRENT_DECAY_CANDIDATES = [
    1.000,
    0.995,
    0.990,
    0.985,
    0.980,
    0.970,
]

# Shrinkageは前回検証で不採用なので、縮小なしで固定する。
SHRINKAGE_FIXED = 1.00

# スコア確率行列を作るときの最大得点。
# LAMBDA_CAP=3.5なら10点までで尾部確率はかなり小さい。
MAX_GOALS_FOR_SCORE_GRID = 10

# 相性Effect設定
MATCHUP_PRIOR_N = 30
MATCHUP_TIME_DECAY = 0.97
COMPAT_FACTOR_MIN = 0.95
COMPAT_FACTOR_MAX = 1.05

# True: λの基準となるリーグ平均得点はraw得点を使う。
#       capは攻守係数の比率計算にだけ使う。
USE_RAW_LEAGUE_AVG_FOR_LAMBDA = True

# 現年前半戦のDecayはCURRENT_DECAY_CANDIDATESで比較する

# 昇格組補正: 前年J1にいないチームは前年由来の中立値1.0を混ぜない
USE_PROMOTED_PREV_ZERO = True
PROMOTED_PREV_WEIGHT = 0.0

# Elo設定
INITIAL_ELO = 1500
K_FACTOR = 16
HOME_ADV = 0

# Eloをλに反映するときのスケール
ELO_FACTOR_SCALE = 4000
ELO_FACTOR_MIN = 0.90
ELO_FACTOR_MAX = 1.10

# シミュレーション設定
N_SIM_SEARCH = 3000
RANDOM_SEED = 42
LAMBDA_CAP = 3.5

# 出力
OUTPUT_DETAIL_CSV = BASE_DIR / "decay_simple_multiyear_detail.csv"
OUTPUT_SUMMARY_CSV = BASE_DIR / "decay_simple_multiyear_summary.csv"
OUTPUT_SUMMARY_HTML = BASE_DIR / "decay_simple_multiyear_summary.html"
OUTPUT_BEST_PREDICTION_CSV = BASE_DIR / "decay_simple_multiyear_best_predictions.csv"


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
            -x[1]["gf"],
        )
    )


# =========================
# 5. 攻守係数 簡易版: 得点のみ + 大勝補正
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
    """
    攻守係数計算用の得点列を追加する。

    raw:
      生得点をそのまま使用。
    cap:
      攻守係数計算用の得点だけ上限をかける。
      実順位・初期勝点・シミュレーション得点には生得点を使う。
    """
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
    """
    得点のみでH/A別の攻撃係数・守備係数を作る。
    シュート数・枠内シュート数は一切使わない。

    大勝補正は攻守係数計算用の得点だけに適用する。
    リーグ平均得点はrawの得点を使い、λの基準値は実得点水準に保つ。
    """
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
        # ホーム試合
        home_games = history_df[history_df["home"] == team].copy()
        home_games = home_games.sort_values("date", ascending=False)

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

        # アウェイ試合
        away_games = history_df[history_df["away"] == team].copy()
        away_games = away_games.sort_values("date", ascending=False)

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

        blended[team] = {
            "home_attack": safe_blend(current_strengths[team]["home_attack"], prev["home_attack"], effective_prev_weight),
            "home_defense": safe_blend(current_strengths[team]["home_defense"], prev["home_defense"], effective_prev_weight),
            "away_attack": safe_blend(current_strengths[team]["away_attack"], prev["away_attack"], effective_prev_weight),
            "away_defense": safe_blend(current_strengths[team]["away_defense"], prev["away_defense"], effective_prev_weight),
        }

    return blended, prev_weight_by_team, prev_games_by_team


def apply_strength_shrinkage(strengths, shrinkage=1.0):
    """
    攻守係数を1.0方向に戻す。

    shrinkage=1.00: そのまま
    shrinkage=0.90: 1.0からの差を90%だけ残す

    例:
      1.40 -> 1.0 + (1.40 - 1.0) * 0.90 = 1.36
      0.70 -> 1.0 + (0.70 - 1.0) * 0.90 = 0.73
    """
    shrinkage = float(shrinkage)
    shrinkage = float(np.clip(shrinkage, 0.0, 1.5))

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
    """
    前年全試合 + 対象年前半戦だけからEloを作る。
    後半戦シミュレーション中はEloを更新せず、固定値としてλ補正に使う。
    """
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
    """
    cutoff_date以前のJ1履歴から、対戦カードごとの相性Effectを作る。

    考え方:
      ・各試合について、その時点のElo期待勝点と実勝点の差をresidualとして取る。
      ・home→away方向、away→home方向にそれぞれresidualを保存する。
      ・古い試合ほど time_decay ** years_ago で弱める。
      ・試合数が少ないカードは prior_n で0方向に縮小する。

    注意:
      ・target_year後半の結果は使わない。
      ・target_year前半までは、その時点で既知の情報として使う。
    """
    df = historical_df[historical_df["date"] <= cutoff_date].copy()
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
# 8. 期待得点・シミュレーション
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
    """0〜max_goalsまでのポアソン確率を計算する。"""
    probs = np.zeros(max_goals + 1, dtype=float)
    probs[0] = np.exp(-lam)
    for k in range(1, max_goals + 1):
        probs[k] = probs[k - 1] * lam / k
    return probs


def sample_score_with_draw_factor(
    lambda_home,
    lambda_away,
    draw_factor,
    max_goals=MAX_GOALS_FOR_SCORE_GRID,
    max_match_draw_prob=None,
):
    """
    独立ポアソンのスコア確率行列を作り、引き分けスコアだけdraw_factor倍する。
    ただし、max_match_draw_prob が指定されている場合は、補正後の
    1試合ごとの引き分け確率がその上限を超えないように調整する。

    draw_factor=1.00なら通常の独立ポアソン。
    draw_factor>1.00なら引き分けスコアが増える。

    注意: max_goalsを超える尾部は切り捨てて再正規化する。
    LAMBDA_CAP=3.5かつmax_goals=10なら影響は小さい。
    """
    home_probs = poisson_pmf_array(lambda_home, max_goals)
    away_probs = poisson_pmf_array(lambda_away, max_goals)
    score_matrix = np.outer(home_probs, away_probs)

    total_prob = score_matrix.sum()
    if not np.isfinite(total_prob) or total_prob <= 0:
        return int(np.random.poisson(lambda_home)), int(np.random.poisson(lambda_away))

    # まず尾部切り捨て後の確率行列として正規化する。
    score_matrix = score_matrix / total_prob

    diag_idx = np.arange(max_goals + 1)
    draw_mask = np.zeros_like(score_matrix, dtype=bool)
    draw_mask[diag_idx, diag_idx] = True

    base_draw_prob = float(score_matrix[draw_mask].sum())
    base_non_draw_prob = 1.0 - base_draw_prob

    # 引き分け確率がゼロまたは全て引き分けなら、安全に通常サンプリングする。
    if base_draw_prob <= 0 or base_non_draw_prob <= 0:
        flat_probs = score_matrix.ravel()
        sampled_idx = np.random.choice(flat_probs.size, p=flat_probs)
        home_goal, away_goal = np.unravel_index(sampled_idx, score_matrix.shape)
        return int(home_goal), int(away_goal)

    # Draw Factor適用後に目標とする引き分け確率。
    target_draw_prob = base_draw_prob * draw_factor

    # 1試合ごとの引き分け確率上限を適用する。
    if max_match_draw_prob is not None:
        target_draw_prob = min(target_draw_prob, float(max_match_draw_prob))

    # 数値的な安全策。
    target_draw_prob = float(np.clip(target_draw_prob, 0.0, 0.95))
    target_non_draw_prob = 1.0 - target_draw_prob

    draw_scale = target_draw_prob / base_draw_prob
    non_draw_scale = target_non_draw_prob / base_non_draw_prob

    score_matrix[draw_mask] *= draw_scale
    score_matrix[~draw_mask] *= non_draw_scale

    # 丸め誤差対策で再正規化。
    score_matrix = score_matrix / score_matrix.sum()

    flat_probs = score_matrix.ravel()
    sampled_idx = np.random.choice(flat_probs.size, p=flat_probs)
    home_goal, away_goal = np.unravel_index(sampled_idx, score_matrix.shape)
    return int(home_goal), int(away_goal)


def simulate_target_year(
    historical_df,
    target_year,
    prev_weight,
    prev_decay,
    elo_lambda_weight,
    goal_adjust_setting,
    compat_weight,
    draw_factor,
    max_match_draw_prob=None,
    current_decay=1.0,
    shrinkage=1.0,
    n_sim=N_SIM_SEARCH,
    seed=42,
):
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

    split = int(len(target_df) * 0.5)
    train_df = target_df.iloc[:split].copy()
    test_df = target_df.iloc[split:].copy()

    # 実順位: 対象年の全試合で計算
    actual_table = calculate_table(target_df, teams)
    actual_ranking = make_ranking(actual_table)
    actual_position = {
        team: pos + 1
        for pos, (team, stats) in enumerate(actual_ranking)
    }

    goal_adjust_name = goal_adjust_setting["name"]
    goal_adjust_mode = goal_adjust_setting["goal_adjust_mode"]
    goal_cap_for_strength = goal_adjust_setting["goal_cap_for_strength"]

    # 現年前半: 得点のみ、CURRENT_DECAY候補、大勝補正固定
    current_strengths, home_avg_goals, away_avg_goals = calculate_strengths_home_away_goals_only(
        train_df,
        teams=teams,
        decay=current_decay,
        goal_adjust_mode=goal_adjust_mode,
        goal_cap_for_strength=goal_cap_for_strength,
    )

    # 前年: 得点のみ、prev_decay候補、大勝補正候補
    previous_strengths, _, _ = calculate_strengths_home_away_goals_only(
        previous_df,
        teams=teams,
        decay=prev_decay,
        goal_adjust_mode=goal_adjust_mode,
        goal_cap_for_strength=goal_cap_for_strength,
    )

    strengths, prev_weight_by_team, prev_games_by_team = blend_with_previous_strengths(
        current_strengths=current_strengths,
        previous_strengths=previous_strengths,
        prev_weight=prev_weight,
        previous_df=previous_df,
        use_promoted_prev_zero=USE_PROMOTED_PREV_ZERO,
        promoted_prev_weight=PROMOTED_PREV_WEIGHT,
    )

    # Current Decayは今回比較しない。前回検証で不採用のため1.00固定。
    strengths = apply_strength_shrinkage(strengths, shrinkage=shrinkage)

    elo_ratings = build_elo_ratings(
        previous_df=previous_df,
        train_df=train_df,
        teams=teams,
        k_factor=K_FACTOR,
        home_adv=HOME_ADV,
    )

    cutoff_date = train_df["date"].max()
    matchup_effects = None
    if compat_weight > 0:
        matchup_effects = build_matchup_effects_j1(
            historical_df=historical_df,
            cutoff_date=cutoff_date,
            target_year=target_year,
            prior_n=MATCHUP_PRIOR_N,
            time_decay=MATCHUP_TIME_DECAY,
            k_factor=K_FACTOR,
            home_adv=HOME_ADV,
        )

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
                compat_weight=compat_weight,
            )

            hg, ag = sample_score_with_draw_factor(
                lambda_home=lambda_home,
                lambda_away=lambda_away,
                draw_factor=draw_factor,
                max_goals=MAX_GOALS_FOR_SCORE_GRID,
                max_match_draw_prob=max_match_draw_prob,
            )

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

        ranking = make_ranking(table)

        for pos, (team, stats) in enumerate(ranking):
            position_counts[team][pos + 1] += 1

        for team in teams:
            points_sum[team] += table[team]["points"]
            gf_sum[team] += table[team]["gf"]
            ga_sum[team] += table[team]["ga"]
            gd_sum[team] += table[team]["gd"]

    max_match_draw_prob_label = (
        "none" if max_match_draw_prob is None else f"cap{int(round(float(max_match_draw_prob) * 100)):03d}"
    )

    rows = []
    n_teams = len(teams)

    for team in teams:
        avg_pred_pos = sum(
            pos * (position_counts[team][pos] / n_sim)
            for pos in range(1, n_teams + 1)
        )
        actual_pos = actual_position[team]

        rows.append({
            "target_year": target_year,
            "previous_year": previous_year,
            "team": team,
            "actual_position": actual_pos,
            "avg_pred_position": avg_pred_pos,
            "position_error": abs(avg_pred_pos - actual_pos),
            "prob_actual_position": position_counts[team][actual_pos] / n_sim,
            "prev_weight": prev_weight,
            "prev_decay": prev_decay,
            "elo_lambda_weight": elo_lambda_weight,
            "compat_weight": compat_weight,
            "draw_factor": draw_factor,
            "max_match_draw_prob_label": max_match_draw_prob_label,
            "max_match_draw_prob": np.nan if max_match_draw_prob is None else float(max_match_draw_prob),
            "max_goals_for_score_grid": MAX_GOALS_FOR_SCORE_GRID,
            "matchup_prior_n": MATCHUP_PRIOR_N,
            "matchup_time_decay": MATCHUP_TIME_DECAY,
            "compat_factor_min": COMPAT_FACTOR_MIN,
            "compat_factor_max": COMPAT_FACTOR_MAX,
            "goal_adjust_name": goal_adjust_name,
            "goal_adjust_mode": goal_adjust_mode,
            "goal_cap_for_strength": goal_cap_for_strength,
            "elo": elo_ratings.get(team, INITIAL_ELO),
            "prev_games": prev_games_by_team.get(team, 0),
            "effective_prev_weight": prev_weight_by_team.get(team, prev_weight),
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

    summary = {
        "target_year": target_year,
        "previous_year": previous_year,
        "prev_weight": prev_weight,
        "prev_decay": prev_decay,
        "elo_lambda_weight": elo_lambda_weight,
        "compat_weight": compat_weight,
        "current_decay": float(current_decay),
        "shrinkage": float(shrinkage),
        "draw_factor": draw_factor,
        "max_match_draw_prob_label": max_match_draw_prob_label,
        "max_match_draw_prob": np.nan if max_match_draw_prob is None else float(max_match_draw_prob),
        "matchup_prior_n": MATCHUP_PRIOR_N,
        "matchup_time_decay": MATCHUP_TIME_DECAY,
        "compat_factor_min": COMPAT_FACTOR_MIN,
        "compat_factor_max": COMPAT_FACTOR_MAX,
        "goal_adjust_name": goal_adjust_name,
        "goal_adjust_mode": goal_adjust_mode,
        "goal_cap_for_strength": goal_cap_for_strength,
        "n_sim": n_sim,
        "n_teams": n_teams,
        "n_previous_matches": len(previous_df),
        "n_target_matches": len(target_df),
        "n_train_matches": len(train_df),
        "n_test_matches": len(test_df),
        "mae": mae,
        "mean_prob_actual_position": mean_prob_actual_position,
        "sim_draw_rate": sim_draw_rate,
    }

    lookup = prediction_df.set_index("team")
    for team in ["京都", "柏", "川崎F", "神戸", "横浜FM", "名古屋", "清水", "町田", "G大阪", "浦和"]:
        col_name = standardize_team_name(team)
        if col_name in lookup.index:
            summary[f"{col_name}_actual_position"] = lookup.loc[col_name, "actual_position"]
            summary[f"{col_name}_avg_pred_position"] = lookup.loc[col_name, "avg_pred_position"]
            summary[f"{col_name}_error"] = lookup.loc[col_name, "position_error"]

    return summary, prediction_df


# =========================
# 9. 出力
# =========================


def export_summary_html(df, output_path):
    display_df = df.copy()

    round_cols = [
        "prev_weight", "prev_decay", "elo_lambda_weight", "compat_weight", "current_decay", "shrinkage", "draw_factor",
        "max_match_draw_prob", "matchup_time_decay", "mean_mae", "std_mae",
        "mean_prob_actual_position", "mean_sim_draw_rate",
        "mae_2023", "mae_2024", "mae_2025",
        "delta_vs_decay100",
    ]
    for col in round_cols:
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(display_df[col], errors="coerce").round(4)

    table_html = display_df.to_html(index=False, classes="prediction-table")

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>Current Decay 3年比較</title>
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
  <h1>Current Decay 3年比較</h1>
  <div class="note">
    <p>
      2022→2023、2023→2024、2024→2025を対象に、PREV_WEIGHT=0.20 / PREV_DECAY=0.995、
      ELO_LAMBDA_WEIGHT=0.20、大勝補正cap4、COMPAT_WEIGHT=0.20、DRAW_FACTOR=1.20を固定し、現年前半戦のDecayを比較しています。
    </p>
    <p>
      λは得点のみで作成し、シュート数・枠内シュート数・優勝ブースト・残留ブーストは使っていません。
      N_SIM_SEARCH={N_SIM_SEARCH}, LAMBDA_CAP={LAMBDA_CAP}, SHRINKAGE_FIXED={SHRINKAGE_FIXED},
      MATCHUP_PRIOR_N={MATCHUP_PRIOR_N}, MATCHUP_TIME_DECAY={MATCHUP_TIME_DECAY},
      MAX_GOALS_FOR_SCORE_GRID={MAX_GOALS_FOR_SCORE_GRID}, DRAW_FACTOR={DRAW_FACTOR_FIXED}, MAX_MATCH_DRAW_PROB={MAX_MATCH_DRAW_PROB_FIXED}。
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
# 10. main
# =========================


def main():
    historical_df = load_historical_j1_csv(HISTORICAL_CSV)

    prev_name = str(FIXED_PREV_SETTING["name"])
    prev_weight = float(FIXED_PREV_SETTING["prev_weight"])
    prev_decay = float(FIXED_PREV_SETTING["prev_decay"])
    elo_lambda_weight = float(ELO_LAMBDA_WEIGHT_FIXED)
    goal_adjust_setting = GOAL_ADJUST_FIXED
    compat_weight = float(COMPAT_WEIGHT_FIXED)

    print("\n==============================")
    print("Current Decay 3年比較 簡易版")
    print("==============================")
    print("HISTORICAL_CSV:", HISTORICAL_CSV)
    print("TARGET_YEARS:", TARGET_YEARS)
    print("FIXED_PREV_SETTING:", FIXED_PREV_SETTING)
    print("ELO_LAMBDA_WEIGHT_FIXED:", ELO_LAMBDA_WEIGHT_FIXED)
    print("GOAL_ADJUST_FIXED:", GOAL_ADJUST_FIXED)
    print("COMPAT_WEIGHT_FIXED:", COMPAT_WEIGHT_FIXED)
    print("DRAW_FACTOR_FIXED:", DRAW_FACTOR_FIXED)
    print("MAX_MATCH_DRAW_PROB_FIXED:", MAX_MATCH_DRAW_PROB_FIXED)
    print("CURRENT_DECAY_CANDIDATES:", CURRENT_DECAY_CANDIDATES)
    print("SHRINKAGE_FIXED:", SHRINKAGE_FIXED)
    print("MAX_GOALS_FOR_SCORE_GRID:", MAX_GOALS_FOR_SCORE_GRID)
    print("MATCHUP_PRIOR_N:", MATCHUP_PRIOR_N)
    print("MATCHUP_TIME_DECAY:", MATCHUP_TIME_DECAY)
    print("COMPAT_FACTOR_MIN/MAX:", COMPAT_FACTOR_MIN, COMPAT_FACTOR_MAX)
    print("USE_RAW_LEAGUE_AVG_FOR_LAMBDA:", USE_RAW_LEAGUE_AVG_FOR_LAMBDA)
    print("N_SIM_SEARCH:", N_SIM_SEARCH)
    print("LAMBDA_CAP:", LAMBDA_CAP)
    print("K_FACTOR:", K_FACTOR)
    print("HOME_ADV:", HOME_ADV)
    print("ELO_FACTOR_SCALE:", ELO_FACTOR_SCALE)
    print("ELO_FACTOR_MIN/MAX:", ELO_FACTOR_MIN, ELO_FACTOR_MAX)
    print("USE_PROMOTED_PREV_ZERO:", USE_PROMOTED_PREV_ZERO)
    print("PROMOTED_PREV_WEIGHT:", PROMOTED_PREV_WEIGHT)

    detail_rows = []
    all_predictions = []

    total = len(CURRENT_DECAY_CANDIDATES) * len(TARGET_YEARS)
    done = 0

    draw_factor = float(DRAW_FACTOR_FIXED)
    max_match_draw_prob = MAX_MATCH_DRAW_PROB_FIXED
    cap_label = "none" if max_match_draw_prob is None else f"cap{int(round(float(max_match_draw_prob) * 100)):03d}"

    shrinkage = float(SHRINKAGE_FIXED)

    for current_decay in CURRENT_DECAY_CANDIDATES:
        for target_year in TARGET_YEARS:
            done += 1
            print(
                f"\n[{done}/{total}] "
                f"{prev_name}, target={target_year}, "
                f"prev_weight={prev_weight:.2f}, prev_decay={prev_decay:.3f}, "
                f"ELO_LAMBDA_WEIGHT={elo_lambda_weight:.2f}, "
                f"GOAL_ADJUST={goal_adjust_setting['name']}, "
                f"COMPAT_WEIGHT={compat_weight:.2f}, "
                f"DRAW_FACTOR={draw_factor:.2f}, "
                f"MAX_MATCH_DRAW_PROB={cap_label}, "
                f"CURRENT_DECAY={current_decay:.3f}, "
                f"SHRINKAGE_FIXED={shrinkage:.2f}"
            )

            cap_seed = 0 if max_match_draw_prob is None else int(round(float(max_match_draw_prob) * 10000))
            seed_offset = (
                int(round(draw_factor * 1000))
                + cap_seed
                + int(round(float(current_decay) * 100000))
                + int(round(float(shrinkage) * 1000))
            )

            summary, prediction_df = simulate_target_year(
                historical_df=historical_df,
                target_year=target_year,
                prev_weight=prev_weight,
                prev_decay=prev_decay,
                elo_lambda_weight=elo_lambda_weight,
                goal_adjust_setting=goal_adjust_setting,
                compat_weight=compat_weight,
                draw_factor=draw_factor,
                max_match_draw_prob=max_match_draw_prob,
                current_decay=current_decay,
                shrinkage=shrinkage,
                n_sim=N_SIM_SEARCH,
                seed=RANDOM_SEED + target_year + seed_offset,
            )
            summary["prev_setting_name"] = prev_name
            detail_rows.append(summary)

            prediction_df["prev_setting_name"] = prev_name
            prediction_df["prev_weight"] = prev_weight
            prediction_df["prev_decay"] = prev_decay
            prediction_df["elo_lambda_weight"] = elo_lambda_weight
            prediction_df["goal_adjust_name"] = goal_adjust_setting["name"]
            prediction_df["compat_weight"] = compat_weight
            prediction_df["current_decay"] = float(current_decay)
            prediction_df["draw_factor"] = draw_factor
            prediction_df["max_match_draw_prob_label"] = cap_label
            prediction_df["max_match_draw_prob"] = np.nan if max_match_draw_prob is None else float(max_match_draw_prob)
            prediction_df["shrinkage"] = float(shrinkage)
            all_predictions.append(prediction_df)

            print(
                f"  MAE={summary['mae']:.4f}, "
                f"実順位確率平均={summary['mean_prob_actual_position']:.4f}, "
                f"引分率={summary['sim_draw_rate']:.4f}"
            )

    detail_df = pd.DataFrame(detail_rows)
    detail_df.to_csv(OUTPUT_DETAIL_CSV, index=False, encoding="utf-8-sig")

    # 年別MAEを横持ちにする
    pivot_mae = detail_df.pivot_table(
        index=[
            "prev_setting_name", "prev_weight", "prev_decay", "elo_lambda_weight",
            "goal_adjust_name", "goal_adjust_mode", "goal_cap_for_strength",
            "compat_weight", "current_decay", "shrinkage", "draw_factor", "max_match_draw_prob_label", "max_goals_for_score_grid",
            "matchup_prior_n", "matchup_time_decay",
            "compat_factor_min", "compat_factor_max",
        ],
        columns="target_year",
        values="mae",
        aggfunc="mean",
    )
    pivot_mae.columns = [f"mae_{int(col)}" for col in pivot_mae.columns]
    pivot_mae = pivot_mae.reset_index()

    grouped = detail_df.groupby(
        [
            "prev_setting_name", "prev_weight", "prev_decay", "elo_lambda_weight",
            "goal_adjust_name", "goal_adjust_mode", "goal_cap_for_strength",
            "compat_weight", "current_decay", "shrinkage", "draw_factor", "max_match_draw_prob_label", "max_goals_for_score_grid",
            "matchup_prior_n", "matchup_time_decay",
            "compat_factor_min", "compat_factor_max",
        ],
        as_index=False,
    ).agg(
        max_match_draw_prob=("max_match_draw_prob", "first"),
        mean_mae=("mae", "mean"),
        std_mae=("mae", "std"),
        mean_prob_actual_position=("mean_prob_actual_position", "mean"),
        mean_sim_draw_rate=("sim_draw_rate", "mean"),
    )

    summary_df = grouped.merge(
        pivot_mae,
        on=[
            "prev_setting_name", "prev_weight", "prev_decay", "elo_lambda_weight",
            "goal_adjust_name", "goal_adjust_mode", "goal_cap_for_strength",
            "compat_weight", "current_decay", "shrinkage", "draw_factor", "max_match_draw_prob_label", "max_goals_for_score_grid",
            "matchup_prior_n", "matchup_time_decay",
            "compat_factor_min", "compat_factor_max",
        ],
        how="left",
    )

    # 比較基準: current_decay=1.000、つまり現年前半戦の減衰なし
    base_rows = summary_df[np.isclose(summary_df["current_decay"].astype(float), 1.000)]
    base_decay100 = float(base_rows.iloc[0]["mean_mae"]) if len(base_rows) > 0 else np.nan
    summary_df["delta_vs_decay100"] = summary_df["mean_mae"] - base_decay100

    summary_df = summary_df.sort_values(["mean_mae", "std_mae"]).reset_index(drop=True)
    summary_df.insert(0, "rank", summary_df.index + 1)
    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    export_summary_html(summary_df, OUTPUT_SUMMARY_HTML)

    best_current_decay = float(summary_df.iloc[0]["current_decay"])
    best_shrinkage = float(summary_df.iloc[0]["shrinkage"])
    best_draw_factor = float(summary_df.iloc[0]["draw_factor"])
    best_cap_label = str(summary_df.iloc[0]["max_match_draw_prob_label"])

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    best_predictions_df = predictions_df[
        np.isclose(predictions_df["draw_factor"], best_draw_factor)
        & (predictions_df["max_match_draw_prob_label"] == best_cap_label)
        & np.isclose(predictions_df["current_decay"], best_current_decay)
        & np.isclose(predictions_df["shrinkage"], best_shrinkage)
    ].copy()
    best_predictions_df.to_csv(OUTPUT_BEST_PREDICTION_CSV, index=False, encoding="utf-8-sig")

    print("\n==============================")
    print("検証結果")
    print("==============================")
    show_cols = [
        "rank", "prev_setting_name", "prev_weight", "prev_decay", "elo_lambda_weight",
        "goal_adjust_name", "compat_weight", "current_decay", "shrinkage", "draw_factor", "max_match_draw_prob_label", "max_goals_for_score_grid",
        "matchup_prior_n", "matchup_time_decay",
        "mean_mae", "std_mae", "mean_prob_actual_position", "mean_sim_draw_rate",
        "mae_2023", "mae_2024", "mae_2025",
        "delta_vs_decay100",
    ]
    existing_show_cols = [col for col in show_cols if col in summary_df.columns]
    print(summary_df[existing_show_cols].to_string(index=False))

    print("\nDETAIL CSV:", OUTPUT_DETAIL_CSV)
    print("SUMMARY CSV:", OUTPUT_SUMMARY_CSV)
    print("SUMMARY HTML:", OUTPUT_SUMMARY_HTML)
    print("BEST PREDICTION CSV:", OUTPUT_BEST_PREDICTION_CSV)

    print("\n最良候補:")
    print("PREV_WEIGHT:", prev_weight)
    print("PREV_DECAY:", prev_decay)
    print("ELO_LAMBDA_WEIGHT:", elo_lambda_weight)
    print("GOAL_ADJUST:", goal_adjust_setting["name"])
    print("COMPAT_WEIGHT:", compat_weight)
    print("DRAW_FACTOR:", best_draw_factor)
    print("MAX_MATCH_DRAW_PROB:", best_cap_label)
    print("CURRENT_DECAY:", best_current_decay)
    print("SHRINKAGE:", best_shrinkage)
    print("mean_mae:", round(float(summary_df.iloc[0]["mean_mae"]), 4))


if __name__ == "__main__":
    main()
