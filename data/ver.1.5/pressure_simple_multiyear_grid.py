import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path

# ============================================================
# J1 終盤補正 3年比較用 簡易版
# ------------------------------------------------------------
# 目的:
#   ・前年補正 PREV_WEIGHT=0.20 / PREV_DECAY=0.995 を固定する
#   ・Elo補正 ELO_LAMBDA_WEIGHT=0.20 を固定する
#   ・大勝補正 cap4 を固定する
#   ・相性Effect COMPAT_WEIGHT=0.20 を固定する
#   ・Draw Factor=1.20 / MAX_MATCH_DRAW_PROB=0.33 を固定する
#   ・LAMBDA_CAP=3.5を固定し、終盤補正を3年版で比較する
#   ・検証年: 2022→2023, 2023→2024, 2024→2025
#   ・1993-2025のJ1履歴CSVから、対象年と前年を抽出する
#   ・λには得点のみを使用し、枠内シュート数・シュート数は使わない
#   ・終盤補正候補を比較する
#
# 検証方法:
#   1. target_yearの前半戦を学習データにする
#   2. target_yearの後半戦をシミュレーション対象にする
#   3. previous_yearの全試合から前年攻守係数を作る
#   4. 攻守係数計算時だけ大勝補正cap4を使う
#   5. 現年前半の攻守係数と前年攻守係数をPREV_WEIGHTで混合する
#   6. Current Decayは1.000、Shrinkageは1.00で固定する
#   7. LAMBDA_CAP=3.5を固定する
#   8. previous_year + target_year前半から固定Eloを作る
#   9. cutoff_date以前の全J1履歴から相性Effectを作る
#  10. Draw Factorと引き分け確率上限を適用する
#  11. 優勝争いブーストと残留争いアンダードッグ引き分け補正を比較する
#  12. target_year最終順位に対するMAEを計算する
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

# Current Decay は前回検証で不採用なので、減衰なしで固定する。
CURRENT_DECAY_FIXED = 1.000

# Shrinkageは前回検証で不採用なので、縮小なしで固定する。
SHRINKAGE_FIXED = 1.00

# 昇格組補正は10000回検証で不採用にしたため、補正なしで固定する。
PROMOTED_PRIOR_FIXED = {
    "name": "none",
    "promoted_prev_weight": 0.00,
    "promoted_attack_prior": 1.00,
    "promoted_defense_prior": 1.00,
}

# 終盤補正候補。
# pressure_off: 終盤補正なし。
# title_*: 優勝争いチームの得点λを終盤だけ少し上げる。
# relegation_draw_weak: 残留争い中かつアンダードッグのチームが1点差で負けた場合だけ、低確率で引き分けに寄せる。
PRESSURE_SETTING_CANDIDATES = [
    {
        "name": "pressure_off",
        "use_pressure": False,
        "pressure_window_games": 8,
        "title_boost": 0.00,
        "title_points_window": 8.0,
        "title_rank_cutoff": 5,
        "relegation_draw_shift": 0.00,
        "relegation_points_window": 6.0,
        "relegation_rank_buffer": 3,
        "min_underdog_lambda_diff": 0.10,
        "underdog_lambda_diff_scale": 0.75,
        "max_draw_shift": 0.00,
    }
]

# 下位3クラブを残留争いの基準として扱う。
RELEGATION_SPOTS = 3

# スコア確率行列を作るときの最大得点。
# LAMBDA_CAP=3.5なら10点までで尾部確率はかなり小さい。
# None の場合も比較の公平性のため、スコアグリッドは10点までで固定する。
MAX_GOALS_FOR_SCORE_GRID = 10

# 相性Effect設定
MATCHUP_PRIOR_N = 30
MATCHUP_TIME_DECAY = 0.97
COMPAT_FACTOR_MIN = 0.95
COMPAT_FACTOR_MAX = 1.05

# True: λの基準となるリーグ平均得点はraw得点を使う。
#       capは攻守係数の比率計算にだけ使う。
USE_RAW_LEAGUE_AVG_FOR_LAMBDA = True

# 昇格組補正は不採用のため、前年J1にいないチームには前年補正を混ぜない。
USE_PROMOTED_PREV_ZERO = True

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
LAMBDA_CAP = 3.5  # 前回検証で採用したλ上限

# 出力
OUTPUT_DETAIL_CSV = BASE_DIR / "pressure_simple_multiyear_detail.csv"
OUTPUT_SUMMARY_CSV = BASE_DIR / "pressure_simple_multiyear_summary.csv"
OUTPUT_SUMMARY_HTML = BASE_DIR / "pressure_simple_multiyear_summary.html"
OUTPUT_BEST_PREDICTION_CSV = BASE_DIR / "pressure_simple_multiyear_best_predictions.csv"
OUTPUT_ALL_PREDICTIONS_CSV = BASE_DIR / "pressure_simple_multiyear_all_predictions.csv"

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

        # 前年J1にいない昇格組だけ、前年成分を「弱い事前値」に置き換える。
        # promoted_prev_weight=0.0なら、この事前値は混ざらないので従来通り。
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




def calculate_total_games_by_team(match_df, teams):
    """既知の日程から、各チームのシーズン総試合数を数える。"""
    return {team: count_team_games(match_df, team) for team in teams}


def calculate_games_played_by_team(match_df, teams):
    """指定時点までに各チームが消化した試合数を数える。"""
    return {team: count_team_games(match_df, team) for team in teams}


def late_season_factor_by_remaining(played, total_games, pressure_window_games):
    """
    残り試合数から終盤度を0〜1で返す。

    pressure_window_games=8なら、残り8試合から少しずつ効き始め、
    最終戦に近づくほど1に近づく。
    """
    total_games = max(int(total_games), 1)
    played = max(int(played), 0)
    pressure_window_games = max(int(pressure_window_games), 1)

    remaining = max(total_games - played, 0)
    if remaining > pressure_window_games:
        return 0.0

    # remaining=pressure_window_games で 1/window、remaining=1 で 1.0 に近い値。
    return float(np.clip((pressure_window_games - remaining + 1) / pressure_window_games, 0.0, 1.0))


def calculate_pressure_context(table, played, total_games_by_team, teams, pressure_setting):
    """
    試合直前の順位表だけを使って、優勝争い・残留争いの圧力を計算する。

    未来情報を避けるため、ここで使うのはシミュレーション中に更新される
    table / played / 既知の日程上の総試合数だけ。
    """
    if not pressure_setting.get("use_pressure", False):
        return {
            team: {
                "rank": np.nan,
                "points": table.get(team, {}).get("points", 0),
                "late": 0.0,
                "title_pressure": 0.0,
                "relegation_pressure": 0.0,
            }
            for team in teams
        }

    ranking = make_ranking(table)
    n_teams = len(ranking)
    rank_by_team = {team: pos + 1 for pos, (team, _) in enumerate(ranking)}
    points_by_team = {team: stats["points"] for team, stats in table.items()}

    top_points = ranking[0][1]["points"]

    # 下位3クラブが降格圏という前提で、残留ラインを「降格圏の1つ上」に置く。
    safety_rank = max(1, n_teams - RELEGATION_SPOTS)
    safety_index = min(max(safety_rank - 1, 0), n_teams - 1)
    safety_line_points = ranking[safety_index][1]["points"]

    pressure_window_games = int(pressure_setting.get("pressure_window_games", 8))
    title_points_window = float(pressure_setting.get("title_points_window", 8.0))
    title_rank_cutoff = int(pressure_setting.get("title_rank_cutoff", 5))
    relegation_points_window = float(pressure_setting.get("relegation_points_window", 6.0))
    relegation_rank_buffer = int(pressure_setting.get("relegation_rank_buffer", 3))

    context = {}

    for team in teams:
        team_points = points_by_team.get(team, 0)
        team_rank = rank_by_team.get(team, n_teams)
        team_played = played.get(team, 0)
        team_total_games = total_games_by_team.get(team, max(team_played, 1))
        late = late_season_factor_by_remaining(team_played, team_total_games, pressure_window_games)

        # 優勝圧力: 上位かつ首位との勝点差が近いほど高い。
        title_distance = max(0.0, float(top_points - team_points))
        title_pressure = max(0.0, 1.0 - title_distance / max(title_points_window, 1e-9))
        title_rank_gate = 1.0 if team_rank <= title_rank_cutoff else 0.0
        title_pressure *= title_rank_gate * late

        # 残留圧力: 残留ラインに近い下位チームほど高い。
        relegation_distance = abs(float(team_points - safety_line_points))
        relegation_pressure = max(0.0, 1.0 - relegation_distance / max(relegation_points_window, 1e-9))
        lower_rank_threshold = max(1, n_teams - RELEGATION_SPOTS - relegation_rank_buffer)
        lower_table_gate = 1.0 if team_rank >= lower_rank_threshold else 0.0
        relegation_pressure *= lower_table_gate * late

        context[team] = {
            "rank": team_rank,
            "points": team_points,
            "late": float(late),
            "title_pressure": float(np.clip(title_pressure, 0.0, 1.0)),
            "relegation_pressure": float(np.clip(relegation_pressure, 0.0, 1.0)),
        }

    return context


def apply_pressure_effects_to_lambdas(lambda_home, lambda_away, home, away, pressure_context, pressure_setting):
    """
    終盤補正のうち、優勝争いブーストだけをλに反映する。

    残留補正はλを直接上げず、アンダードッグの1点差負けを
    低確率で引き分けへ寄せる後処理として扱う。
    """
    if not pressure_setting.get("use_pressure", False):
        return safe_lambda(lambda_home), safe_lambda(lambda_away), {
            "home_title_pressure": 0.0,
            "away_title_pressure": 0.0,
            "home_relegation_pressure": 0.0,
            "away_relegation_pressure": 0.0,
            "home_rank": np.nan,
            "away_rank": np.nan,
        }

    home_ctx = pressure_context.get(home, {})
    away_ctx = pressure_context.get(away, {})

    home_title = float(home_ctx.get("title_pressure", 0.0))
    away_title = float(away_ctx.get("title_pressure", 0.0))
    home_relegation = float(home_ctx.get("relegation_pressure", 0.0))
    away_relegation = float(away_ctx.get("relegation_pressure", 0.0))

    title_boost = float(pressure_setting.get("title_boost", 0.0))

    lambda_home = safe_lambda(lambda_home * (1.0 + title_boost * home_title))
    lambda_away = safe_lambda(lambda_away * (1.0 + title_boost * away_title))

    pressure_info = {
        "home_title_pressure": home_title,
        "away_title_pressure": away_title,
        "home_relegation_pressure": home_relegation,
        "away_relegation_pressure": away_relegation,
        "home_rank": home_ctx.get("rank", np.nan),
        "away_rank": away_ctx.get("rank", np.nan),
    }

    return lambda_home, lambda_away, pressure_info


def apply_relegation_underdog_draw_shift(home_goal, away_goal, lambda_home, lambda_away, pressure_info, pressure_setting):
    """
    残留争い中かつアンダードッグのチームが1点差で負けた場合だけ、
    低確率で引き分けへ寄せる。

    例:
      homeが残留争い中・不利で 0-1 負け -> 一定確率で 1-1
      awayが残留争い中・不利で 2-1 負け -> 一定確率で 2-2
    """
    if not pressure_setting.get("use_pressure", False):
        return home_goal, away_goal

    if abs(home_goal - away_goal) != 1:
        return home_goal, away_goal

    base_shift = float(pressure_setting.get("relegation_draw_shift", 0.0))
    if base_shift <= 0:
        return home_goal, away_goal

    min_diff = float(pressure_setting.get("min_underdog_lambda_diff", 0.10))
    diff_scale = max(float(pressure_setting.get("underdog_lambda_diff_scale", 0.75)), 1e-9)
    max_draw_shift = float(pressure_setting.get("max_draw_shift", 0.07))

    home_relegation = float(pressure_info.get("home_relegation_pressure", 0.0))
    away_relegation = float(pressure_info.get("away_relegation_pressure", 0.0))

    # homeがアンダードッグで1点差負け
    if home_goal + 1 == away_goal and home_relegation > 0:
        lambda_diff = float(lambda_away - lambda_home)
        if lambda_diff >= min_diff:
            underdog_factor = float(np.clip(lambda_diff / diff_scale, 0.0, 1.0))
            shift_prob = float(np.clip(base_shift * home_relegation * underdog_factor, 0.0, max_draw_shift))
            if np.random.random() < shift_prob:
                home_goal = away_goal

    # awayがアンダードッグで1点差負け
    elif away_goal + 1 == home_goal and away_relegation > 0:
        lambda_diff = float(lambda_home - lambda_away)
        if lambda_diff >= min_diff:
            underdog_factor = float(np.clip(lambda_diff / diff_scale, 0.0, 1.0))
            shift_prob = float(np.clip(base_shift * away_relegation * underdog_factor, 0.0, max_draw_shift))
            if np.random.random() < shift_prob:
                away_goal = home_goal

    return home_goal, away_goal

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
    promoted_prior_name="none",
    promoted_prev_weight=0.0,
    promoted_attack_prior=1.0,
    promoted_defense_prior=1.0,
    pressure_setting=None,
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
        promoted_prev_weight=promoted_prev_weight,
        promoted_attack_prior=promoted_attack_prior,
        promoted_defense_prior=promoted_defense_prior,
    )

    # Shrinkageは今回比較しない。前回検証で不採用のため1.00固定。
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

    if pressure_setting is None:
        pressure_setting = PRESSURE_SETTING_CANDIDATES[0]

    pressure_name = str(pressure_setting.get("name", "pressure_off"))
    total_games_by_team = calculate_total_games_by_team(target_df, teams)

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
        played = calculate_games_played_by_team(train_df, teams)

        for row in test_df.itertuples(index=False):
            row_dict = row._asdict()
            home = row_dict["home"]
            away = row_dict["away"]

            pressure_context = calculate_pressure_context(
                table=table,
                played=played,
                total_games_by_team=total_games_by_team,
                teams=teams,
                pressure_setting=pressure_setting,
            )

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

            lambda_home, lambda_away, pressure_info = apply_pressure_effects_to_lambdas(
                lambda_home=lambda_home,
                lambda_away=lambda_away,
                home=home,
                away=away,
                pressure_context=pressure_context,
                pressure_setting=pressure_setting,
            )

            hg, ag = sample_score_with_draw_factor(
                lambda_home=lambda_home,
                lambda_away=lambda_away,
                draw_factor=draw_factor,
                max_goals=MAX_GOALS_FOR_SCORE_GRID,
                max_match_draw_prob=max_match_draw_prob,
            )

            hg, ag = apply_relegation_underdog_draw_shift(
                home_goal=hg,
                away_goal=ag,
                lambda_home=lambda_home,
                lambda_away=lambda_away,
                pressure_info=pressure_info,
                pressure_setting=pressure_setting,
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
            "lambda_cap_label": "none" if LAMBDA_CAP is None else f"cap{float(LAMBDA_CAP):g}",
            "lambda_cap": np.nan if LAMBDA_CAP is None else float(LAMBDA_CAP),
            "promoted_prior_name": promoted_prior_name,
            "promoted_prev_weight": float(promoted_prev_weight),
            "promoted_attack_prior": float(promoted_attack_prior),
            "promoted_defense_prior": float(promoted_defense_prior),
            "pressure_setting": pressure_name,
            "use_pressure": bool(pressure_setting.get("use_pressure", False)),
            "pressure_window_games": int(pressure_setting.get("pressure_window_games", 8)),
            "title_boost": float(pressure_setting.get("title_boost", 0.0)),
            "title_points_window": float(pressure_setting.get("title_points_window", 8.0)),
            "title_rank_cutoff": int(pressure_setting.get("title_rank_cutoff", 5)),
            "relegation_draw_shift": float(pressure_setting.get("relegation_draw_shift", 0.0)),
            "relegation_points_window": float(pressure_setting.get("relegation_points_window", 6.0)),
            "relegation_rank_buffer": int(pressure_setting.get("relegation_rank_buffer", 3)),
            "min_underdog_lambda_diff": float(pressure_setting.get("min_underdog_lambda_diff", 0.10)),
            "underdog_lambda_diff_scale": float(pressure_setting.get("underdog_lambda_diff_scale", 0.75)),
            "max_draw_shift": float(pressure_setting.get("max_draw_shift", 0.0)),
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
            "is_promoted": bool(prev_games_by_team.get(team, 0) == 0),
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
        "lambda_cap_label": "none" if LAMBDA_CAP is None else f"cap{float(LAMBDA_CAP):g}",
        "lambda_cap": np.nan if LAMBDA_CAP is None else float(LAMBDA_CAP),
        "promoted_prior_name": promoted_prior_name,
        "promoted_prev_weight": float(promoted_prev_weight),
        "promoted_attack_prior": float(promoted_attack_prior),
        "promoted_defense_prior": float(promoted_defense_prior),
        "pressure_setting": pressure_name,
        "use_pressure": bool(pressure_setting.get("use_pressure", False)),
        "pressure_window_games": int(pressure_setting.get("pressure_window_games", 8)),
        "title_boost": float(pressure_setting.get("title_boost", 0.0)),
        "title_points_window": float(pressure_setting.get("title_points_window", 8.0)),
        "title_rank_cutoff": int(pressure_setting.get("title_rank_cutoff", 5)),
        "relegation_draw_shift": float(pressure_setting.get("relegation_draw_shift", 0.0)),
        "relegation_points_window": float(pressure_setting.get("relegation_points_window", 6.0)),
        "relegation_rank_buffer": int(pressure_setting.get("relegation_rank_buffer", 3)),
        "min_underdog_lambda_diff": float(pressure_setting.get("min_underdog_lambda_diff", 0.10)),
        "underdog_lambda_diff_scale": float(pressure_setting.get("underdog_lambda_diff_scale", 0.75)),
        "max_draw_shift": float(pressure_setting.get("max_draw_shift", 0.0)),
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

    promoted_mask = prediction_df["is_promoted"].astype(bool)
    summary["promoted_team_count"] = int(promoted_mask.sum())
    summary["promoted_mean_error"] = (
        float(prediction_df.loc[promoted_mask, "position_error"].mean())
        if promoted_mask.any()
        else np.nan
    )

    title_mask = prediction_df["actual_position"] <= min(3, n_teams)
    relegation_mask = prediction_df["actual_position"] >= max(1, n_teams - RELEGATION_SPOTS + 1)
    summary["title_contender_mean_error"] = (
        float(prediction_df.loc[title_mask, "position_error"].mean())
        if title_mask.any()
        else np.nan
    )
    summary["relegation_zone_mean_error"] = (
        float(prediction_df.loc[relegation_mask, "position_error"].mean())
        if relegation_mask.any()
        else np.nan
    )

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

# =========================
# 9. 出力
# =========================


def export_summary_html(df, output_path):
    display_df = df.copy()

    round_cols = [
        "prev_weight", "prev_decay", "elo_lambda_weight", "compat_weight", "current_decay", "shrinkage", "draw_factor",
        "max_match_draw_prob", "lambda_cap", "title_boost", "title_points_window", "relegation_draw_shift",
        "relegation_points_window", "min_underdog_lambda_diff", "underdog_lambda_diff_scale", "max_draw_shift",
        "matchup_time_decay", "mean_mae", "std_mae", "mean_prob_actual_position", "mean_sim_draw_rate",
        "mean_title_contender_error", "mean_relegation_zone_error", "mae_2023", "mae_2024", "mae_2025",
        "delta_vs_pressure_off",
    ]
    for col in round_cols:
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(display_df[col], errors="coerce").round(4)

    table_html = display_df.to_html(index=False, classes="prediction-table")

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>終盤補正 3年比較</title>
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
  <h1>終盤補正 3年比較</h1>
  <div class="note">
    <p>
      2022→2023、2023→2024、2024→2025を対象に、PREV_WEIGHT=0.20 / PREV_DECAY=0.995、
      ELO_LAMBDA_WEIGHT=0.20、大勝補正cap4、COMPAT_WEIGHT=0.20、DRAW_FACTOR=1.20、MAX_MATCH_DRAW_PROB=0.33、
      CURRENT_DECAY=1.000、LAMBDA_CAP=3.5を固定し、終盤補正を比較しています。
    </p>
    <p>
      昇格組補正は不採用のため固定でOFFです。終盤補正は、優勝争いチームのλブーストと、
      残留争いアンダードッグの1点差負けを引き分けへ寄せる補正だけを比較しています。
      N_SIM_SEARCH={N_SIM_SEARCH}, SHRINKAGE_FIXED={SHRINKAGE_FIXED},
      MATCHUP_PRIOR_N={MATCHUP_PRIOR_N}, MATCHUP_TIME_DECAY={MATCHUP_TIME_DECAY},
      MAX_GOALS_FOR_SCORE_GRID={MAX_GOALS_FOR_SCORE_GRID}。
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

    promoted_prior_name = str(PROMOTED_PRIOR_FIXED["name"])
    promoted_prev_weight = float(PROMOTED_PRIOR_FIXED["promoted_prev_weight"])
    promoted_attack_prior = float(PROMOTED_PRIOR_FIXED["promoted_attack_prior"])
    promoted_defense_prior = float(PROMOTED_PRIOR_FIXED["promoted_defense_prior"])

    print("\n==============================")
    print("終盤補正 3年比較 簡易版")
    print("==============================")
    print("HISTORICAL_CSV:", HISTORICAL_CSV)
    print("TARGET_YEARS:", TARGET_YEARS)
    print("FIXED_PREV_SETTING:", FIXED_PREV_SETTING)
    print("ELO_LAMBDA_WEIGHT_FIXED:", ELO_LAMBDA_WEIGHT_FIXED)
    print("GOAL_ADJUST_FIXED:", GOAL_ADJUST_FIXED)
    print("COMPAT_WEIGHT_FIXED:", COMPAT_WEIGHT_FIXED)
    print("DRAW_FACTOR_FIXED:", DRAW_FACTOR_FIXED)
    print("MAX_MATCH_DRAW_PROB_FIXED:", MAX_MATCH_DRAW_PROB_FIXED)
    print("CURRENT_DECAY_FIXED:", CURRENT_DECAY_FIXED)
    print("LAMBDA_CAP_FIXED:", LAMBDA_CAP)
    print("PROMOTED_PRIOR_FIXED:", PROMOTED_PRIOR_FIXED)
    print("PRESSURE_SETTING_CANDIDATES:", PRESSURE_SETTING_CANDIDATES)
    print("SHRINKAGE_FIXED:", SHRINKAGE_FIXED)
    print("MAX_GOALS_FOR_SCORE_GRID:", MAX_GOALS_FOR_SCORE_GRID)
    print("MATCHUP_PRIOR_N:", MATCHUP_PRIOR_N)
    print("MATCHUP_TIME_DECAY:", MATCHUP_TIME_DECAY)
    print("COMPAT_FACTOR_MIN/MAX:", COMPAT_FACTOR_MIN, COMPAT_FACTOR_MAX)
    print("USE_RAW_LEAGUE_AVG_FOR_LAMBDA:", USE_RAW_LEAGUE_AVG_FOR_LAMBDA)
    print("N_SIM_SEARCH:", N_SIM_SEARCH)
    print("K_FACTOR:", K_FACTOR)
    print("HOME_ADV:", HOME_ADV)
    print("ELO_FACTOR_SCALE:", ELO_FACTOR_SCALE)
    print("ELO_FACTOR_MIN/MAX:", ELO_FACTOR_MIN, ELO_FACTOR_MAX)
    print("USE_PROMOTED_PREV_ZERO:", USE_PROMOTED_PREV_ZERO)

    detail_rows = []
    all_predictions = []

    total = len(PRESSURE_SETTING_CANDIDATES) * len(TARGET_YEARS)
    done = 0

    draw_factor = float(DRAW_FACTOR_FIXED)
    max_match_draw_prob = MAX_MATCH_DRAW_PROB_FIXED
    cap_label = "none" if max_match_draw_prob is None else f"cap{int(round(float(max_match_draw_prob) * 100)):03d}"
    shrinkage = float(SHRINKAGE_FIXED)
    current_decay = float(CURRENT_DECAY_FIXED)
    lambda_cap_label = f"cap{float(LAMBDA_CAP):g}"

    for pressure_setting in PRESSURE_SETTING_CANDIDATES:
        pressure_name = str(pressure_setting["name"])

        for target_year in TARGET_YEARS:
            done += 1
            print(
                f"\n[{done}/{total}] "
                f"{prev_name}, target={target_year}, "
                f"ELO_LAMBDA_WEIGHT={elo_lambda_weight:.2f}, "
                f"GOAL_ADJUST={goal_adjust_setting['name']}, "
                f"COMPAT_WEIGHT={compat_weight:.2f}, "
                f"DRAW_FACTOR={draw_factor:.2f}, "
                f"MAX_MATCH_DRAW_PROB={cap_label}, "
                f"LAMBDA_CAP={lambda_cap_label}, "
                f"PRESSURE={pressure_name}, "
                f"TITLE_BOOST={float(pressure_setting.get('title_boost', 0.0)):.3f}, "
                f"REL_DRAW_SHIFT={float(pressure_setting.get('relegation_draw_shift', 0.0)):.3f}"
            )

            pressure_seed = sum(ord(ch) for ch in pressure_name)
            cap_seed = 0 if max_match_draw_prob is None else int(round(float(max_match_draw_prob) * 10000))
            lambda_cap_seed = int(round(float(LAMBDA_CAP) * 1000))
            seed_offset = (
                int(round(draw_factor * 1000))
                + cap_seed
                + lambda_cap_seed
                + pressure_seed
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
                promoted_prior_name=promoted_prior_name,
                promoted_prev_weight=promoted_prev_weight,
                promoted_attack_prior=promoted_attack_prior,
                promoted_defense_prior=promoted_defense_prior,
                pressure_setting=pressure_setting,
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
            prediction_df["lambda_cap_label"] = lambda_cap_label
            prediction_df["lambda_cap"] = float(LAMBDA_CAP)
            prediction_df["promoted_prior_name"] = promoted_prior_name
            prediction_df["promoted_prev_weight"] = promoted_prev_weight
            prediction_df["promoted_attack_prior"] = promoted_attack_prior
            prediction_df["promoted_defense_prior"] = promoted_defense_prior
            prediction_df["pressure_setting"] = pressure_name
            prediction_df["use_pressure"] = bool(pressure_setting.get("use_pressure", False))
            prediction_df["title_boost"] = float(pressure_setting.get("title_boost", 0.0))
            prediction_df["relegation_draw_shift"] = float(pressure_setting.get("relegation_draw_shift", 0.0))
            prediction_df["shrinkage"] = float(shrinkage)
            all_predictions.append(prediction_df)

            print(
                f"  MAE={summary['mae']:.4f}, "
                f"実順位確率平均={summary['mean_prob_actual_position']:.4f}, "
                f"引分率={summary['sim_draw_rate']:.4f}"
            )

    detail_df = pd.DataFrame(detail_rows)
    detail_df.to_csv(OUTPUT_DETAIL_CSV, index=False, encoding="utf-8-sig")

    index_cols = [
        "prev_setting_name", "prev_weight", "prev_decay", "elo_lambda_weight",
        "goal_adjust_name", "goal_adjust_mode", "goal_cap_for_strength",
        "compat_weight", "current_decay", "shrinkage", "draw_factor", "max_match_draw_prob_label", "lambda_cap_label",
        "promoted_prior_name", "promoted_prev_weight", "promoted_attack_prior", "promoted_defense_prior",
        "pressure_setting", "use_pressure", "pressure_window_games", "title_boost", "title_points_window", "title_rank_cutoff",
        "relegation_draw_shift", "relegation_points_window", "relegation_rank_buffer",
        "min_underdog_lambda_diff", "underdog_lambda_diff_scale", "max_draw_shift",
        "max_goals_for_score_grid", "matchup_prior_n", "matchup_time_decay", "compat_factor_min", "compat_factor_max",
    ]

    pivot_mae = detail_df.pivot_table(
        index=index_cols,
        columns="target_year",
        values="mae",
        aggfunc="mean",
    )
    pivot_mae.columns = [f"mae_{int(col)}" for col in pivot_mae.columns]
    pivot_mae = pivot_mae.reset_index()

    grouped = detail_df.groupby(index_cols, as_index=False).agg(
        max_match_draw_prob=("max_match_draw_prob", "first"),
        lambda_cap=("lambda_cap", "first"),
        mean_mae=("mae", "mean"),
        std_mae=("mae", "std"),
        mean_prob_actual_position=("mean_prob_actual_position", "mean"),
        mean_sim_draw_rate=("sim_draw_rate", "mean"),
        mean_title_contender_error=("title_contender_mean_error", "mean"),
        mean_relegation_zone_error=("relegation_zone_mean_error", "mean"),
    )

    summary_df = grouped.merge(pivot_mae, on=index_cols, how="left")

    base_rows = summary_df[summary_df["pressure_setting"].astype(str) == "pressure_off"]
    base_pressure_off = float(base_rows.iloc[0]["mean_mae"]) if len(base_rows) > 0 else np.nan
    summary_df["delta_vs_pressure_off"] = summary_df["mean_mae"] - base_pressure_off

    summary_df = summary_df.sort_values(["mean_mae", "std_mae"]).reset_index(drop=True)
    summary_df.insert(0, "rank", summary_df.index + 1)
    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    export_summary_html(summary_df, OUTPUT_SUMMARY_HTML)

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    best_pressure_setting = str(summary_df.iloc[0]["pressure_setting"])
    best_predictions_df = predictions_df[predictions_df["pressure_setting"] == best_pressure_setting].copy()
    best_predictions_df.to_csv(OUTPUT_BEST_PREDICTION_CSV, index=False, encoding="utf-8-sig")
    predictions_df.to_csv(OUTPUT_ALL_PREDICTIONS_CSV, index=False, encoding="utf-8-sig")

    print("\n==============================")
    print("検証結果")
    print("==============================")
    show_cols = [
        "rank", "pressure_setting", "use_pressure", "title_boost", "relegation_draw_shift", "pressure_window_games",
        "mean_mae", "std_mae", "mean_prob_actual_position", "mean_sim_draw_rate",
        "mean_title_contender_error", "mean_relegation_zone_error",
        "mae_2023", "mae_2024", "mae_2025", "delta_vs_pressure_off",
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
    print("DRAW_FACTOR:", draw_factor)
    print("MAX_MATCH_DRAW_PROB:", cap_label)
    print("CURRENT_DECAY:", current_decay)
    print("LAMBDA_CAP:", lambda_cap_label)
    print("PROMOTED_PRIOR:", promoted_prior_name)
    print("SHRINKAGE:", shrinkage)
    print("PRESSURE_SETTING:", best_pressure_setting)
    print("mean_mae:", round(float(summary_df.iloc[0]["mean_mae"]), 4))


if __name__ == "__main__":
    main()
