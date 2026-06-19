import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path
from itertools import product

# ============================================================
# J1順位予測 ver.1.5 候補: 前年スタッツのみで前年レーティングを構成
# ------------------------------------------------------------
# 目的:
#   jleague_team_stats_yearly_wide_2019_2025.csv を使い、
#   前年レーティングを「前年度公式チームスタッツのみ」で作った場合を検証する。
#
# 前回の grid_search_v15_prev_stats.py との違い:
#   前回: 既存の前年試合別係数に、公式スタッツ係数を少し混ぜる
#   今回: 既存の前年試合別係数を使わず、前年公式スタッツ係数だけで前年レーティングを作る
#
# 方針:
#   ・v1.4の大勝補正 cap4 は維持
#   ・当年前半の係数は従来通り「得点 + 枠内シュート」
#   ・前年係数は「得点/xG/枠内/シュート」および「失点/xGA/被枠内/被シュート」から作る
#   ・前年公式スタッツは年間合計なので、前年側は home/away を分けない
#   ・昇格組も前年J2/J3スタッツがあれば、リーグ補正をかけて反映する
#
# 使い方:
#   python grid_search_v15_prev_stats_only.py
#
# 出力:
#   v15_prev_stats_only_grid_results.csv
#   v15_prev_stats_only_best_prediction.csv
#   v15_prev_stats_only_best_prediction.html
#   v15_prev_stats_only_baseline_v14_prediction.csv
#   v15_prev_stats_only_baseline_v14_prediction.html
# ============================================================


# =========================
# 1. 設定
# =========================

BASE_DIR = Path(__file__).resolve().parent

# 検証対象年。
# j1_2021_match_stats_merged_fixed.csv などがそろったら複数年に拡張できます。
TARGET_YEARS = [2025]

# ファイル名
TEAM_STATS_CSV_NAME = "jleague_team_stats_yearly_wide_2019_2025.csv"
HISTORICAL_J1_FILENAME = "j1_historical_results_1993_2025_table_fixed.csv"

# v1.4 大勝補正
GOAL_ADJUST_MODE = "cap4"
GOAL_CAP_FOR_STRENGTH = 4
USE_RAW_LEAGUE_AVG_FOR_LAMBDA = True

# 当年前半の基本モデル
SOT_WEIGHT = 0.1

# v1.4 baseline用: 前年試合別データで使う特徴量
PREV_SHOT_WEIGHT = 0.1

# v1.4 baseline用: 当年前半と前年試合別レーティングの混合比率
PREV_WEIGHT_BASELINE = 0.4

# 今回の探索対象: 当年前半と「前年スタッツのみレーティング」の混合比率
# 0.0は前年を使わない比較用。0.4がv1.4と同じ混合比率。
PREV_WEIGHT_CANDIDATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

# リーグ補正: J2/J3の前年スタッツをJ1基準へ近づけるため、1.0へ縮小する
# 例: J2で1.20 → 1 + (1.20 - 1) * 0.60 = 1.12
LEAGUE_SHRINK = {
    "j1": 1.00,
    "j2": 0.60,
    "j3": 0.45,
}

# Elo補正
INITIAL_ELO = 1500
K_FACTOR = 16
HOME_ADV = 0
ELO_LAMBDA_WEIGHT = 0.20
ELO_FACTOR_SCALE = 4000
ELO_FACTOR_MIN = 0.90
ELO_FACTOR_MAX = 1.10

# 相性Effect
USE_MATCHUP_EFFECT = True
COMPAT_WEIGHT = 0.20
MATCHUP_PRIOR_N = 30
MATCHUP_TIME_DECAY = 0.97
COMPAT_FACTOR_MIN = 0.95
COMPAT_FACTOR_MAX = 1.05

# シミュレーション
# 探索中は1000程度、最終候補だけ10000で確認。
N_SIM_SEARCH = 1000
N_SIM_FINAL = 10000
RUN_FINAL_BEST = True
RUN_BASELINE_FINAL = True
DECAY = 1.0
LAMBDA_CAP = 3.5
RANDOM_SEED = 42

# 出力
OUTPUT_GRID_CSV = BASE_DIR / "v15_prev_stats_only_grid_results.csv"
OUTPUT_BEST_CSV = BASE_DIR / "v15_prev_stats_only_best_prediction.csv"
OUTPUT_BEST_HTML = BASE_DIR / "v15_prev_stats_only_best_prediction.html"
OUTPUT_BASELINE_CSV = BASE_DIR / "v15_prev_stats_only_baseline_v14_prediction.csv"
OUTPUT_BASELINE_HTML = BASE_DIR / "v15_prev_stats_only_baseline_v14_prediction.html"


# =========================
# 2. グリッド候補
# =========================

# 攻撃側: 合計1.0
ATTACK_WEIGHT_CANDIDATES = [
    {"goals": 0.50, "xg": 0.20, "sot": 0.20, "shots": 0.10},
    {"goals": 0.40, "xg": 0.30, "sot": 0.20, "shots": 0.10},
    {"goals": 0.40, "xg": 0.20, "sot": 0.30, "shots": 0.10},
    {"goals": 0.35, "xg": 0.35, "sot": 0.20, "shots": 0.10},
    {"goals": 0.30, "xg": 0.40, "sot": 0.20, "shots": 0.10},
]

# 守備側: 高いほど守備が悪い係数。合計1.0
DEFENSE_WEIGHT_CANDIDATES = [
    {"ga": 0.50, "xga": 0.20, "sota": 0.20, "sa": 0.10},
    {"ga": 0.40, "xga": 0.30, "sota": 0.20, "sa": 0.10},
    {"ga": 0.40, "xga": 0.20, "sota": 0.30, "sa": 0.10},
    {"ga": 0.35, "xga": 0.35, "sota": 0.20, "sa": 0.10},
    {"ga": 0.30, "xga": 0.40, "sota": 0.20, "sa": 0.10},
]


# =========================
# 3. ユーティリティ
# =========================

def find_file(filename, required=True):
    candidates = [
        BASE_DIR / filename,
        BASE_DIR / "data" / filename,
        BASE_DIR.parent / "data" / filename,
        BASE_DIR.parent / filename,
        Path(filename),
    ]

    for path in candidates:
        if path.exists():
            return path

    if required:
        raise FileNotFoundError(
            f"{filename} が見つかりません。スクリプトと同じフォルダ、または data フォルダに置いてください。"
        )
    return None


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
        "Ｖ・ファーレン長崎": "長崎",
        "V・ファーレン長崎": "長崎",
        "ロアッソ熊本": "熊本",
        "水戸ホーリーホック": "水戸",
        "ツエーゲン金沢": "金沢",
        "栃木ＳＣ": "栃木",
        "栃木SC": "栃木",
        "ＦＣ琉球": "琉球",
        "FC琉球": "琉球",
        "ＦＣ岐阜": "岐阜",
        "FC岐阜": "岐阜",
        "カマタマーレ讃岐": "讃岐",
        "ギラヴァンツ北九州": "北九州",
        "ブラウブリッツ秋田": "秋田",
        "いわきＦＣ": "いわき",
        "いわきFC": "いわき",
        "藤枝ＭＹＦＣ": "藤枝",
        "藤枝MYFC": "藤枝",
    }
    return name_map.get(name, name)


def clean_team_names(df):
    df = df.copy()
    for col in ["home", "away"]:
        if col in df.columns:
            df[col] = df[col].apply(standardize_team_name)
    return df


def safe_positive_mean(series, fallback=1.0):
    value = pd.to_numeric(series, errors="coerce").mean()
    if not np.isfinite(value) or value <= 0:
        return fallback
    return float(value)


def safe_ratio(row, col, default=1.0):
    if col not in row.index:
        return default
    val = pd.to_numeric(row[col], errors="coerce")
    if not np.isfinite(val) or val <= 0:
        return default
    return float(val)


def shrink_ratio(ratio, league):
    shrink = LEAGUE_SHRINK.get(str(league).lower(), 1.0)
    return 1.0 + (ratio - 1.0) * shrink


# =========================
# 4. データ読み込み
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

    if "home_goal" not in df.columns:
        df["home_goal"] = pd.to_numeric(df["home_goals"], errors="coerce")
    else:
        df["home_goal"] = pd.to_numeric(df["home_goal"], errors="coerce")

    if "away_goal" not in df.columns:
        df["away_goal"] = pd.to_numeric(df["away_goals"], errors="coerce")
    else:
        df["away_goal"] = pd.to_numeric(df["away_goal"], errors="coerce")

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


def load_team_stats_csv(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "team_std" in df.columns:
        df["team_std"] = df["team_std"].apply(standardize_team_name)
    elif "team" in df.columns:
        df["team_std"] = df["team"].apply(standardize_team_name)
    else:
        raise ValueError("team または team_std 列が必要です。")

    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["league"] = df["league"].astype(str).str.lower()
    return df


# =========================
# 5. 順位表
# =========================

def calculate_table(match_df, teams):
    table = {team: {"points": 0, "gf": 0, "ga": 0, "gd": 0} for team in teams}

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


def copy_table(table):
    return {team: stats.copy() for team, stats in table.items()}


def make_ranking(table):
    return sorted(
        table.items(),
        key=lambda x: (-x[1]["points"], -x[1]["gd"], -x[1]["gf"])
    )


# =========================
# 6. 大勝補正・当年前半/前年試合別攻守係数
# =========================

def add_goal_for_strength_columns(df):
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


def calculate_strengths_home_away(history_df, teams, decay=1.0, feature_weight=0.1, feature_type="sot"):
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

    if use_feature:
        history_df[home_feature_col] = pd.to_numeric(history_df[home_feature_col], errors="coerce")
        history_df[away_feature_col] = pd.to_numeric(history_df[away_feature_col], errors="coerce")
        home_avg_feature = safe_positive_mean(history_df[home_feature_col], fallback=1.0)
        away_avg_feature = safe_positive_mean(history_df[away_feature_col], fallback=1.0)
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
        home_games = history_df[history_df["home"] == team].copy().sort_values("date", ascending=False)
        away_games = history_df[history_df["away"] == team].copy().sort_values("date", ascending=False)

        home_gf = home_ga = home_feature_for = home_feature_against = home_w = 0.0
        for i, row in enumerate(home_games.itertuples(index=False)):
            rd = row._asdict()
            weight = decay ** i
            home_gf += rd["home_goal_strength"] * weight
            home_ga += rd["away_goal_strength"] * weight
            if use_feature:
                home_feature_for += rd[home_feature_col] * weight
                home_feature_against += rd[away_feature_col] * weight
            home_w += weight

        if home_w == 0:
            home_goal_attack = home_goal_defense = home_feature_attack = home_feature_defense = 1.0
        else:
            home_goal_attack = (home_gf / home_w) / strength_home_avg_goals
            home_goal_defense = (home_ga / home_w) / strength_away_avg_goals
            if use_feature:
                home_feature_attack = (home_feature_for / home_w) / home_avg_feature
                home_feature_defense = (home_feature_against / home_w) / away_avg_feature
            else:
                home_feature_attack = home_feature_defense = 1.0

        away_gf = away_ga = away_feature_for = away_feature_against = away_w = 0.0
        for i, row in enumerate(away_games.itertuples(index=False)):
            rd = row._asdict()
            weight = decay ** i
            away_gf += rd["away_goal_strength"] * weight
            away_ga += rd["home_goal_strength"] * weight
            if use_feature:
                away_feature_for += rd[away_feature_col] * weight
                away_feature_against += rd[home_feature_col] * weight
            away_w += weight

        if away_w == 0:
            away_goal_attack = away_goal_defense = away_feature_attack = away_feature_defense = 1.0
        else:
            away_goal_attack = (away_gf / away_w) / strength_away_avg_goals
            away_goal_defense = (away_ga / away_w) / strength_home_avg_goals
            if use_feature:
                away_feature_attack = (away_feature_for / away_w) / away_avg_feature
                away_feature_defense = (away_feature_against / away_w) / home_avg_feature
            else:
                away_feature_attack = away_feature_defense = 1.0

        strengths[team] = {
            "home_attack": safe_strength(goal_weight * home_goal_attack + feature_weight * home_feature_attack),
            "home_defense": safe_strength(goal_weight * home_goal_defense + feature_weight * home_feature_defense),
            "away_attack": safe_strength(goal_weight * away_goal_attack + feature_weight * away_feature_attack),
            "away_defense": safe_strength(goal_weight * away_goal_defense + feature_weight * away_feature_defense),
        }

    return strengths, home_avg_goals, away_avg_goals


# =========================
# 7. 前年公式スタッツのみレーティング
# =========================

def build_prev_stats_only_strengths(team_stats_df, target_year, teams, attack_weights, defense_weights):
    prev_year = target_year - 1
    prev_stats = team_stats_df[team_stats_df["season"] == prev_year].copy()

    stats_by_team = {}
    for _, row in prev_stats.iterrows():
        team = row["team_std"]
        if team not in stats_by_team:
            stats_by_team[team] = row

    strengths = {}
    meta = {}

    for team in teams:
        if team not in stats_by_team:
            strengths[team] = {
                "home_attack": 1.0,
                "away_attack": 1.0,
                "home_defense": 1.0,
                "away_defense": 1.0,
            }
            meta[team] = {
                "prev_stats_year": int(prev_year),
                "prev_stats_league": "missing",
                "prev_stats_attack_ratio": 1.0,
                "prev_stats_defense_bad_ratio": 1.0,
                "has_prev_stats": False,
            }
            continue

        row = stats_by_team[team]
        league = str(row.get("league", "j1")).lower()

        # 攻撃: 高いほど強い
        attack_raw = (
            attack_weights["goals"] * safe_ratio(row, "goals_ratio")
            + attack_weights["xg"] * safe_ratio(row, "xg_ratio")
            + attack_weights["sot"] * safe_ratio(row, "shots_on_target_ratio")
            + attack_weights["shots"] * safe_ratio(row, "shots_ratio")
        )

        # 守備: 高いほど守備が悪い
        defense_raw = (
            defense_weights["ga"] * safe_ratio(row, "goals_against_ratio")
            + defense_weights["xga"] * safe_ratio(row, "xga_ratio")
            + defense_weights["sota"] * safe_ratio(row, "shots_on_target_against_ratio")
            + defense_weights["sa"] * safe_ratio(row, "shots_against_ratio")
        )

        attack_ratio = shrink_ratio(attack_raw, league)
        defense_bad_ratio = shrink_ratio(defense_raw, league)

        # 前年度公式スタッツは年間合計なので、前年側ではhome/awayを分けない
        strengths[team] = {
            "home_attack": attack_ratio,
            "away_attack": attack_ratio,
            "home_defense": defense_bad_ratio,
            "away_defense": defense_bad_ratio,
        }
        meta[team] = {
            "prev_stats_year": int(prev_year),
            "prev_stats_league": league,
            "prev_stats_attack_ratio": attack_ratio,
            "prev_stats_defense_bad_ratio": defense_bad_ratio,
            "has_prev_stats": True,
        }

    return strengths, meta


def count_team_games(df, team):
    return int(((df["home"] == team) | (df["away"] == team)).sum())


def blend_current_with_previous(current_strengths, previous_strengths, teams, prev_weight, prev_stats_meta=None, previous_df=None, promoted_prev_zero=False):
    blended = {}
    prev_weight_by_team = {}
    prev_games_by_team = {}

    def safe_blend(current_value, previous_value, weight):
        current_ok = np.isfinite(current_value)
        previous_ok = np.isfinite(previous_value)
        if current_ok and previous_ok:
            return (1.0 - weight) * current_value + weight * previous_value
        if current_ok:
            return current_value
        if previous_ok:
            return previous_value
        return 1.0

    for team in teams:
        prev_games = count_team_games(previous_df, team) if previous_df is not None else None
        has_prev_stats = True
        if prev_stats_meta is not None:
            has_prev_stats = bool(prev_stats_meta.get(team, {}).get("has_prev_stats", False))

        if promoted_prev_zero and previous_df is not None and prev_games == 0:
            # v1.4 baseline: 昇格組は前年J1データがないので前年重み0
            effective_prev_weight = 0.0
        elif prev_stats_meta is not None and not has_prev_stats:
            # stats-only: 前年スタッツがなければ無理に混ぜない
            effective_prev_weight = 0.0
        else:
            effective_prev_weight = prev_weight

        prev_games_by_team[team] = prev_games
        prev_weight_by_team[team] = effective_prev_weight

        blended[team] = {
            key: safe_blend(
                current_strengths[team][key],
                previous_strengths.get(team, {}).get(key, 1.0),
                effective_prev_weight,
            )
            for key in ["home_attack", "home_defense", "away_attack", "away_defense"]
        }

    return blended, prev_weight_by_team, prev_games_by_team


# =========================
# 8. Elo / 相性Effect / 期待得点
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
    all_teams = pd.unique(pd.concat([previous_df[["home", "away"]], train_df[["home", "away"]]]).values.ravel())
    ratings = {team: INITIAL_ELO for team in all_teams}
    combined = pd.concat([previous_df, train_df], ignore_index=True).sort_values("date").reset_index(drop=True)

    for _, row in combined.iterrows():
        update_elo_one_match(
            ratings=ratings,
            home=row["home"],
            away=row["away"],
            home_goal=int(row["home_goal"]),
            away_goal=int(row["away_goal"]),
            k_factor=k_factor,
            home_adv=home_adv,
        )
    return {team: ratings.get(team, INITIAL_ELO) for team in teams}


def build_matchup_effects_j1(historical_j1_df, cutoff_date, target_year, prior_n=20, time_decay=0.94, k_factor=16, home_adv=0):
    df = historical_j1_df.copy()
    df = df[df["date"] <= cutoff_date].copy().sort_values("date").reset_index(drop=True)

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

        update_elo_one_match(ratings, home, away, hg, ag, k_factor=k_factor, home_adv=home_adv)

    effects = {}
    for key in weighted_residual_sum:
        weighted_n = weighted_match_sum[key]
        raw_effect = weighted_residual_sum[key] / weighted_n if weighted_n > 0 else 0.0
        shrink = weighted_n / (weighted_n + prior_n)
        effects[key] = raw_effect * shrink
    return effects


def safe_lambda(lam):
    if lam is None or not np.isfinite(lam) or lam < 0:
        return 0.05
    if LAMBDA_CAP is not None:
        lam = min(lam, LAMBDA_CAP)
    return float(max(lam, 0.05))


def expected_goals_home_away(home, away, strengths, home_avg_goals, away_avg_goals, elo_ratings=None, matchup_effects=None):
    lambda_home = strengths[home]["home_attack"] * strengths[away]["away_defense"] * home_avg_goals
    lambda_away = strengths[away]["away_attack"] * strengths[home]["home_defense"] * away_avg_goals

    if elo_ratings is not None and ELO_LAMBDA_WEIGHT > 0:
        home_elo = elo_ratings.get(home, INITIAL_ELO)
        away_elo = elo_ratings.get(away, INITIAL_ELO)
        elo_diff = home_elo - away_elo
        home_factor = 10 ** ((ELO_LAMBDA_WEIGHT * elo_diff) / ELO_FACTOR_SCALE)
        away_factor = 10 ** ((-ELO_LAMBDA_WEIGHT * elo_diff) / ELO_FACTOR_SCALE)
        lambda_home *= np.clip(home_factor, ELO_FACTOR_MIN, ELO_FACTOR_MAX)
        lambda_away *= np.clip(away_factor, ELO_FACTOR_MIN, ELO_FACTOR_MAX)

    if matchup_effects is not None and COMPAT_WEIGHT > 0:
        home_effect = matchup_effects.get((home, away), 0.0)
        away_effect = matchup_effects.get((away, home), 0.0)
        home_factor = np.clip(1 + COMPAT_WEIGHT * home_effect, COMPAT_FACTOR_MIN, COMPAT_FACTOR_MAX)
        away_factor = np.clip(1 + COMPAT_WEIGHT * away_effect, COMPAT_FACTOR_MIN, COMPAT_FACTOR_MAX)
        lambda_home *= home_factor
        lambda_away *= away_factor

    return safe_lambda(lambda_home), safe_lambda(lambda_away)


# =========================
# 9. シーズン文脈の準備
# =========================

def prepare_season_context(target_year, team_stats_df):
    current_path = find_file(f"j1_{target_year}_match_stats_merged_fixed.csv", required=False)
    previous_path = find_file(f"j1_{target_year - 1}_match_stats_merged_fixed.csv", required=False)

    if current_path is None or previous_path is None:
        print(f"[skip] {target_year}: 必要CSVが見つかりません current={current_path}, previous={previous_path}")
        return None

    current_df = load_match_stats_csv(current_path)
    previous_df = load_match_stats_csv(previous_path)
    teams = list(pd.unique(current_df[["home", "away"]].values.ravel()))

    split = int(len(current_df) * 0.5)
    train_df = current_df.iloc[:split].copy()
    test_df = current_df.iloc[split:].copy()

    current_strengths, home_avg_goals, away_avg_goals = calculate_strengths_home_away(
        train_df,
        teams=teams,
        decay=DECAY,
        feature_weight=SOT_WEIGHT,
        feature_type="sot",
    )

    # v1.4 baseline比較用。stats-only本体では使わない。
    previous_strengths_base, _, _ = calculate_strengths_home_away(
        previous_df,
        teams=teams,
        decay=DECAY,
        feature_weight=PREV_SHOT_WEIGHT,
        feature_type="shots",
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
        historical_path = find_file(HISTORICAL_J1_FILENAME, required=False)
        if historical_path is not None:
            historical_j1_df = load_historical_j1_csv(historical_path)
            matchup_effects = build_matchup_effects_j1(
                historical_j1_df=historical_j1_df,
                cutoff_date=train_df["date"].max(),
                target_year=target_year,
                prior_n=MATCHUP_PRIOR_N,
                time_decay=MATCHUP_TIME_DECAY,
                k_factor=K_FACTOR,
                home_adv=HOME_ADV,
            )
        else:
            print(f"[info] {target_year}: historical csvがないため相性EffectはOFF扱いにします。")

    actual_table = calculate_table(current_df, teams)
    actual_position = {team: pos + 1 for pos, (team, _) in enumerate(make_ranking(actual_table))}
    initial_table = calculate_table(train_df, teams)

    return {
        "target_year": target_year,
        "teams": teams,
        "current_df": current_df,
        "previous_df": previous_df,
        "train_df": train_df,
        "test_df": test_df,
        "current_strengths": current_strengths,
        "previous_strengths_base": previous_strengths_base,
        "home_avg_goals": home_avg_goals,
        "away_avg_goals": away_avg_goals,
        "elo_ratings": elo_ratings,
        "matchup_effects": matchup_effects,
        "actual_position": actual_position,
        "initial_table": initial_table,
    }


# =========================
# 10. 評価
# =========================

def simulate_with_strengths(context, strengths, prev_weight_by_team, prev_games_by_team, prev_stats_meta, n_sim, seed, return_prediction=False):
    teams = context["teams"]

    fixtures = []
    for row in context["test_df"].itertuples(index=False):
        rd = row._asdict()
        home = rd["home"]
        away = rd["away"]
        lam_home, lam_away = expected_goals_home_away(
            home=home,
            away=away,
            strengths=strengths,
            home_avg_goals=context["home_avg_goals"],
            away_avg_goals=context["away_avg_goals"],
            elo_ratings=context["elo_ratings"],
            matchup_effects=context["matchup_effects"],
        )
        fixtures.append((home, away, lam_home, lam_away))

    rng = np.random.default_rng(seed)
    position_counts = {team: Counter() for team in teams}
    points_sum = {team: 0.0 for team in teams}
    gf_sum = {team: 0.0 for team in teams}
    ga_sum = {team: 0.0 for team in teams}
    gd_sum = {team: 0.0 for team in teams}

    for _ in range(n_sim):
        table = copy_table(context["initial_table"])

        for home, away, lam_home, lam_away in fixtures:
            hg = int(rng.poisson(lam_home))
            ag = int(rng.poisson(lam_away))

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
        for pos, (team, _) in enumerate(ranking):
            position_counts[team][pos + 1] += 1

        for team in teams:
            points_sum[team] += table[team]["points"]
            gf_sum[team] += table[team]["gf"]
            ga_sum[team] += table[team]["ga"]
            gd_sum[team] += table[team]["gd"]

    rows = []
    n_teams = len(teams)
    for team in teams:
        avg_pred_pos = sum(pos * (position_counts[team][pos] / n_sim) for pos in range(1, n_teams + 1))
        meta = prev_stats_meta.get(team, {}) if prev_stats_meta is not None else {}
        rows.append({
            "season": context["target_year"],
            "team": team,
            "actual_position": context["actual_position"][team],
            "avg_pred_position": avg_pred_pos,
            "position_error": abs(avg_pred_pos - context["actual_position"][team]),
            "most_likely_position": position_counts[team].most_common(1)[0][0],
            "champion_prob": position_counts[team][1] / n_sim,
            "top3_prob": sum(position_counts[team][p] for p in range(1, 4)) / n_sim,
            "top5_prob": sum(position_counts[team][p] for p in range(1, 6)) / n_sim,
            "bottom3_prob": sum(position_counts[team][p] for p in range(n_teams - 2, n_teams + 1)) / n_sim,
            "avg_points": points_sum[team] / n_sim,
            "avg_gf": gf_sum[team] / n_sim,
            "avg_ga": ga_sum[team] / n_sim,
            "avg_gd": gd_sum[team] / n_sim,
            "elo": context["elo_ratings"].get(team, INITIAL_ELO),
            "prev_games_j1": prev_games_by_team.get(team),
            "effective_prev_weight": prev_weight_by_team.get(team),
            "prev_stats_league": meta.get("prev_stats_league", ""),
            "prev_stats_attack_ratio": meta.get("prev_stats_attack_ratio", np.nan),
            "prev_stats_defense_bad_ratio": meta.get("prev_stats_defense_bad_ratio", np.nan),
        })

    pred_df = pd.DataFrame(rows).sort_values(["season", "avg_pred_position"]).reset_index(drop=True)
    pred_df.insert(0, "pred_rank", pred_df.groupby("season").cumcount() + 1)
    mae = float(pred_df["position_error"].mean())

    if return_prediction:
        return mae, pred_df
    return mae, None


def evaluate_stats_only_config(context, team_stats_df, attack_weights, defense_weights, prev_weight, n_sim, seed, return_prediction=False):
    prev_stats_strengths, prev_stats_meta = build_prev_stats_only_strengths(
        team_stats_df=team_stats_df,
        target_year=context["target_year"],
        teams=context["teams"],
        attack_weights=attack_weights,
        defense_weights=defense_weights,
    )

    strengths, prev_weight_by_team, prev_games_by_team = blend_current_with_previous(
        current_strengths=context["current_strengths"],
        previous_strengths=prev_stats_strengths,
        teams=context["teams"],
        prev_weight=prev_weight,
        prev_stats_meta=prev_stats_meta,
        previous_df=context["previous_df"],
        promoted_prev_zero=False,
    )

    return simulate_with_strengths(
        context=context,
        strengths=strengths,
        prev_weight_by_team=prev_weight_by_team,
        prev_games_by_team=prev_games_by_team,
        prev_stats_meta=prev_stats_meta,
        n_sim=n_sim,
        seed=seed,
        return_prediction=return_prediction,
    )


def evaluate_v14_baseline(context, n_sim, seed, return_prediction=False):
    strengths, prev_weight_by_team, prev_games_by_team = blend_current_with_previous(
        current_strengths=context["current_strengths"],
        previous_strengths=context["previous_strengths_base"],
        teams=context["teams"],
        prev_weight=PREV_WEIGHT_BASELINE,
        prev_stats_meta=None,
        previous_df=context["previous_df"],
        promoted_prev_zero=True,
    )

    return simulate_with_strengths(
        context=context,
        strengths=strengths,
        prev_weight_by_team=prev_weight_by_team,
        prev_games_by_team=prev_games_by_team,
        prev_stats_meta={},
        n_sim=n_sim,
        seed=seed,
        return_prediction=return_prediction,
    )


# =========================
# 11. HTML出力
# =========================

def export_prediction_html(df, output_path, title="J1 順位予測 ver.1.5 前年スタッツのみ"):
    display_df = df.copy()
    percent_cols = ["champion_prob", "top3_prob", "top5_prob", "bottom3_prob"]
    for col in percent_cols:
        if col in display_df.columns:
            display_df[col] = (display_df[col] * 100).round(1).astype(str) + "%"

    round_cols = [
        "avg_pred_position", "position_error", "avg_points", "avg_gf", "avg_ga", "avg_gd", "elo",
        "prev_stats_attack_ratio", "prev_stats_defense_bad_ratio", "effective_prev_weight",
    ]
    for col in round_cols:
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(display_df[col], errors="coerce").round(3)

    column_names = {
        "pred_rank": "予測順位",
        "season": "年度",
        "team": "チーム",
        "actual_position": "実順位",
        "avg_pred_position": "平均予測順位",
        "position_error": "順位誤差",
        "prev_games_j1": "前年J1試合数",
        "effective_prev_weight": "有効前年重み",
        "prev_stats_league": "前年リーグ",
        "prev_stats_attack_ratio": "前年攻撃スタッツ係数",
        "prev_stats_defense_bad_ratio": "前年守備悪化係数",
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
    th, td {{ border: 1px solid #ddd; padding: 8px 10px; text-align: center; white-space: nowrap; }}
    th {{ background: #222; color: white; }}
    tr:nth-child(even) {{ background: #f5f5f5; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="note">
    <p>
      前年公式チームスタッツから前年レーティングを作成し、当年前半の攻守係数と混合した予測結果です。
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
# 12. メイン
# =========================

def main():
    team_stats_path = find_file(TEAM_STATS_CSV_NAME, required=True)
    team_stats_df = load_team_stats_csv(team_stats_path)

    print("\n==============================")
    print("ver.1.5 前年スタッツのみレーティング グリッドサーチ")
    print("==============================")
    print("TEAM_STATS_CSV:", team_stats_path)
    print("TARGET_YEARS:", TARGET_YEARS)
    print("N_SIM_SEARCH:", N_SIM_SEARCH)
    print("N_SIM_FINAL:", N_SIM_FINAL)
    print("GOAL_ADJUST_MODE:", GOAL_ADJUST_MODE)
    print("PREV_WEIGHT_CANDIDATES:", PREV_WEIGHT_CANDIDATES)
    print("v1.4 baseline PREV_WEIGHT:", PREV_WEIGHT_BASELINE)

    contexts = []
    for year in TARGET_YEARS:
        ctx = prepare_season_context(year, team_stats_df)
        if ctx is not None:
            contexts.append(ctx)
            print(f"[ok] {year}: teams={len(ctx['teams'])}, train={len(ctx['train_df'])}, test={len(ctx['test_df'])}")

    if not contexts:
        raise RuntimeError("検証できる年度がありません。j1_YYYY_match_stats_merged_fixed.csv を確認してください。")

    # v1.4 baselineを探索時と同じN_SIMで比較
    baseline_search_maes = []
    for ctx in contexts:
        seed = RANDOM_SEED + ctx["target_year"] * 1000
        mae, _ = evaluate_v14_baseline(ctx, n_sim=N_SIM_SEARCH, seed=seed, return_prediction=False)
        baseline_search_maes.append((ctx["target_year"], mae))

    print("\n[v1.4 baseline / search sim]")
    print("year_maes:", baseline_search_maes)
    print("mean_mae:", round(float(np.mean([m for _, m in baseline_search_maes])), 4))

    configs = []
    for prev_weight, attack_weights, defense_weights in product(
        PREV_WEIGHT_CANDIDATES,
        ATTACK_WEIGHT_CANDIDATES,
        DEFENSE_WEIGHT_CANDIDATES,
    ):
        # prev_weight=0では攻守重みが結果に影響しないので、最初の1件だけ残す
        if prev_weight == 0.0:
            if attack_weights != ATTACK_WEIGHT_CANDIDATES[0] or defense_weights != DEFENSE_WEIGHT_CANDIDATES[0]:
                continue
        configs.append((prev_weight, attack_weights, defense_weights))

    print("config数:", len(configs))

    grid_rows = []
    for idx, (prev_weight, attack_weights, defense_weights) in enumerate(configs, start=1):
        year_maes = []
        for ctx in contexts:
            seed = RANDOM_SEED + ctx["target_year"] * 1000
            mae, _ = evaluate_stats_only_config(
                context=ctx,
                team_stats_df=team_stats_df,
                attack_weights=attack_weights,
                defense_weights=defense_weights,
                prev_weight=prev_weight,
                n_sim=N_SIM_SEARCH,
                seed=seed,
                return_prediction=False,
            )
            year_maes.append((ctx["target_year"], mae))

        maes = [m for _, m in year_maes]
        row = {
            "config_id": idx,
            "prev_rating_mode": "official_stats_only",
            "prev_weight": prev_weight,
            "attack_goals": attack_weights["goals"],
            "attack_xg": attack_weights["xg"],
            "attack_sot": attack_weights["sot"],
            "attack_shots": attack_weights["shots"],
            "defense_ga": defense_weights["ga"],
            "defense_xga": defense_weights["xga"],
            "defense_sota": defense_weights["sota"],
            "defense_sa": defense_weights["sa"],
            "mean_mae": float(np.mean(maes)),
            "std_mae": float(np.std(maes, ddof=0)),
            "max_mae": float(np.max(maes)),
            "baseline_v14_mean_mae_search": float(np.mean([m for _, m in baseline_search_maes])),
            "delta_vs_baseline_search": float(np.mean(maes) - np.mean([m for _, m in baseline_search_maes])),
        }
        for year, mae in year_maes:
            row[f"mae_{year}"] = mae
        grid_rows.append(row)

        if idx % 10 == 0 or idx == len(configs):
            print(f"{idx}/{len(configs)} configs 完了")

    grid_df = pd.DataFrame(grid_rows).sort_values(["mean_mae", "std_mae", "max_mae"]).reset_index(drop=True)
    grid_df.to_csv(OUTPUT_GRID_CSV, index=False, encoding="utf-8-sig")

    print("\n==============================")
    print("前年スタッツのみ グリッドサーチ結果 TOP10")
    print("==============================")
    print(grid_df.head(10).to_string(index=False))
    print("CSV:", OUTPUT_GRID_CSV)

    if RUN_BASELINE_FINAL:
        all_baseline_pred = []
        baseline_final_maes = []
        for ctx in contexts:
            seed = RANDOM_SEED + ctx["target_year"] * 1000
            mae, pred_df = evaluate_v14_baseline(ctx, n_sim=N_SIM_FINAL, seed=seed, return_prediction=True)
            baseline_final_maes.append((ctx["target_year"], mae))
            pred_df.insert(1, "model", "v1.4_match_prev_baseline")
            all_baseline_pred.append(pred_df)
        baseline_pred_df = pd.concat(all_baseline_pred, ignore_index=True)
        baseline_pred_df.to_csv(OUTPUT_BASELINE_CSV, index=False, encoding="utf-8-sig")
        export_prediction_html(baseline_pred_df, OUTPUT_BASELINE_HTML, title="J1 順位予測 v1.4 baseline 前年試合別レーティング")

        print("\n==============================")
        print("v1.4 baselineをN_SIM_FINALで再評価")
        print("==============================")
        print("baseline_final_maes:", baseline_final_maes)
        print("mean_baseline_final_mae:", round(float(np.mean([m for _, m in baseline_final_maes])), 4))
        print("BASELINE CSV:", OUTPUT_BASELINE_CSV)
        print("BASELINE HTML:", OUTPUT_BASELINE_HTML)

    if RUN_FINAL_BEST:
        best = grid_df.iloc[0]
        best_attack = {
            "goals": float(best["attack_goals"]),
            "xg": float(best["attack_xg"]),
            "sot": float(best["attack_sot"]),
            "shots": float(best["attack_shots"]),
        }
        best_defense = {
            "ga": float(best["defense_ga"]),
            "xga": float(best["defense_xga"]),
            "sota": float(best["defense_sota"]),
            "sa": float(best["defense_sa"]),
        }
        best_prev_weight = float(best["prev_weight"])

        all_pred = []
        final_maes = []
        for ctx in contexts:
            seed = RANDOM_SEED + ctx["target_year"] * 1000
            mae, pred_df = evaluate_stats_only_config(
                context=ctx,
                team_stats_df=team_stats_df,
                attack_weights=best_attack,
                defense_weights=best_defense,
                prev_weight=best_prev_weight,
                n_sim=N_SIM_FINAL,
                seed=seed,
                return_prediction=True,
            )
            final_maes.append((ctx["target_year"], mae))
            pred_df.insert(1, "model", "v1.5_prev_stats_only")
            pred_df.insert(2, "prev_weight", best_prev_weight)
            pred_df.insert(3, "attack_weights", str(best_attack))
            pred_df.insert(4, "defense_weights", str(best_defense))
            all_pred.append(pred_df)

        best_pred_df = pd.concat(all_pred, ignore_index=True)
        best_pred_df.to_csv(OUTPUT_BEST_CSV, index=False, encoding="utf-8-sig")
        export_prediction_html(best_pred_df, OUTPUT_BEST_HTML)

        print("\n==============================")
        print("最良の前年スタッツのみ設定をN_SIM_FINALで再評価")
        print("==============================")
        print("best config_id:", int(best["config_id"]))
        print("prev_weight:", best_prev_weight)
        print("attack:", best_attack)
        print("defense:", best_defense)
        print("final_maes:", final_maes)
        print("mean_final_mae:", round(float(np.mean([m for _, m in final_maes])), 4))
        print("BEST CSV:", OUTPUT_BEST_CSV)
        print("BEST HTML:", OUTPUT_BEST_HTML)


if __name__ == "__main__":
    main()
