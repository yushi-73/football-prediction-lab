import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path

# =========================
# 1. 設定
# =========================

BASE_DIR = Path(__file__).resolve().parent

CURRENT_CSV = BASE_DIR / "j1_2025_match_stats_merged_fixed.csv"
PREVIOUS_CSV = BASE_DIR / "j1_2024_match_stats_merged_fixed.csv"

OUTPUT_SUMMARY_CSV = BASE_DIR / "elo_validation_summary.csv"
OUTPUT_BEST_CSV = BASE_DIR / "elo_best_prediction.csv"

# 現在年は枠内シュート、前年は総シュートを使う
SOT_WEIGHT = 0.1
PREV_SHOT_WEIGHT = 0.1
PREV_WEIGHT = 0.4

# シミュレーション設定
N_SIM = 1000
DECAY = 1.0
LAMBDA_CAP = 3.5
RANDOM_SEED = 42

# Elo検証設定
INITIAL_ELO = 1500
K_FACTOR_LIST = [8, 12, 16, 20]
HOME_ADV_LIST = [0, 25, 50]

ELO_LAMBDA_WEIGHT_LIST = [0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2]

ELO_FACTOR_SCALE = 4000

ELO_FACTOR_MIN = 0.90
ELO_FACTOR_MAX = 1.10


# =========================
# 2. データ準備
# =========================

def clean_team_names(df):
    df = df.copy()
    for col in ["home", "away"]:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("【公式】", "", regex=False)
            .str.strip()
        )
    return df


def load_match_csv(path):
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

    # 総シュート数
    if "home_shots_official" in df.columns:
        df["home_shots"] = pd.to_numeric(df["home_shots_official"], errors="coerce")
        df["away_shots"] = pd.to_numeric(df["away_shots_official"], errors="coerce")
    elif "home_shots" in df.columns:
        df["home_shots"] = pd.to_numeric(df["home_shots"], errors="coerce")
        df["away_shots"] = pd.to_numeric(df["away_shots"], errors="coerce")
    else:
        raise ValueError("総シュート数の列が見つかりません。home_shots_official を確認してください。")

    # 枠内シュート数。2024以前は欠損があってもよい
    if "home_shots_on_target" in df.columns:
        df["home_shots_on_target"] = pd.to_numeric(df["home_shots_on_target"], errors="coerce")
        df["away_shots_on_target"] = pd.to_numeric(df["away_shots_on_target"], errors="coerce")
    else:
        df["home_shots_on_target"] = np.nan
        df["away_shots_on_target"] = np.nan

    df = df.dropna(subset=["home_goal", "away_goal"]).copy()
    df = df.sort_values("date").reset_index(drop=True)

    return df


# =========================
# 3. 順位表
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
# 4. 攻守係数
# =========================

def calculate_strengths_home_away(
    history_df,
    teams,
    decay=1.0,
    feature_weight=0.1,
    feature_type="sot"
):
    """
    feature_type:
        "sot"   : 枠内シュート数を使う
        "shots" : 総シュート数を使う
        "none"  : 得点だけを使う
    """
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
        if home_feature_col not in history_df.columns or away_feature_col not in history_df.columns:
            raise ValueError(f"{home_feature_col} または {away_feature_col} がありません。")

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
        # ホーム成績
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

        # アウェイ成績
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
# 5. Eloレーティング
# =========================

def update_elo_ratings(match_df, initial_ratings, k_factor=20, home_adv=50):
    ratings = initial_ratings.copy()

    for _, row in match_df.sort_values("date").iterrows():
        home = row["home"]
        away = row["away"]
        hg = int(row["home_goal"])
        ag = int(row["away_goal"])

        if home not in ratings:
            ratings[home] = INITIAL_ELO
        if away not in ratings:
            ratings[away] = INITIAL_ELO

        home_rating = ratings[home]
        away_rating = ratings[away]

        expected_home = 1 / (1 + 10 ** ((away_rating - (home_rating + home_adv)) / 400))

        if hg > ag:
            actual_home = 1.0
        elif hg < ag:
            actual_home = 0.0
        else:
            actual_home = 0.5

        goal_diff = abs(hg - ag)

        if goal_diff <= 1:
            margin_multiplier = 1.0
        else:
            margin_multiplier = np.log(goal_diff) + 1.0

        change = k_factor * margin_multiplier * (actual_home - expected_home)

        ratings[home] = home_rating + change
        ratings[away] = away_rating - change

    return ratings


def build_elo_ratings(previous_df, train_df, teams, k_factor, home_adv):
    # 全チーム1500スタート
    all_teams = pd.unique(
        pd.concat([
            previous_df[["home", "away"]],
            train_df[["home", "away"]]
        ]).values.ravel()
    )

    ratings = {team: INITIAL_ELO for team in all_teams}

    # 2024全試合で更新
    ratings = update_elo_ratings(
        previous_df,
        ratings,
        k_factor=k_factor,
        home_adv=home_adv
    )

    # 2025前半戦でさらに更新
    ratings = update_elo_ratings(
        train_df,
        ratings,
        k_factor=k_factor,
        home_adv=home_adv
    )

    # 2025所属チームに限定して返す
    return {
        team: ratings.get(team, INITIAL_ELO)
        for team in teams
    }


# =========================
# 6. 期待得点
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
    elo_lambda_weight=0.0
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

    # Elo差をlambdaに反映
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

    return safe_lambda(lambda_home), safe_lambda(lambda_away)


# =========================
# 7. 1設定の検証
# =========================

def run_one_setting(
    current_df,
    previous_df,
    train_df,
    test_df,
    teams,
    k_factor,
    home_adv,
    elo_lambda_weight
):
    # 設定ごとに同じ乱数から始めて、比較しやすくする
    if RANDOM_SEED is not None:
        np.random.seed(RANDOM_SEED)

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
        k_factor=k_factor,
        home_adv=home_adv
    )

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
                home,
                away,
                strengths,
                home_avg_goals,
                away_avg_goals,
                elo_ratings=elo_ratings,
                elo_lambda_weight=elo_lambda_weight
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
            "bottom3_prob": sum(position_counts[team][p] for p in range(len(teams)-2, len(teams)+1)) / N_SIM,
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

    return mae, result_df


# =========================
# 8. 全設定を検証
# =========================

def main():
    current_df = load_match_csv(CURRENT_CSV)
    previous_df = load_match_csv(PREVIOUS_CSV)

    teams = list(pd.unique(current_df[["home", "away"]].values.ravel()))

    split = int(len(current_df) * 0.5)
    train_df = current_df.iloc[:split].copy()
    test_df = current_df.iloc[split:].copy()

    all_results = []
    best_mae = None
    best_result_df = None
    best_setting = None

    total = (
        len(K_FACTOR_LIST)
        * len(HOME_ADV_LIST)
        * len(ELO_LAMBDA_WEIGHT_LIST)
    )
    count = 0

    for k_factor in K_FACTOR_LIST:
        for home_adv in HOME_ADV_LIST:
            for elo_lambda_weight in ELO_LAMBDA_WEIGHT_LIST:
                count += 1

                print("\n==============================")
                print(f"{count}/{total} 検証中")
                print(
                    f"K={k_factor}, HOME_ADV={home_adv}, "
                    f"ELO_LAMBDA_WEIGHT={elo_lambda_weight}"
                )
                print("==============================")

                mae, result_df = run_one_setting(
                    current_df=current_df,
                    previous_df=previous_df,
                    train_df=train_df,
                    test_df=test_df,
                    teams=teams,
                    k_factor=k_factor,
                    home_adv=home_adv,
                    elo_lambda_weight=elo_lambda_weight
                )

                print("MAE:", round(mae, 4))

                all_results.append({
                    "k_factor": k_factor,
                    "home_adv": home_adv,
                    "elo_lambda_weight": elo_lambda_weight,
                    "mae": mae
                })

                if best_mae is None or mae < best_mae:
                    best_mae = mae
                    best_result_df = result_df.copy()
                    best_setting = {
                        "k_factor": k_factor,
                        "home_adv": home_adv,
                        "elo_lambda_weight": elo_lambda_weight,
                        "mae": mae
                    }

    summary_df = pd.DataFrame(all_results).sort_values("mae").reset_index(drop=True)
    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    print("\n==============================")
    print("=== Elo検証まとめ ===")
    print("==============================")
    print(summary_df)

    print("\n==============================")
    print("=== 最良設定 ===")
    print("==============================")
    print(best_setting)

    if best_result_df is not None:
        best_result_df.to_csv(OUTPUT_BEST_CSV, index=False, encoding="utf-8-sig")
        print("\n最良設定の順位予測を保存しました:")
        print(OUTPUT_BEST_CSV)

    print("\n検証結果一覧を保存しました:")
    print(OUTPUT_SUMMARY_CSV)


if __name__ == "__main__":
    main()
