
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
OUTPUT_CSV = BASE_DIR / "j1_2025_prediction_prev04.csv"

# 2025は枠内シュート、2024は総シュートを使う
SOT_WEIGHT = 0.1          # 2025用：枠内シュート数の重み
PREV_SHOT_WEIGHT = 0.1    # 2024用：総シュート数の重み
PREV_WEIGHT = 0.4         # 今回の検証で最良だった前年レーティング重み

N_SIM = 10000             
DECAY = 1.0
LAMBDA_CAP = 3.5
RANDOM_SEED = None         
OUTPUT_CSV = "j1_2025_prediction_prev04.csv"


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


def prepare_match_df(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = clean_team_names(df)

    df["date"] = (
        df["date"]
        .astype(str)
        .str.replace(r"\(.*\)", "", regex=True)
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df["home_goal"] = pd.to_numeric(df["home_goals"], errors="coerce")
    df["away_goal"] = pd.to_numeric(df["away_goals"], errors="coerce")

    # 2025用：枠内シュート
    if "home_shots_on_target" in df.columns:
        df["home_shots_on_target"] = pd.to_numeric(df["home_shots_on_target"], errors="coerce")
    if "away_shots_on_target" in df.columns:
        df["away_shots_on_target"] = pd.to_numeric(df["away_shots_on_target"], errors="coerce")

    # 2024以前用：総シュート数。official列があればそちらを優先
    if "home_shots_official" in df.columns:
        df["home_shots"] = pd.to_numeric(df["home_shots_official"], errors="coerce")
    elif "home_shots" in df.columns:
        df["home_shots"] = pd.to_numeric(df["home_shots"], errors="coerce")

    if "away_shots_official" in df.columns:
        df["away_shots"] = pd.to_numeric(df["away_shots_official"], errors="coerce")
    elif "away_shots" in df.columns:
        df["away_shots"] = pd.to_numeric(df["away_shots"], errors="coerce")

    df = df.dropna(subset=["date", "home", "away", "home_goal", "away_goal"]).copy()
    df = df.sort_values("date").reset_index(drop=True)
    return df


# =========================
# 3. 順位表関連
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

    for team in teams:
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
# 4. チーム力計算
# =========================

def calculate_strengths_home_away(
    history_df,
    teams,
    decay=1.0,
    feature_weight=0.3,
    feature_type="sot"
):
    """
    feature_type:
        "sot"   -> 枠内シュート数を使う
        "shots" -> 総シュート数を使う
        "none"  -> 得点だけを使う
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
        raise ValueError("feature_type は 'sot', 'shots', 'none' のどれかにしてください")

    use_feature = feature_weight > 0 and home_feature_col is not None
    goal_weight = 1.0 - feature_weight

    history_df["home_goal"] = pd.to_numeric(history_df["home_goal"], errors="coerce")
    history_df["away_goal"] = pd.to_numeric(history_df["away_goal"], errors="coerce")
    history_df = history_df.dropna(subset=["home_goal", "away_goal"]).copy()

    if use_feature:
        if home_feature_col not in history_df.columns or away_feature_col not in history_df.columns:
            raise ValueError(
                f"{feature_type} を使おうとしましたが、"
                f"{home_feature_col} または {away_feature_col} がCSVにありません"
            )

        history_df[home_feature_col] = pd.to_numeric(history_df[home_feature_col], errors="coerce")
        history_df[away_feature_col] = pd.to_numeric(history_df[away_feature_col], errors="coerce")

    home_avg_goals = history_df["home_goal"].mean()
    away_avg_goals = history_df["away_goal"].mean()

    if not np.isfinite(home_avg_goals) or home_avg_goals <= 0:
        home_avg_goals = 1.0
    if not np.isfinite(away_avg_goals) or away_avg_goals <= 0:
        away_avg_goals = 1.0

    if use_feature:
        home_avg_feature = history_df[home_feature_col].mean()
        away_avg_feature = history_df[away_feature_col].mean()

        if not np.isfinite(home_avg_feature) or home_avg_feature <= 0:
            home_avg_feature = 1.0
        if not np.isfinite(away_avg_feature) or away_avg_feature <= 0:
            away_avg_feature = 1.0

        # 欠損はリーグ平均で補完。2024以前は総シュートなので基本は欠損しない想定。
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
            weight = decay ** i
            row_dict = row._asdict()

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
            weight = decay ** i
            row_dict = row._asdict()

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

    def safe_blend(current_value, previous_value, weight):
        current_ok = np.isfinite(current_value)
        previous_ok = np.isfinite(previous_value)

        if current_ok and previous_ok:
            return float((1 - weight) * current_value + weight * previous_value)
        if current_ok:
            return float(current_value)
        if previous_ok:
            return float(previous_value)
        return 1.0

    for team in current_strengths:
        if team in previous_strengths:
            blended[team] = {
                "home_attack": safe_blend(current_strengths[team]["home_attack"], previous_strengths[team]["home_attack"], prev_weight),
                "home_defense": safe_blend(current_strengths[team]["home_defense"], previous_strengths[team]["home_defense"], prev_weight),
                "away_attack": safe_blend(current_strengths[team]["away_attack"], previous_strengths[team]["away_attack"], prev_weight),
                "away_defense": safe_blend(current_strengths[team]["away_defense"], previous_strengths[team]["away_defense"], prev_weight),
            }
        else:
            blended[team] = current_strengths[team]

    return blended


def expected_goals_home_away(home, away, strengths, home_avg_goals, away_avg_goals):
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

    def safe_lambda(lam):
        if lam is None or not np.isfinite(lam) or lam < 0:
            return 0.05
        if LAMBDA_CAP is not None:
            lam = min(lam, LAMBDA_CAP)
        return float(max(lam, 0.05))

    return safe_lambda(lambda_home), safe_lambda(lambda_away)


# =========================
# 5. 予測実行
# =========================

def predict_final_table():
    if RANDOM_SEED is not None:
        np.random.seed(RANDOM_SEED)

    df = prepare_match_df(CURRENT_CSV)
    prev_df = prepare_match_df(PREVIOUS_CSV)

    teams = pd.unique(df[["home", "away"]].values.ravel())
    n_teams = len(teams)

    # 2025全日程の前半50%を学習、後半50%を予測
    split = int(len(df) * 0.5)
    train_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()

    print("==============================")
    print("順位予測設定")
    print("==============================")
    print(f"使用CSV: {CURRENT_CSV}")
    print(f"前年CSV: {PREVIOUS_CSV}")
    print(f"学習試合数: {len(train_df)}")
    print(f"予測試合数: {len(test_df)}")
    print(f"学習期間: {train_df['date'].min().date()} ～ {train_df['date'].max().date()}")
    print(f"予測期間: {test_df['date'].min().date()} ～ {test_df['date'].max().date()}")
    print(f"SOT_WEIGHT: {SOT_WEIGHT}")
    print(f"PREV_SHOT_WEIGHT: {PREV_SHOT_WEIGHT}")
    print(f"PREV_WEIGHT: {PREV_WEIGHT}")
    print(f"N_SIM: {N_SIM}")
    print()

    current_strengths, home_avg_goals, away_avg_goals = calculate_strengths_home_away(
        train_df,
        teams,
        decay=DECAY,
        feature_weight=SOT_WEIGHT,
        feature_type="sot"
    )

    previous_strengths, _, _ = calculate_strengths_home_away(
        prev_df,
        teams,
        decay=DECAY,
        feature_weight=PREV_SHOT_WEIGHT,
        feature_type="shots"
    )

    strengths = blend_with_previous_strengths(
        current_strengths,
        previous_strengths,
        PREV_WEIGHT
    )

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
                away_avg_goals
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

        for pos, (team, stats) in enumerate(ranking, start=1):
            position_counts[team][pos] += 1
            points_sum[team] += stats["points"]
            gf_sum[team] += stats["gf"]
            ga_sum[team] += stats["ga"]
            gd_sum[team] += stats["gd"]

        if sim % 500 == 0:
            print(f"{sim}/{N_SIM} 回終了")

    # 実際順位。2025全試合が入っているCSVの場合だけ比較に使う。
    actual_table = calculate_table(df, teams)
    actual_ranking = make_ranking(actual_table)
    actual_position = {
        team: pos
        for pos, (team, _) in enumerate(actual_ranking, start=1)
    }

    rows = []
    for team in teams:
        avg_pred_pos = sum(
            pos * (position_counts[team][pos] / N_SIM)
            for pos in range(1, n_teams + 1)
        )
        most_likely_position = position_counts[team].most_common(1)[0][0]

        rows.append({
            "pred_rank": None,  # あとで入れる
            "team": team,
            "avg_pred_position": avg_pred_pos,
            "most_likely_position": most_likely_position,
            "avg_points": points_sum[team] / N_SIM,
            "avg_gf": gf_sum[team] / N_SIM,
            "avg_ga": ga_sum[team] / N_SIM,
            "avg_gd": gd_sum[team] / N_SIM,
            "champion_prob": position_counts[team][1] / N_SIM,
            "top3_prob": sum(position_counts[team][p] for p in range(1, 4)) / N_SIM,
            "top5_prob": sum(position_counts[team][p] for p in range(1, 6)) / N_SIM,
            "bottom3_prob": sum(position_counts[team][p] for p in range(n_teams - 2, n_teams + 1)) / N_SIM,
            "actual_position": actual_position.get(team),
        })

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values(
        ["avg_pred_position", "avg_points", "avg_gd", "avg_gf"],
        ascending=[True, False, False, False]
    ).reset_index(drop=True)
    result_df["pred_rank"] = result_df.index + 1

    if "actual_position" in result_df.columns:
        result_df["position_error"] = (result_df["avg_pred_position"] - result_df["actual_position"]).abs()
        mae = result_df["position_error"].mean()
    else:
        mae = None

    # 見やすく丸める
    display_df = result_df.copy()
    for col in [
        "avg_pred_position", "avg_points", "avg_gf", "avg_ga", "avg_gd",
        "champion_prob", "top3_prob", "top5_prob", "bottom3_prob", "position_error"
    ]:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(4)

    display_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\n==============================")
    print("=== 予測順位表 ===")
    print("==============================")
    print(display_df.to_string(index=False))

    if mae is not None:
        print("\nMAE:", round(mae, 4))

    print(f"\nCSV出力: {OUTPUT_CSV}")

    return display_df


if __name__ == "__main__":
    predict_final_table()

