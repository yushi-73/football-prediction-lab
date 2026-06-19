import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path

# ============================================================
# J1 前年レーティング係数 検証用 簡易版
# ------------------------------------------------------------
# 目的:
#   ・前年係数 PREV_WEIGHT / PREV_DECAY を複数年で検証する
#   ・検証年: 2022→2023, 2023→2024, 2024→2025
#   ・1993-2025のJ1履歴CSVから、対象年と前年を抽出する
#   ・λには得点のみを使用し、枠内シュート数・シュート数は使わない
#   ・優勝ブースト、残留ブースト、Elo補正、相性Effectは使わない
#
# 検証方法:
#   1. target_yearの前半戦を学習データにする
#   2. target_yearの後半戦をシミュレーション対象にする
#   3. previous_yearの全試合から前年レーティングを作る
#   4. target_year前半の攻守係数と前年攻守係数をブレンドする
#   5. target_year最終順位に対する平均順位誤差MAEを計算する
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

# 前年レーティング係数候補
# 必要に応じて増減してください。
PREV_WEIGHT_CANDIDATES = [
    0.00, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80,
]

PREV_DECAY_CANDIDATES = [
    1.000, 0.995, 0.990, 0.985, 0.980, 0.970, 0.950,
]

# 現年の前半戦係数は、まずは減衰なしに固定
CURRENT_DECAY = 1.0

# 昇格組補正: 前年J1にいないチームは前年由来の中立値1.0を混ぜない
USE_PROMOTED_PREV_ZERO = True
PROMOTED_PREV_WEIGHT = 0.0

# シミュレーション設定
N_SIM_SEARCH = 1000
RANDOM_SEED = 42
LAMBDA_CAP = 3.5

# 出力
OUTPUT_DETAIL_CSV = BASE_DIR / "prev_weight_decay_simple_multiyear_detail.csv"
OUTPUT_SUMMARY_CSV = BASE_DIR / "prev_weight_decay_simple_multiyear_summary.csv"
OUTPUT_SUMMARY_HTML = BASE_DIR / "prev_weight_decay_simple_multiyear_summary.html"
OUTPUT_BEST_PREDICTION_CSV = BASE_DIR / "prev_weight_decay_simple_multiyear_best_predictions.csv"


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

    # date列
    if "date" not in df.columns:
        raise ValueError("CSVに date 列が必要です。")

    df["date"] = (
        df["date"]
        .astype(str)
        .str.replace(r"\(.*\)", "", regex=True)
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 得点列: home_goal / away_goal または home_goals / away_goals に対応
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
# 5. 攻守係数 簡易版: 得点のみ
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


def calculate_strengths_home_away_goals_only(history_df, teams, decay=1.0):
    """
    得点のみでH/A別の攻撃係数・守備係数を作る。
    シュート数・枠内シュート数は一切使わない。

    defense係数は「大きいほど失点しやすい」として扱う。
    例:
      lambda_home = home_attack * away_defense * home_avg_goals
    """
    history_df = history_df.copy().sort_values("date").reset_index(drop=True)

    home_avg_goals = safe_positive_mean(history_df["home_goal"], fallback=1.0)
    away_avg_goals = safe_positive_mean(history_df["away_goal"], fallback=1.0)

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
            home_gf += float(row_dict["home_goal"]) * weight
            home_ga += float(row_dict["away_goal"]) * weight
            home_w += weight

        if home_w > 0:
            home_attack = (home_gf / home_w) / home_avg_goals
            home_defense = (home_ga / home_w) / away_avg_goals
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
            away_gf += float(row_dict["away_goal"]) * weight
            away_ga += float(row_dict["home_goal"]) * weight
            away_w += weight

        if away_w > 0:
            away_attack = (away_gf / away_w) / away_avg_goals
            away_defense = (away_ga / away_w) / home_avg_goals
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


# =========================
# 6. 期待得点・シミュレーション
# =========================


def safe_lambda(lam):
    if lam is None or not np.isfinite(lam) or lam < 0:
        return 0.05
    if LAMBDA_CAP is not None:
        lam = min(lam, LAMBDA_CAP)
    return float(max(lam, 0.05))


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
    return safe_lambda(lambda_home), safe_lambda(lambda_away)


def simulate_target_year(
    historical_df,
    target_year,
    prev_weight,
    prev_decay,
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

    # 現年前半: 得点のみ、減衰なし
    current_strengths, home_avg_goals, away_avg_goals = calculate_strengths_home_away_goals_only(
        train_df,
        teams=teams,
        decay=CURRENT_DECAY,
    )

    # 前年: 得点のみ、prev_decay候補で作る
    previous_strengths, _, _ = calculate_strengths_home_away_goals_only(
        previous_df,
        teams=teams,
        decay=prev_decay,
    )

    strengths, prev_weight_by_team, prev_games_by_team = blend_with_previous_strengths(
        current_strengths=current_strengths,
        previous_strengths=previous_strengths,
        prev_weight=prev_weight,
        previous_df=previous_df,
        use_promoted_prev_zero=USE_PROMOTED_PREV_ZERO,
        promoted_prev_weight=PROMOTED_PREV_WEIGHT,
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
            )

            hg = int(np.random.poisson(lambda_home))
            ag = int(np.random.poisson(lambda_away))

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

    # よく見るチームが存在する年だけ個別誤差を追加
    lookup = prediction_df.set_index("team")
    for team in ["京都", "柏", "川崎F", "神戸", "横浜FM", "名古屋", "清水", "町田", "G大阪", "浦和"]:
        col_name = standardize_team_name(team)
        if col_name in lookup.index:
            summary[f"{col_name}_actual_position"] = lookup.loc[col_name, "actual_position"]
            summary[f"{col_name}_avg_pred_position"] = lookup.loc[col_name, "avg_pred_position"]
            summary[f"{col_name}_error"] = lookup.loc[col_name, "position_error"]

    return summary, prediction_df


# =========================
# 7. 出力
# =========================


def export_summary_html(df, output_path):
    display_df = df.copy()

    round_cols = [
        "prev_weight", "prev_decay", "mean_mae", "std_mae", "mean_prob_actual_position",
        "mean_sim_draw_rate", "mae_2023", "mae_2024", "mae_2025",
        "delta_vs_prev000", "delta_vs_prev040_decay0995", "delta_vs_prev060_decay0985",
    ]
    for col in round_cols:
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(display_df[col], errors="coerce").round(4)

    table_html = display_df.to_html(index=False, classes="prediction-table")

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>前年レーティング係数 複数年検証</title>
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
  <h1>前年レーティング係数 複数年検証</h1>
  <div class="note">
    <p>
      2022→2023、2023→2024、2024→2025を対象に、PREV_WEIGHT / PREV_DECAYを比較しています。
      λは得点のみで作成し、シュート数・枠内シュート数・Elo・相性Effect・終盤補正は使っていません。
    </p>
    <p>
      N_SIM_SEARCH={N_SIM_SEARCH}, LAMBDA_CAP={LAMBDA_CAP}, CURRENT_DECAY={CURRENT_DECAY}。
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
# 8. main
# =========================


def main():
    historical_df = load_historical_j1_csv(HISTORICAL_CSV)

    print("\n==============================")
    print("前年レーティング係数 複数年検証 簡易版")
    print("==============================")
    print("HISTORICAL_CSV:", HISTORICAL_CSV)
    print("TARGET_YEARS:", TARGET_YEARS)
    print("PREV_WEIGHT_CANDIDATES:", PREV_WEIGHT_CANDIDATES)
    print("PREV_DECAY_CANDIDATES:", PREV_DECAY_CANDIDATES)
    print("N_SIM_SEARCH:", N_SIM_SEARCH)
    print("CURRENT_DECAY:", CURRENT_DECAY)
    print("LAMBDA_CAP:", LAMBDA_CAP)
    print("USE_PROMOTED_PREV_ZERO:", USE_PROMOTED_PREV_ZERO)
    print("PROMOTED_PREV_WEIGHT:", PROMOTED_PREV_WEIGHT)

    detail_rows = []
    all_predictions = []

    total = len(PREV_WEIGHT_CANDIDATES) * len(PREV_DECAY_CANDIDATES) * len(TARGET_YEARS)
    done = 0

    for prev_weight in PREV_WEIGHT_CANDIDATES:
        for prev_decay in PREV_DECAY_CANDIDATES:
            for target_year in TARGET_YEARS:
                done += 1
                print(
                    f"\n[{done}/{total}] "
                    f"target={target_year}, prev_weight={prev_weight:.2f}, prev_decay={prev_decay:.3f}"
                )

                summary, prediction_df = simulate_target_year(
                    historical_df=historical_df,
                    target_year=target_year,
                    prev_weight=float(prev_weight),
                    prev_decay=float(prev_decay),
                    n_sim=N_SIM_SEARCH,
                    seed=RANDOM_SEED + target_year,
                )
                detail_rows.append(summary)

                prediction_df["prev_weight"] = prev_weight
                prediction_df["prev_decay"] = prev_decay
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
        index=["prev_weight", "prev_decay"],
        columns="target_year",
        values="mae",
        aggfunc="mean",
    )
    pivot_mae.columns = [f"mae_{int(col)}" for col in pivot_mae.columns]
    pivot_mae = pivot_mae.reset_index()

    grouped = detail_df.groupby(["prev_weight", "prev_decay"], as_index=False).agg(
        mean_mae=("mae", "mean"),
        std_mae=("mae", "std"),
        mean_prob_actual_position=("mean_prob_actual_position", "mean"),
        mean_sim_draw_rate=("sim_draw_rate", "mean"),
    )

    summary_df = grouped.merge(pivot_mae, on=["prev_weight", "prev_decay"], how="left")

    # 比較基準: 前年なし / 旧候補0.40-0.995 / 新候補0.60-0.985
    def baseline_mean_mae(weight, decay):
        rows = summary_df[
            np.isclose(summary_df["prev_weight"], weight)
            & np.isclose(summary_df["prev_decay"], decay)
        ]
        if len(rows) == 0:
            return np.nan
        return float(rows.iloc[0]["mean_mae"])

    base_prev000 = baseline_mean_mae(0.00, 1.000)
    base_040_0995 = baseline_mean_mae(0.40, 0.995)
    base_060_0985 = baseline_mean_mae(0.60, 0.985)

    summary_df["delta_vs_prev000"] = summary_df["mean_mae"] - base_prev000
    summary_df["delta_vs_prev040_decay0995"] = summary_df["mean_mae"] - base_040_0995
    summary_df["delta_vs_prev060_decay0985"] = summary_df["mean_mae"] - base_060_0985

    summary_df = summary_df.sort_values(["mean_mae", "std_mae"]).reset_index(drop=True)
    summary_df.insert(0, "rank", summary_df.index + 1)
    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    export_summary_html(summary_df, OUTPUT_SUMMARY_HTML)

    best_weight = float(summary_df.iloc[0]["prev_weight"])
    best_decay = float(summary_df.iloc[0]["prev_decay"])

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    best_predictions_df = predictions_df[
        np.isclose(predictions_df["prev_weight"], best_weight)
        & np.isclose(predictions_df["prev_decay"], best_decay)
    ].copy()
    best_predictions_df.to_csv(OUTPUT_BEST_PREDICTION_CSV, index=False, encoding="utf-8-sig")

    print("\n==============================")
    print("検証結果 上位")
    print("==============================")
    show_cols = [
        "rank", "prev_weight", "prev_decay", "mean_mae", "std_mae",
        "mean_prob_actual_position", "mean_sim_draw_rate",
        "mae_2023", "mae_2024", "mae_2025",
        "delta_vs_prev000", "delta_vs_prev040_decay0995", "delta_vs_prev060_decay0985",
    ]
    existing_show_cols = [col for col in show_cols if col in summary_df.columns]
    print(summary_df[existing_show_cols].head(20).to_string(index=False))

    print("\nDETAIL CSV:", OUTPUT_DETAIL_CSV)
    print("SUMMARY CSV:", OUTPUT_SUMMARY_CSV)
    print("SUMMARY HTML:", OUTPUT_SUMMARY_HTML)
    print("BEST PREDICTION CSV:", OUTPUT_BEST_PREDICTION_CSV)

    print("\n最良候補:")
    print("PREV_WEIGHT:", best_weight)
    print("PREV_DECAY:", best_decay)
    print("mean_mae:", round(float(summary_df.iloc[0]["mean_mae"]), 4))


if __name__ == "__main__":
    main()
