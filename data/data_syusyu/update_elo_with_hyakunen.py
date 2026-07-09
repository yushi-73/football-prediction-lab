# update_elo_with_hyakunen.py
# J1/J2/J3の2025終了時点Eloを、2026 J2・J3百年構想リーグ結果で更新する単体スクリプト
#
# 使い方:
# 1. このpyファイル、elo_final_ratings.csv、j2j3_2026_match_stats_0106_completed.csv を同じフォルダに置く
# 2. VSCodeやターミナルで python update_elo_with_hyakunen.py を実行
#
# 出力:
# - hyakunen_elo_latest.csv
# - hyakunen_elo_participants.csv
# - hyakunen_elo_history_by_match.csv

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


# =========================
# 設定
# =========================

BASE_DIR = Path(__file__).resolve().parent

# ファイル名は必要に応じて変更してください。
# まず通常名を探し、なければ (1) 付きのファイル名も探します。
ELO_INPUT_CANDIDATES = [
    "elo_final_ratings.csv",
    "elo_final_ratings(1).csv",
]

HYAKUNEN_INPUT_CANDIDATES = [
    "j2j3_2026_match_stats_0106_completed.csv",
    "j2j3_2026_match_stats_0106_completed(1).csv",
]

OUTPUT_DIR = BASE_DIR / "hyakunen_elo_output"

# Elo設定
HOME_ADVANTAGE_ELO = 50.0

# 百年構想リーグは通常リーグより弱めに更新する
K_REGIONAL = 10.0

# 2026-05-24までを地域ラウンド、2026-05-25以降をプレーオフ扱いにする
# プレーオフを同じ重みにしたい場合は K_PLAYOFF = K_REGIONAL にしてください
PLAYOFF_START_DATE = "2026-05-25"
K_PLAYOFF = 5.0

# 未登録チームが出た場合の初期値
# 基本的には今回の40チームは全部既存Eloに存在する想定
DEFAULT_NEW_TEAM_ELO = 1350.0

# 得点差補正
USE_MARGIN_MULTIPLIER = True


# 百年構想CSV側の短縮表記 → Elo側の正式表記
TEAM_ALIASES = {
    "FC大阪": "ＦＣ大阪",
    "いわき": "いわきＦＣ",
    "今治": "ＦＣ今治",
    "仙台": "ベガルタ仙台",
    "八戸": "ヴァンラーレ八戸",
    "北九州": "ギラヴァンツ北九州",
    "大分": "大分トリニータ",
    "大宮": "ＲＢ大宮アルディージャ",
    "奈良": "奈良クラブ",
    "宮崎": "テゲバジャーロ宮崎",
    "富山": "カターレ富山",
    "山口": "レノファ山口ＦＣ",
    "山形": "モンテディオ山形",
    "岐阜": "ＦＣ岐阜",
    "徳島": "徳島ヴォルティス",
    "愛媛": "愛媛ＦＣ",
    "新潟": "アルビレックス新潟",
    "札幌": "北海道コンサドーレ札幌",
    "松本": "松本山雅ＦＣ",
    "栃木C": "栃木シティ",
    "栃木SC": "栃木ＳＣ",
    "横浜FC": "横浜ＦＣ",
    "湘南": "湘南ベルマーレ",
    "滋賀": "レイラック滋賀ＦＣ",
    "熊本": "ロアッソ熊本",
    "琉球": "ＦＣ琉球",
    "甲府": "ヴァンフォーレ甲府",
    "相模原": "ＳＣ相模原",
    "磐田": "ジュビロ磐田",
    "福島": "福島ユナイテッドＦＣ",
    "秋田": "ブラウブリッツ秋田",
    "群馬": "ザスパ群馬",
    "藤枝": "藤枝ＭＹＦＣ",
    "讃岐": "カマタマーレ讃岐",
    "金沢": "ツエーゲン金沢",
    "長野": "ＡＣ長野パルセイロ",
    "高知": "高知ユナイテッドSC",
    "鳥取": "ガイナーレ鳥取",
    "鳥栖": "サガン鳥栖",
    "鹿児島": "鹿児島ユナイテッドＦＣ",
}


# =========================
# ユーティリティ
# =========================

def find_existing_file(candidates: list[str]) -> Path:
    for name in candidates:
        path = BASE_DIR / name
        if path.exists():
            return path
    joined = "\n".join(f"- {name}" for name in candidates)
    raise FileNotFoundError(f"入力ファイルが見つかりません。候補:\n{joined}")


def canonical_team_name(name: str) -> str:
    name = str(name).strip()
    return TEAM_ALIASES.get(name, name)


def expected_score_home(home_elo: float, away_elo: float, home_advantage: float) -> float:
    """
    ホーム補正込みのホーム勝率期待値。
    """
    adjusted_home = home_elo + home_advantage
    return 1.0 / (1.0 + 10.0 ** ((away_elo - adjusted_home) / 400.0))


def actual_score_home(home_goals: int, away_goals: int) -> float:
    if home_goals > away_goals:
        return 1.0
    if home_goals < away_goals:
        return 0.0
    return 0.5


def margin_multiplier(home_goals: int, away_goals: int) -> float:
    """
    サッカー用にかなり控えめな得点差補正。
    1点差は1.0、2点差以上で少しだけKを増やす。
    """
    if not USE_MARGIN_MULTIPLIER:
        return 1.0

    margin = abs(int(home_goals) - int(away_goals))
    if margin <= 1:
        return 1.0

    # 2点差: 約1.10, 3点差: 約1.39, 4点差: 約1.61
    return min(1.75, math.log1p(margin))


def get_k_value(match_date: pd.Timestamp) -> float:
    playoff_start = pd.to_datetime(PLAYOFF_START_DATE)
    if match_date >= playoff_start:
        return K_PLAYOFF
    return K_REGIONAL


def validate_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} に必要な列がありません: {missing}")


# =========================
# メイン処理
# =========================

def main() -> None:
    elo_path = find_existing_file(ELO_INPUT_CANDIDATES)
    hyakunen_path = find_existing_file(HYAKUNEN_INPUT_CANDIDATES)

    print(f"Elo input      : {elo_path.name}")
    print(f"Hyakunen input : {hyakunen_path.name}")

    elo_df = pd.read_csv(elo_path)
    hyakunen_df = pd.read_csv(hyakunen_path)

    validate_columns(
        elo_df,
        ["team", "elo"],
        "Elo CSV",
    )
    validate_columns(
        hyakunen_df,
        ["date", "home", "away", "home_goals", "away_goals"],
        "百年構想リーグCSV",
    )

    # 日付・得点の整形
    hyakunen_df = hyakunen_df.copy()
    hyakunen_df["date"] = pd.to_datetime(hyakunen_df["date"], errors="coerce")
    hyakunen_df["home_goals"] = pd.to_numeric(hyakunen_df["home_goals"], errors="coerce")
    hyakunen_df["away_goals"] = pd.to_numeric(hyakunen_df["away_goals"], errors="coerce")

    bad_rows = hyakunen_df[
        hyakunen_df["date"].isna()
        | hyakunen_df["home_goals"].isna()
        | hyakunen_df["away_goals"].isna()
    ]
    if not bad_rows.empty:
        raise ValueError(
            "百年構想リーグCSVに日付または得点が読めない行があります。\n"
            f"{bad_rows[['date', 'home', 'away', 'home_goals', 'away_goals']].head(20)}"
        )

    hyakunen_df["home"] = hyakunen_df["home"].map(canonical_team_name)
    hyakunen_df["away"] = hyakunen_df["away"].map(canonical_team_name)

    hyakunen_df["home_goals"] = hyakunen_df["home_goals"].astype(int)
    hyakunen_df["away_goals"] = hyakunen_df["away_goals"].astype(int)

    # 同日内は元CSVの順番を保持
    hyakunen_df["_original_order"] = range(len(hyakunen_df))
    hyakunen_df = hyakunen_df.sort_values(["date", "_original_order"]).reset_index(drop=True)

    # Elo辞書を作る
    ratings: dict[str, float] = {}
    matches: dict[str, int] = {}
    last_division: dict[str, str] = {}
    last_match_date: dict[str, str] = {}

    for _, row in elo_df.iterrows():
        team = str(row["team"]).strip()
        ratings[team] = float(row["elo"])
        matches[team] = int(row["matches"]) if "matches" in elo_df.columns and pd.notna(row.get("matches")) else 0
        last_division[team] = str(row["last_division"]) if "last_division" in elo_df.columns and pd.notna(row.get("last_division")) else ""
        last_match_date[team] = str(row["last_match_date"]) if "last_match_date" in elo_df.columns and pd.notna(row.get("last_match_date")) else ""

    # 入力側チームがElo側に存在するかチェック
    participants = sorted(set(hyakunen_df["home"]) | set(hyakunen_df["away"]))
    missing_teams = [t for t in participants if t not in ratings]
    if missing_teams:
        print("\nWARNING: Elo CSVに存在しないチームがあります。初期値を付与します。")
        for team in missing_teams:
            print(f"  - {team}: {DEFAULT_NEW_TEAM_ELO}  # 未登録チームはJFL昇格相当の初期値")
            ratings[team] = DEFAULT_NEW_TEAM_ELO
            matches[team] = 0
            last_division[team] = "NEW"
            last_match_date[team] = ""

    history_rows = []

    for i, row in hyakunen_df.iterrows():
        date = row["date"]
        home = row["home"]
        away = row["away"]
        hg = int(row["home_goals"])
        ag = int(row["away_goals"])

        pre_home = ratings[home]
        pre_away = ratings[away]

        expected_home = expected_score_home(pre_home, pre_away, HOME_ADVANTAGE_ELO)
        actual_home = actual_score_home(hg, ag)

        k = get_k_value(date)
        mult = margin_multiplier(hg, ag)

        delta = k * mult * (actual_home - expected_home)

        post_home = pre_home + delta
        post_away = pre_away - delta

        ratings[home] = post_home
        ratings[away] = post_away

        matches[home] = matches.get(home, 0) + 1
        matches[away] = matches.get(away, 0) + 1

        last_division[home] = "百年構想"
        last_division[away] = "百年構想"
        last_match_date[home] = date.strftime("%Y-%m-%d")
        last_match_date[away] = date.strftime("%Y-%m-%d")

        history_rows.append(
            {
                "match_no": i + 1,
                "date": date.strftime("%Y-%m-%d"),
                "stage": "playoff" if date >= pd.to_datetime(PLAYOFF_START_DATE) else "regional",
                "home": home,
                "away": away,
                "home_goals": hg,
                "away_goals": ag,
                "k": k,
                "margin_multiplier": mult,
                "home_elo_before": pre_home,
                "away_elo_before": pre_away,
                "expected_home": expected_home,
                "actual_home": actual_home,
                "elo_delta_home": delta,
                "home_elo_after": post_home,
                "away_elo_after": post_away,
            }
        )

    latest_df = pd.DataFrame(
        [
            {
                "team": team,
                "elo": elo,
                "last_division": last_division.get(team, ""),
                "matches": matches.get(team, 0),
                "last_match_date": last_match_date.get(team, ""),
                "participated_hyakunen": team in participants,
            }
            for team, elo in ratings.items()
        ]
    )

    latest_df = latest_df.sort_values("elo", ascending=False).reset_index(drop=True)
    latest_df.insert(0, "rank", latest_df.index + 1)

    participants_df = latest_df[latest_df["participated_hyakunen"]].copy().reset_index(drop=True)
    participants_df["hyakunen_rank"] = participants_df.index + 1

    history_df = pd.DataFrame(history_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    latest_path = OUTPUT_DIR / "hyakunen_elo_latest.csv"
    participants_path = OUTPUT_DIR / "hyakunen_elo_participants.csv"
    history_path = OUTPUT_DIR / "hyakunen_elo_history_by_match.csv"

    latest_df.to_csv(latest_path, index=False, encoding="utf-8-sig")
    participants_df.to_csv(participants_path, index=False, encoding="utf-8-sig")
    history_df.to_csv(history_path, index=False, encoding="utf-8-sig")

    print("\nDone.")
    print(f"Matches updated : {len(hyakunen_df)}")
    print(f"Participants    : {len(participants)}")
    print(f"Regional K      : {K_REGIONAL}")
    print(f"Playoff K       : {K_PLAYOFF}")
    print(f"Home advantage  : {HOME_ADVANTAGE_ELO}")
    print("\nOutput:")
    print(f"- {latest_path}")
    print(f"- {participants_path}")
    print(f"- {history_path}")

    print("\nTop 20 participating teams:")
    cols = ["hyakunen_rank", "team", "elo", "last_match_date"]
    print(participants_df[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
