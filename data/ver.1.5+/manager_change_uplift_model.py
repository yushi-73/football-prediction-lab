# -*- coding: utf-8 -*-
"""
Manager Change Uplift Model / 監督交代アップリフト判定モデル

目的:
  監督交代後1〜5試合の勝点上振れが、同じように低迷していた
  「監督交代なし疑似イベント」の平均回帰を上回るかを判定する。

入力:
  - manager_boost_condition_dataset.csv
  - manager_manual_labels_filled_all118.csv
  - manager_boost_true_vs_pseudo_by_timing.csv

主な出力:
  - manager_change_uplift_labeled_dataset.csv
  - manager_change_uplift_label_summary.csv
  - manager_change_uplift_combo_summary.csv
  - manager_change_uplift_rule_score_summary.csv
  - manager_change_uplift_model_metrics.csv
  - manager_change_uplift_tree_rules.txt
  - manager_change_prediction_input_template.csv

注意:
  厳密な因果推論ではなく、疑似イベント平均を控除した「追加上振れ」を
  予測・説明するための探索モデルです。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "scikit-learn が必要です。インストールされていない場合は `pip install scikit-learn` を実行してください。\n"
        f"元エラー: {e}"
    )

BASE = Path(__file__).resolve().parent

CONDITION_CSV = BASE / "manager_boost_condition_dataset.csv"
LABEL_CSV = BASE / "manager_manual_labels_filled_all118.csv"
PSEUDO_BY_TIMING_CSV = BASE / "manager_boost_true_vs_pseudo_by_timing.csv"

OUT_DATASET = BASE / "manager_change_uplift_labeled_dataset.csv"
OUT_LABEL_SUMMARY = BASE / "manager_change_uplift_label_summary.csv"
OUT_COMBO_SUMMARY = BASE / "manager_change_uplift_combo_summary.csv"
OUT_RULE_SUMMARY = BASE / "manager_change_uplift_rule_score_summary.csv"
OUT_METRICS = BASE / "manager_change_uplift_model_metrics.csv"
OUT_RULES = BASE / "manager_change_uplift_tree_rules.txt"
OUT_FEATURE_IMPORTANCE = BASE / "manager_change_uplift_feature_importance.csv"
OUT_TEMPLATE = BASE / "manager_change_prediction_input_template.csv"
OUT_HTML = BASE / "manager_change_uplift_summary.html"

WINDOWS = ["after_1_3", "after_1_5", "after_1_10"]
MAIN_WINDOW = "after_1_5"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {path}")
    return pd.read_csv(path)


def normalize_bool_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map({"1": 1, "1.0": 1, "true": 1, "yes": 1, "y": 1, "0": 0, "0.0": 0, "false": 0, "no": 0, "n": 0})
        .fillna(0)
        .astype(int)
    )


def get_baseline_table(pseudo_by_timing: pd.DataFrame) -> pd.DataFrame:
    """疑似イベントの timing_bucket × window 別平均残差を横持ちにする。"""
    pseudo = pseudo_by_timing[pseudo_by_timing["group"] == "pseudo_no_change"].copy()
    base = pseudo.pivot_table(
        index="timing_bucket",
        columns="window",
        values="mean_event_sum_points_residual",
        aggfunc="mean",
    )
    base = base.rename(columns={w: f"pseudo_baseline_sum_{w}" for w in base.columns})
    base = base.reset_index()
    return base


def make_uplift_dataset() -> pd.DataFrame:
    cond = read_csv(CONDITION_CSV)
    labels = read_csv(LABEL_CSV)
    pseudo_timing = read_csv(PSEUDO_BY_TIMING_CSV)

    # ラベル側と条件側に重複している分析列があるため、手入力ラベル列だけを優先して結合する
    keep_label_cols = [
        "event_id",
        "manual_priority",
        "new_manager_type",
        "new_manager_jleague_experience",
        "change_type",
        "style_change",
        "prior_relationship_to_club",
        "is_internal_continuity",
        "is_caretaker_or_interim",
        "was_event_true_dismissal",
        "reason_category",
        "source_url",
        "label_confidence",
        "memo",
    ]
    labels = labels[[c for c in keep_label_cols if c in labels.columns]].copy()

    df = cond.merge(labels, on="event_id", how="left", suffixes=("", "_label"))

    base = get_baseline_table(pseudo_timing)
    df = df.merge(base, on="timing_bucket", how="left")

    # 手入力ブール列の整形
    for c in ["is_internal_continuity", "is_caretaker_or_interim", "was_event_true_dismissal"]:
        if c in df.columns:
            df[c] = normalize_bool_series(df[c])

    # 疑似イベント平均との差分 = 監督交代アップリフト
    for w in WINDOWS:
        residual_col = f"points_residual_sum_{w}"
        if residual_col not in df.columns:
            # 古い列名にフォールバック
            residual_col = f"sum_points_residual_{w}"
        baseline_col = f"pseudo_baseline_sum_{w}"
        df[f"uplift_vs_pseudo_sum_{w}"] = df[residual_col] - df[baseline_col]
        df[f"uplift_flag_{w}"] = (df[f"uplift_vs_pseudo_sum_{w}"] > 0).astype(int)
        df[f"strong_uplift_flag_{w}"] = (df[f"uplift_vs_pseudo_sum_{w}"] >= 1.0).astype(int)
        df[f"strong_downlift_flag_{w}"] = (df[f"uplift_vs_pseudo_sum_{w}"] <= -1.0).astype(int)

    # 分析対象フラグ: 一時不在・予定交代・健康理由などは通常の解任ブーストと別物なので主分析から除外候補
    exclude_change = {"temporary_absence", "scheduled", "health_or_personal"}
    df["analysis_include"] = (~df["change_type"].isin(exclude_change)).astype(int)
    df.loc[df["new_manager_type"].eq("unknown"), "analysis_include"] = 0
    df.loc[df["change_type"].eq("unknown"), "analysis_include"] = 0

    # ルールベースの暫定スコア。最終モデルの前に説明しやすい判定軸として使う。
    df["uplift_rule_score"] = 0
    df.loc[df["timing_bucket"].eq("17_24"), "uplift_rule_score"] += 2
    df.loc[df["timing_bucket"].isin(["00_08", "25_plus"]), "uplift_rule_score"] -= 2
    df.loc[df["last5_points"].le(3), "uplift_rule_score"] += 1
    df.loc[df["pre_context_auto"].isin(["both_bad", "defense_collapse", "attack_defense_bad"]), "uplift_rule_score"] += 1
    df.loc[df["new_manager_type"].isin(["internal", "caretaker", "returning"]), "uplift_rule_score"] += 1
    df.loc[df["new_manager_type"].eq("external"), "uplift_rule_score"] -= 1
    df.loc[df["reason_category"].isin(["relegation_battle", "poor_form"]), "uplift_rule_score"] += 1
    df.loc[df["reason_category"].isin(["temporary_absence", "scheduled_transition", "personal_reason"]), "uplift_rule_score"] -= 2
    # hard schedule は強い相手、easy schedule は弱い相手という前提
    df.loc[df["opponent_elo_bucket_after_1_3"].astype(str).str.contains("hard", na=False), "uplift_rule_score"] -= 1
    df.loc[df["opponent_elo_bucket_after_1_3"].astype(str).str.contains("easy", na=False), "uplift_rule_score"] += 1

    def level(score: float) -> str:
        if score >= 4:
            return "high"
        if score >= 2:
            return "medium"
        if score >= 0:
            return "low"
        return "very_low"

    df["uplift_rule_level"] = df["uplift_rule_score"].apply(level)

    return df


def summary_by(df: pd.DataFrame, group_cols: Iterable[str], window: str = MAIN_WINDOW) -> pd.DataFrame:
    target = f"uplift_vs_pseudo_sum_{window}"
    flag = f"uplift_flag_{window}"
    strong = f"strong_uplift_flag_{window}"
    down = f"strong_downlift_flag_{window}"
    residual = f"points_residual_sum_{window}"
    if residual not in df.columns:
        residual = f"sum_points_residual_{window}"

    d = df[df["analysis_include"].eq(1)].copy()
    out = (
        d.groupby(list(group_cols), dropna=False)
        .agg(
            n_events=("event_id", "count"),
            mean_uplift=(target, "mean"),
            median_uplift=(target, "median"),
            mean_raw_residual=(residual, "mean"),
            uplift_rate=(flag, "mean"),
            strong_uplift_rate=(strong, "mean"),
            strong_downlift_rate=(down, "mean"),
        )
        .reset_index()
        .sort_values(["mean_uplift", "n_events"], ascending=[False, False])
    )
    return out


def make_summaries(df: pd.DataFrame) -> None:
    tables = []
    for col in [
        "new_manager_type",
        "new_manager_jleague_experience",
        "change_type",
        "style_change",
        "prior_relationship_to_club",
        "reason_category",
        "label_confidence",
        "timing_bucket",
        "pre_context_auto",
        "uplift_rule_level",
    ]:
        s = summary_by(df, [col])
        s.insert(0, "summary_axis", col)
        s = s.rename(columns={col: "category"})
        tables.append(s)
    pd.concat(tables, ignore_index=True).to_csv(OUT_LABEL_SUMMARY, index=False, encoding="utf-8-sig")

    combos = []
    for cols in [
        ["timing_bucket", "new_manager_type"],
        ["timing_bucket", "reason_category"],
        ["new_manager_type", "reason_category"],
        ["pre_context_auto", "new_manager_type"],
        ["uplift_rule_level", "new_manager_type"],
    ]:
        s = summary_by(df, cols)
        s.insert(0, "summary_axis", " × ".join(cols))
        combos.append(s)
    pd.concat(combos, ignore_index=True).to_csv(OUT_COMBO_SUMMARY, index=False, encoding="utf-8-sig")

    summary_by(df, ["uplift_rule_level"]).to_csv(OUT_RULE_SUMMARY, index=False, encoding="utf-8-sig")


def train_explainable_models(df: pd.DataFrame) -> None:
    d = df[df["analysis_include"].eq(1)].copy()

    target_col = f"uplift_flag_{MAIN_WINDOW}"
    y = d[target_col].astype(int)

    numeric_features = [
        "matches_played_at_change",
        "remaining_matches_at_change_including_first_new",
        "last5_points",
        "last5_gf",
        "last5_ga",
        "last5_gd",
        "last5_gf_per_match",
        "last5_ga_per_match",
        "team_elo_at_cutoff",
        "opponent_elo_mean_after_1_3",
        "expected_points_sum_after_1_5",
        "pre_ppg",
        "pre_gd_per_match",
        "home_rate_after_1_3_from_matches",
        "is_internal_continuity",
        "is_caretaker_or_interim",
        "was_event_true_dismissal",
        "uplift_rule_score",
    ]
    categorical_features = [
        "timing_bucket",
        "pre_context_auto",
        "new_manager_type",
        "new_manager_jleague_experience",
        "change_type",
        "style_change",
        "prior_relationship_to_club",
        "reason_category",
        "opponent_elo_bucket_after_1_3",
        "event_kind_auto",
    ]

    numeric_features = [c for c in numeric_features if c in d.columns]
    categorical_features = [c for c in categorical_features if c in d.columns]
    feature_cols = numeric_features + categorical_features
    X = d[feature_cols].copy()

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    tree = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=8,
        class_weight="balanced",
        random_state=42,
    )
    clf = Pipeline(steps=[("preprocess", preprocessor), ("model", tree)])

    # 小標本なので、精度は参考値。件数不足でAUCが計算不能な場合に備える。
    metrics = []
    if len(d) >= 30 and y.nunique() == 2:
        n_splits = min(5, y.value_counts().min())
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        pred = cross_val_predict(clf, X, y, cv=cv, method="predict")
        proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
        metrics.append(
            {
                "model": "decision_tree_classifier_cv",
                "target": target_col,
                "n_events": len(d),
                "positive_rate": y.mean(),
                "accuracy": accuracy_score(y, pred),
                "balanced_accuracy": balanced_accuracy_score(y, pred),
                "roc_auc": roc_auc_score(y, proba),
            }
        )

    # 全データで説明用モデルを学習してルールを出す
    clf.fit(X, y)
    fitted_pre = clf.named_steps["preprocess"]
    fitted_tree = clf.named_steps["model"]

    # 特徴名を取得
    feature_names = []
    feature_names.extend(numeric_features)
    try:
        cat_names = fitted_pre.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(categorical_features).tolist()
        feature_names.extend(cat_names)
    except Exception:
        feature_names.extend(categorical_features)

    rules = export_text(fitted_tree, feature_names=feature_names, decimals=3)
    with open(OUT_RULES, "w", encoding="utf-8") as f:
        f.write("Decision Tree rules for uplift_flag_after_1_5\n")
        f.write("Target: uplift_vs_pseudo_sum_after_1_5 > 0\n")
        f.write("Note: small-sample exploratory model; use as rule discovery, not definitive prediction.\n\n")
        f.write(rules)

    importances = pd.DataFrame(
        {"feature": feature_names, "importance": fitted_tree.feature_importances_}
    ).sort_values("importance", ascending=False)
    importances.to_csv(OUT_FEATURE_IMPORTANCE, index=False, encoding="utf-8-sig")

    # 参考としてRFもCVだけ出す。説明用の主役ではない。
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=4,
        min_samples_leaf=6,
        class_weight="balanced",
        random_state=42,
    )
    rf_pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", rf)])
    if len(d) >= 30 and y.nunique() == 2:
        n_splits = min(5, y.value_counts().min())
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        pred = cross_val_predict(rf_pipe, X, y, cv=cv, method="predict")
        proba = cross_val_predict(rf_pipe, X, y, cv=cv, method="predict_proba")[:, 1]
        metrics.append(
            {
                "model": "random_forest_classifier_cv_reference",
                "target": target_col,
                "n_events": len(d),
                "positive_rate": y.mean(),
                "accuracy": accuracy_score(y, pred),
                "balanced_accuracy": balanced_accuracy_score(y, pred),
                "roc_auc": roc_auc_score(y, proba),
            }
        )

    pd.DataFrame(metrics).to_csv(OUT_METRICS, index=False, encoding="utf-8-sig")


def make_prediction_template(df: pd.DataFrame) -> None:
    cols = [
        "season",
        "team",
        "effective_change_date",
        "matches_played_at_change",
        "remaining_matches_at_change_including_first_new",
        "last5_points",
        "last5_gf",
        "last5_ga",
        "last5_gd",
        "team_elo_at_cutoff",
        "opponent_elo_mean_after_1_3",
        "expected_points_sum_after_1_5",
        "pre_ppg",
        "pre_gd_per_match",
        "home_rate_after_1_3_from_matches",
        "timing_bucket",
        "pre_context_auto",
        "new_manager_type",
        "new_manager_jleague_experience",
        "change_type",
        "style_change",
        "prior_relationship_to_club",
        "reason_category",
        "opponent_elo_bucket_after_1_3",
        "memo",
    ]
    sample = pd.DataFrame([{c: "" for c in cols}])
    sample.to_csv(OUT_TEMPLATE, index=False, encoding="utf-8-sig")


def make_html_summary(df: pd.DataFrame) -> None:
    label_summary = pd.read_csv(OUT_LABEL_SUMMARY)
    combo_summary = pd.read_csv(OUT_COMBO_SUMMARY)
    rule_summary = pd.read_csv(OUT_RULE_SUMMARY)
    metrics = pd.read_csv(OUT_METRICS) if OUT_METRICS.exists() else pd.DataFrame()

    # 見やすいように主要軸だけ抽出
    manager_type = label_summary[label_summary["summary_axis"].eq("new_manager_type")]
    change_type = label_summary[label_summary["summary_axis"].eq("change_type")]
    timing_manager = combo_summary[combo_summary["summary_axis"].eq("timing_bucket × new_manager_type")]

    def fmt(df_: pd.DataFrame) -> str:
        return df_.round(3).to_html(index=False, classes="result-table", border=0)

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>監督交代アップリフト判定モデル | Football Prediction Lab</title>
  <link rel="stylesheet" href="style.css">
  <style>
    .result-table {{ width: 100%; border-collapse: collapse; font-size: 13px; background: white; }}
    .result-table th, .result-table td {{ border: 1px solid #ddd; padding: 8px 10px; text-align: center; white-space: nowrap; }}
    .result-table th {{ background: #222; color: white; }}
    .result-table tbody tr:nth-child(even) {{ background: #f5f5f5; }}
    .table-wrap {{ overflow-x: auto; }}
    .note-box {{ background: #f8fafc; border-left: 5px solid #9400d3; padding: 16px 18px; border-radius: 10px; line-height: 1.8; }}
  </style>
</head>
<body>
  <header>
    <h1>監督交代アップリフト判定モデル</h1>
    <p>平均回帰を差し引いたうえで、監督交代後1〜5試合の上振れ条件を確認</p>
  </header>
  <main>
    <section>
      <h2>概要</h2>
      <div class="note-box">
        このページでは、監督交代イベントの勝点残差から、同じ交代時期の疑似イベント平均を差し引いた
        <strong>uplift_vs_pseudo_sum_after_1_5</strong> を目的変数として扱う。
        これは「低迷後の自然回復を超えて、監督交代による追加上振れがあったか」を見るための探索指標である。
        <br><br>
        分析対象イベント数: {int(df['analysis_include'].sum())} / 全イベント数: {len(df)}
      </div>
    </section>
    <section><h2>後任監督タイプ別</h2><div class="table-wrap">{fmt(manager_type)}</div></section>
    <section><h2>交代種別別</h2><div class="table-wrap">{fmt(change_type)}</div></section>
    <section><h2>交代時期 × 後任監督タイプ</h2><div class="table-wrap">{fmt(timing_manager)}</div></section>
    <section><h2>ルールスコア別</h2><div class="table-wrap">{fmt(rule_summary)}</div></section>
    <section><h2>モデル評価（参考）</h2><div class="table-wrap">{fmt(metrics) if not metrics.empty else '<p>metricsなし</p>'}</div></section>
  </main>
</body>
</html>"""
    OUT_HTML.write_text(html, encoding="utf-8")


def main() -> None:
    df = make_uplift_dataset()
    df.to_csv(OUT_DATASET, index=False, encoding="utf-8-sig")
    make_summaries(df)
    train_explainable_models(df)
    make_prediction_template(df)
    make_html_summary(df)

    print("完了しました。主な出力:")
    for p in [
        OUT_DATASET,
        OUT_LABEL_SUMMARY,
        OUT_COMBO_SUMMARY,
        OUT_RULE_SUMMARY,
        OUT_METRICS,
        OUT_FEATURE_IMPORTANCE,
        OUT_RULES,
        OUT_TEMPLATE,
        OUT_HTML,
    ]:
        print(" -", p.name)


if __name__ == "__main__":
    main()
