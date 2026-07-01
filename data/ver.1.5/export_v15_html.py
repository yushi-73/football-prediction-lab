import pandas as pd
from pathlib import Path
from html import escape

# ============================================================
# ver1.5 Web掲載用 HTML出力
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

SUMMARY_CSV = BASE_DIR / "v15_base_multiyear_summary.csv"
DETAIL_CSV = BASE_DIR / "v15_base_multiyear_detail.csv"
PREDICTIONS_CSV = BASE_DIR / "v15_base_multiyear_predictions.csv"
MATCH_LOG_CSV = BASE_DIR / "v15_base_multiyear_match_log.csv"

OUTPUT_HTML = BASE_DIR / "ver15_predictions.html"

TARGET_YEARS = [2023, 2024, 2025]


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"{path.name} が見つかりません。先に v15_base_model_multiyear.py を実行してください。")
    return pd.read_csv(path, encoding="utf-8-sig")


def fmt_num(value, digits=3):
    if pd.isna(value):
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def fmt_pct(value, digits=1):
    if pd.isna(value):
        return "-"
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except Exception:
        return str(value)


def fmt_int(value):
    if pd.isna(value):
        return "-"
    try:
        return str(int(round(float(value))))
    except Exception:
        return str(value)


def make_html_table(df, table_class="result-table"):
    return df.to_html(index=False, escape=False, border=0, classes=table_class)


summary_df = read_csv(SUMMARY_CSV)
detail_df = read_csv(DETAIL_CSV)
pred_df = read_csv(PREDICTIONS_CSV)

summary = summary_df.iloc[0]

# =========================
# 1. サマリーカード
# =========================

summary_cards = [
    ("3年平均MAE", fmt_num(summary.get("mean_mae"), 4), "小さいほど実順位に近い"),
    ("MAE標準偏差", fmt_num(summary.get("std_mae"), 4), "年度ごとのブレ"),
    ("実順位的中確率平均", fmt_pct(summary.get("mean_prob_actual_position"), 2), "実順位に入った割合の平均"),
    ("シミュレーション引分率", fmt_pct(summary.get("mean_sim_draw_rate"), 2), "モデル上の引分率"),
]

card_html = ""
for title, value, note in summary_cards:
    card_html += f"""
    <div class="metric-card">
      <div class="metric-title">{escape(title)}</div>
      <div class="metric-value">{escape(value)}</div>
      <div class="metric-note">{escape(note)}</div>
    </div>
    """


# =========================
# 2. 年度別サマリー
# =========================

year_summary_rows = []

for year in TARGET_YEARS:
    row = detail_df[detail_df["target_year"] == year]
    if row.empty:
        continue

    r = row.iloc[0]
    year_summary_rows.append({
        "年度": year,
        "前年": int(r.get("previous_year")),
        "MAE": fmt_num(r.get("mae"), 4),
        "実順位的中確率平均": fmt_pct(r.get("mean_prob_actual_position"), 2),
        "シミュレーション引分率": fmt_pct(r.get("sim_draw_rate"), 2),
        "対象試合数": fmt_int(r.get("n_target_matches")),
        "学習試合数": fmt_int(r.get("n_train_matches")),
        "予測試合数": fmt_int(r.get("n_test_matches")),
    })

year_summary_df = pd.DataFrame(year_summary_rows)


# =========================
# 3. 採用設定
# =========================

settings = [
    ("対象年度", "2023, 2024, 2025"),
    ("モデル", "ver1.5 base"),
    ("PREV_WEIGHT", summary.get("prev_weight")),
    ("PREV_DECAY", summary.get("prev_decay")),
    ("ELO_LAMBDA_WEIGHT", summary.get("elo_lambda_weight")),
    ("COMPAT_WEIGHT", summary.get("compat_weight")),
    ("CURRENT_DECAY", summary.get("current_decay")),
    ("SHRINKAGE", summary.get("shrinkage")),
    ("DRAW_FACTOR", summary.get("draw_factor")),
    ("MAX_MATCH_DRAW_PROB", summary.get("max_match_draw_prob")),
    ("LAMBDA_CAP", summary.get("lambda_cap")),
    ("GOAL_ADJUST", summary.get("goal_adjust_name")),
    ("GOAL_CAP_FOR_STRENGTH", summary.get("goal_cap_for_strength")),
    ("MATCHUP_PRIOR_N", summary.get("matchup_prior_n")),
    ("MATCHUP_TIME_DECAY", summary.get("matchup_time_decay")),
]

settings_df = pd.DataFrame(settings, columns=["項目", "設定値"])
settings_df["設定値"] = settings_df["設定値"].apply(lambda x: "-" if pd.isna(x) else str(x))


# =========================
# 4. 年度別順位表
# =========================

def make_year_table(year):
    df = pred_df[pred_df["target_year"] == year].copy()

    if df.empty:
        return pd.DataFrame()

    if "pred_rank" in df.columns:
        df = df.sort_values("pred_rank")
    else:
        df = df.sort_values("avg_pred_position")
        df.insert(0, "pred_rank", range(1, len(df) + 1))

    cols = [
        "pred_rank",
        "team",
        "actual_position",
        "avg_pred_position",
        "position_error",
        "prob_actual_position",
        "most_likely_position",
        "champion_prob",
        "top3_prob",
        "bottom3_prob",
        "avg_points",
        "avg_gf",
        "avg_ga",
        "avg_gd",
    ]

    cols = [c for c in cols if c in df.columns]
    df = df[cols].copy()

    rename = {
        "pred_rank": "予測順位",
        "team": "チーム",
        "actual_position": "実順位",
        "avg_pred_position": "平均予測順位",
        "position_error": "順位誤差",
        "prob_actual_position": "実順位確率",
        "most_likely_position": "最頻順位",
        "champion_prob": "優勝確率",
        "top3_prob": "TOP3確率",
        "bottom3_prob": "下位3確率",
        "avg_points": "平均勝点",
        "avg_gf": "平均得点",
        "avg_ga": "平均失点",
        "avg_gd": "平均得失点差",
    }

    df = df.rename(columns=rename)

    for c in ["予測順位", "実順位", "最頻順位"]:
        if c in df.columns:
            df[c] = df[c].apply(fmt_int)

    for c in ["平均予測順位", "順位誤差", "平均勝点", "平均得点", "平均失点", "平均得失点差"]:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: fmt_num(x, 2))

    for c in ["実順位確率", "優勝確率", "TOP3確率", "下位3確率"]:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: fmt_pct(x, 1))

    return df


year_sections = ""

for year in TARGET_YEARS:
    table_df = make_year_table(year)

    if table_df.empty:
        table_html = "<p>この年度の予測結果が見つかりませんでした。</p>"
    else:
        table_html = make_html_table(table_df)

    year_detail = detail_df[detail_df["target_year"] == year]
    if not year_detail.empty:
        d = year_detail.iloc[0]
        note = (
            f"MAE={fmt_num(d.get('mae'), 4)} / "
            f"実順位的中確率平均={fmt_pct(d.get('mean_prob_actual_position'), 2)} / "
            f"引分率={fmt_pct(d.get('sim_draw_rate'), 2)}"
        )
    else:
        note = ""

    year_sections += f"""
    <section>
      <h2>{year}年シーズン予測順位表</h2>
      <p class="section-note">{escape(note)}</p>
      <div class="table-wrap">
        {table_html}
      </div>
    </section>
    """


# =========================
# 5. HTML生成
# =========================

html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>ver1.5 3年検証結果 | Football Prediction Lab</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">

  <link rel="stylesheet" href="style.css">

  <style>
    .page-nav {{
      display: flex;
      justify-content: center;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 24px;
    }}

    .page-nav a {{
      color: white;
      text-decoration: none;
      border: 1px solid rgba(255,255,255,0.7);
      padding: 8px 14px;
      border-radius: 999px;
      font-weight: 700;
    }}

    .version-badge {{
      display: inline-block;
      background: #e9d5ff;
      color: #6b21a8;
      padding: 6px 12px;
      border-radius: 999px;
      font-weight: 800;
      margin-bottom: 12px;
    }}

    .lead {{
      font-size: 16px;
      line-height: 1.9;
      color: #374151;
    }}

    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 16px;
      margin-top: 20px;
    }}

    .metric-card {{
      background: #f8fafc;
      border: 1px solid #e5e7eb;
      border-radius: 14px;
      padding: 18px;
      text-align: center;
    }}

    .metric-title {{
      color: #64748b;
      font-size: 13px;
      font-weight: 700;
      margin-bottom: 8px;
    }}

    .metric-value {{
      font-size: 26px;
      font-weight: 800;
      color: #9400d3;
    }}

    .metric-note {{
      color: #64748b;
      font-size: 12px;
      margin-top: 6px;
    }}

    .table-wrap {{
      overflow-x: auto;
      margin-top: 12px;
    }}

    .result-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      background: white;
    }}

    .result-table th,
    .result-table td {{
      border: 1px solid #ddd;
      padding: 8px 10px;
      text-align: center;
      white-space: nowrap;
    }}

    .result-table th {{
      background: #222;
      color: #fff;
    }}

    .result-table tbody tr:nth-child(even) {{
      background: #f5f5f5;
    }}

    .section-note {{
      color: #64748b;
      line-height: 1.7;
    }}

    .note-box {{
      background: #f8fafc;
      border-left: 5px solid #9400d3;
      padding: 16px 18px;
      border-radius: 10px;
      line-height: 1.8;
      color: #374151;
    }}
  </style>
</head>

<body>
  <header>
    <h1>ver1.5 3年検証結果</h1>
    <p>2023〜2025年の複数年度で再検証した汎用基準モデル</p>

    <nav class="page-nav">
      <a href="index.html">トップ</a>
      <a href="model.html">モデル説明</a>
      <a href="devlog.html">開発ログ</a>
      <a href="glossary.html">用語集</a>
    </nav>
  </header>

  <main>
    <section>
      <span class="version-badge">ver1.5</span>
      <h2>概要</h2>
      <p class="lead">
        ver1.5では、ver1.4までのように2025年単年だけを対象にするのではなく、
        2023年・2024年・2025年の3年分を対象に係数を再検証した。
        単年度の結果に過度に最適化するのではなく、複数年度で安定して機能する汎用的な順位予測モデルを目指している。
      </p>

      <div class="metric-grid">
        {card_html}
      </div>
    </section>

    <section>
      <h2>年度別検証サマリー</h2>
      <p class="section-note">
        各年度の前半戦を学習データ、後半戦を予測対象として、最終順位に対する平均順位誤差を計算した。
      </p>
      <div class="table-wrap">
        {make_html_table(year_summary_df)}
      </div>
    </section>

    <section>
      <h2>採用設定</h2>
      <p class="section-note">
        以下の設定をver1.5の基準モデルとして固定し、今後の監督解任ブーストや終盤補正の比較対象とする。
      </p>
      <div class="table-wrap">
        {make_html_table(settings_df)}
      </div>
    </section>

    {year_sections}

    <section>
      <h2>今後の検証への接続</h2>
      <div class="note-box">
        ver1.5では試合単位の予測ログも出力している。
        このログには、各試合の期待得点、勝敗確率、期待勝点、実勝点との差分が含まれる。
        そのため、監督解任後の試合だけを抽出し、ver1.5の期待値に対して実際の勝点や得失点が上振れしているかを検証できる。
      </div>
    </section>
  </main>
</body>
</html>
"""

OUTPUT_HTML.write_text(html, encoding="utf-8")
print(f"HTMLを出力しました: {OUTPUT_HTML}")