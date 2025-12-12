import os
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from flask import Flask, jsonify, render_template_string

from Taller.taller_world_cup_ml import (
    PREDICTION_YEAR,
    build_pipeline,
    generate_year_ranking,
    load_dataset,
    prepare_team_seasons,
    train_full_model,
)

app = Flask(__name__)

# Precarga del modelo y del ranking para servir rapido.
_raw_df = load_dataset()
_df = prepare_team_seasons(_raw_df)
_pipeline, _features, _target = build_pipeline(_df)
_final_model = train_full_model(_pipeline, _features, _target)
_ranking_2026 = generate_year_ranking(_final_model, _df, target_year=PREDICTION_YEAR)


def _format_record(row: pd.Series) -> Dict[str, Any]:
    return {
        "team": row["Team"],
        "year": int(row["Year"]),
        "probability": round(float(row["Champion_Prob"]), 4),
    }


@app.route("/api/predictions")
def predictions_api():
    data = [_format_record(row) for _, row in _ranking_2026.iterrows()]
    return jsonify({"prediction_year": PREDICTION_YEAR, "predictions": data})


INDEX_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>World Cup 2026 Prediction</title>
  <style>
    body { font-family: Arial, sans-serif; background: #0d1117; color: #e6edf3; margin: 0; padding: 1.5rem; }
    header { margin-bottom: 1rem; }
    h1 { margin: 0; font-size: 1.8rem; }
    p { margin: 0.2rem 0; }
    table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
    th, td { padding: 0.65rem; text-align: left; }
    th { background: #161b22; }
    tr:nth-child(even) { background: #0f141a; }
    tr:nth-child(odd) { background: #0b0f14; }
    .badge { display: inline-block; padding: 0.2rem 0.5rem; border-radius: 0.5rem; background: #238636; color: #fff; font-weight: bold; }
  </style>
</head>
<body>
  <header>
    <h1>Predicci&oacute;n Campe&oacute;n {year}</h1>
    <p>Modelo: Regresi&oacute;n Log&iacute;stica con balanceo de clases</p>
    <p>Fuente: dataset/world_cup_prediction_dataset.xlsx</p>
  </header>
  <section>
    <table>
      <thead>
        <tr><th>#</th><th>Equipo</th><th>A&ntilde;o</th><th>Prob. Campe&oacute;n</th></tr>
      </thead>
      <tbody>
      {% for i, item in enumerate(predictions, start=1) %}
        <tr>
          <td>{{ i }}</td>
          <td>
            {% if i == 1 %}<span class="badge">Favorito</span> {% endif %}
            {{ item.team }}
          </td>
          <td>{{ item.year }}</td>
          <td>{{ '%.3f'|format(item.probability) }}</td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
  </section>
</body>
</html>
"""


@app.route("/")
def index():
    data = [_format_record(row) for _, row in _ranking_2026.iterrows()]
    return render_template_string(INDEX_TEMPLATE, predictions=data, year=PREDICTION_YEAR)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
