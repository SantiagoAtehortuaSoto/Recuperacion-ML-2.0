"""
Solucion del taller de refuerzo de Machine Learning.

Flujo:
- Carga el dataset de la Copa Mundial desde ../dataset/world_cup_prediction_dataset.xlsx.
- Agrega una unica fila por equipo/ano promediando las variables numericas.
- Prepara los datos (OneHotEncoder para Team y escalado para numericas).
- Entrena y evalua un modelo de clasificacion (Regresion Logistica con class_weight balanced).
- Reentrena el modelo con todos los datos y genera el ranking de probabilidades
  proyectado para 2026.
"""

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path(__file__).resolve().parent.parent / "dataset" / "world_cup_prediction_dataset.xlsx"
RANDOM_STATE = 42
PREDICTION_YEAR = 2026


def load_dataset() -> pd.DataFrame:
    """Carga el archivo Excel en un DataFrame de pandas."""
    df = pd.read_excel(DATA_PATH)
    return df


def prepare_team_seasons(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega una unica fila por equipo y ano promediando las variables numericas.
    Esto estabiliza los datos y evita duplicados por equipo/ano.
    """
    grouping_keys = ["Year", "Team"]
    numeric_columns = [col for col in df.columns if col not in grouping_keys + ["Champion"]]

    aggregated = (
        df.groupby(grouping_keys, as_index=False)
        .agg({**{col: "mean" for col in numeric_columns}, "Champion": "max"})
        .sort_values(grouping_keys)
        .reset_index(drop=True)
    )

    ordered_columns = grouping_keys + numeric_columns + ["Champion"]
    return aggregated[ordered_columns]


def build_pipeline(df: pd.DataFrame) -> Tuple[Pipeline, pd.DataFrame, pd.Series]:
    """
    Separa variables, define transformaciones y construye el pipeline.
    Devuelve el pipeline, las caracteristicas y el target.
    """
    target = df["Champion"]
    features = df.drop(columns=["Champion"])

    categorical_features = ["Team"]
    numerical_features = [col for col in features.columns if col not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numeric", StandardScaler(), numerical_features),
        ],
        remainder="drop",
    )

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    return pipeline, features, target


def evaluate_model(
    pipeline: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> Dict[str, float]:
    """Entrena el pipeline, evalua en test y devuelve metricas principales."""
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    print("\n=== Metricas en el conjunto de prueba ===")
    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")

    print("\n=== Matriz de confusion ===")
    print(confusion_matrix(y_test, y_pred))

    print("\n=== Reporte de clasificacion ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    baseline = y_test.value_counts(normalize=True).max()
    print(f"Baseline (mayoria de clase en test): {baseline:.3f}")

    return metrics


def train_full_model(pipeline: Pipeline, features: pd.DataFrame, target: pd.Series) -> Pipeline:
    """
    Crea una copia del pipeline y lo entrena con todo el dataset
    para utilizarlo en la generacion de probabilidades finales.
    """
    final_model = clone(pipeline)
    final_model.fit(features, target)
    return final_model


def build_year_records(df: pd.DataFrame, target_year: int) -> pd.DataFrame:
    """
    Devuelve las filas del ano solicitado. Si no existen en el dataset,
    reutiliza la ultima temporada disponible de cada equipo y actualiza el ano.
    """
    year_records = df[df["Year"] == target_year]
    if not year_records.empty:
        return year_records.reset_index(drop=True)

    latest_per_team = (
        df.sort_values("Year")
        .groupby("Team", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    return latest_per_team.assign(Year=target_year)


def generate_year_ranking(model: Pipeline, df: pd.DataFrame, target_year: int) -> pd.DataFrame:
    """
    Genera un ranking de probabilidades de campeon para el ano indicado
    y lo guarda como CSV.
    """
    records = build_year_records(df, target_year).copy()
    feature_frame = records.drop(columns=["Champion"])
    records["Champion_Prob"] = model.predict_proba(feature_frame)[:, 1]

    ranking = (
        records[["Team", "Year", "Champion_Prob"]]
        .sort_values(by="Champion_Prob", ascending=False)
        .reset_index(drop=True)
    )

    output_path = Path(__file__).resolve().parent / f"predicciones_campeon_{target_year}.csv"
    ranking.to_csv(output_path, index=False)

    top_team = ranking.iloc[0]
    print(f"\nRanking de probabilidades para {target_year} guardado en: {output_path}")
    print(f"Proyeccion {target_year}: {top_team['Team']} ({top_team['Champion_Prob']:.3f}) como favorito.")
    print(ranking.head(10).to_string(index=False))

    return ranking


def main() -> None:
    raw_df = load_dataset()
    df = prepare_team_seasons(raw_df)

    print(f"Filas y columnas del dataset (agregado por equipo/ano): {df.shape}")
    print("Valores faltantes por columna:")
    print(df.isna().sum())

    pipeline, features, target = build_pipeline(df)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        stratify=target,
        random_state=RANDOM_STATE,
    )

    metrics = evaluate_model(pipeline, X_train, X_test, y_train, y_test)

    final_model = train_full_model(pipeline, features, target)
    generate_year_ranking(final_model, df, target_year=PREDICTION_YEAR)

    metrics_path = Path(__file__).resolve().parent / "metricas_modelo.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print(f"\nMetricas guardadas en: {metrics_path}")


if __name__ == "__main__":
    main()
