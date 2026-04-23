import pandas as pd
import mlflow
import mlflow.sklearn
from pathlib import Path
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from mlflow.models import infer_signature
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=UserWarning)


def load_data(data_path: str = "data/processed/datos_integrados.csv"):
    """Función para cargar los datos procesados."""
    # ---------------------------------------------------------------------------
    # Lectura de datos
    # ---------------------------------------------------------------------------
    df = pd.read_csv(data_path)

    target = "falta_pago"
    features_X = df.drop(columns=[target])
    labels_y = df[target]

    # ---------------------------------------------------------------------------
    # División en train / validation / test
    # ---------------------------------------------------------------------------
    X_temp, X_test, y_temp, y_test = train_test_split(
        features_X, labels_y, test_size=0.10, random_state=42, stratify=labels_y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.22, random_state=42, stratify=y_temp
    )
    return X_train, y_train, X_val, y_val, X_test, y_test, features_X, labels_y


def create_preprocessor(features_X):
    """Función para crear el preprocesador."""
    # ---------------------------------------------------------------------------
    # Columnas numéricas y categóricas
    # ---------------------------------------------------------------------------
    num_cols = features_X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = features_X.select_dtypes(include=["object", "category"]).columns.tolist()

    # ---------------------------------------------------------------------------
    # Preprocesamiento
    # ---------------------------------------------------------------------------
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("scaler", StandardScaler())]), num_cols),
        ("cat", Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])

    return preprocessor


def train_model(
    data_path: str = "data/processed/datos_integrados.csv",
    model_output_path: str = "models/prod_model.pkl",
    preprocessor_output_path: str = "models/prod_preprocessor.pkl",
    metrics_output_path: str = "metrics/train_metrics.json"
):
    # ---------------------------------------------------------------------------
    # Configuración MLflow
    # ---------------------------------------------------------------------------
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Proyecto 13MBID - Model Training Pipeline")
    
    # ---------------------------------------------------------------------------
    # Carga de datos
    # ---------------------------------------------------------------------------
    X_train, y_train, X_val, y_val, X_test, y_test, features_X, labels_y = load_data(data_path)
    
    
    # ---------------------------------------------------------------------------
    # Codificación de la variable objetivo (N→0, Y→1)
    # ---------------------------------------------------------------------------
    if set(labels_y.dropna().unique()) == {"N", "Y"}:
        y_train_eval = y_train.map({"N": 0, "Y": 1})
        y_test_eval  = y_test.map({"N": 0, "Y": 1})
    else:
        y_train_eval = y_train.copy()
        y_test_eval  = y_test.copy()

    # ---------------------------------------------------------------------------
    # Pipeline con el mejor modelo: RandomForestClassifier
    # ---------------------------------------------------------------------------
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)

    preprocessor = create_preprocessor(features_X)

    pipeline = ImbPipeline([
        ("prep", preprocessor),
        ("undersample", RandomUnderSampler(random_state=42)),
        ("model", modelo),
    ])

    pipeline.fit(X_train, y_train_eval)

    # ---------------------------------------------------------------------------
    # Evaluación en test
    # ---------------------------------------------------------------------------
    y_test_pred  = pipeline.predict(X_test)
    y_test_score = (
        pipeline.predict_proba(X_test)[:, 1]
        if hasattr(pipeline, "predict_proba")
        else pipeline.decision_function(X_test)
    )

    metrics = {
        "test_accuracy":  accuracy_score(y_test_eval, y_test_pred),
        "test_precision": precision_score(y_test_eval, y_test_pred, zero_division=0),
        "test_recall":    recall_score(y_test_eval, y_test_pred, zero_division=0),
        "test_f1":        f1_score(y_test_eval, y_test_pred, zero_division=0),
        "test_roc_auc":   roc_auc_score(y_test_eval, y_test_score),
    }

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # ---------------------------------------------------------------------------
    # Matriz de confusión (artefacto)
    # ---------------------------------------------------------------------------
    cm = confusion_matrix(y_test_eval, y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Matriz de Confusión - RandomForestClassifier")
    cm_path = "docs/figures/confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()

    # ---------------------------------------------------------------------------
    # Registro en MLflow
    # ---------------------------------------------------------------------------
    signature = infer_signature(X_train, pipeline.predict(X_train))

    with mlflow.start_run(run_name="Pipeline (prod) - RandomForestClassifier"):
        mlflow.log_params(modelo.get_params())
        mlflow.log_params({
            "train_samples":    len(X_train),
            "validation_samples": len(X_val),
            "test_samples":     len(X_test),
            "balancing_method": "undersampling",
        })
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(cm_path)
        mlflow.sklearn.log_model(pipeline, artifact_path="model", signature=signature)

        run_id = mlflow.active_run().info.run_id
        print(f"Modelo registrado en MLflow. run_id: {run_id}")
    
    Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(preprocessor_output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_output_path).parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(pipeline, model_output_path)
    joblib.dump(pipeline.named_steps["prep"], preprocessor_output_path)
    
    # Guardar métricas
    with open(metrics_output_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    return pipeline, pipeline.named_steps["prep"], metrics


if __name__ == "__main__":
    train_model()

####################################################################################
# Se ha seleccionado RandomForestClassifier (n_estimators=100, random_state=42)
# como modelo definitivo del proyecto, por ser el que obtiene el mejor desempeño
# en la experimentación previa (ROC-AUC: 0.9403, Recall: 0.8751).
#####################################################################################