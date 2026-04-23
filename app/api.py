from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Literal
import pandas as pd
import joblib
 
app = FastAPI(
    title="Modelo de Predicción de Mora en Créditos",
    description="API para predecir la probabilidad de mora en créditos utilizando un modelo "
                "de machine learning entrenado con datos históricos. Proyecto 13MBID.",
    version="1.0.0"
)
 
# ---------------------------------------------------------------------------
# Esquema de entrada
# 22 variables predictoras del dataset datos_integrados.csv
# (excluye 'falta_pago', variable objetivo)
# ---------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    # --- Variables del cliente bancario ---
    edad: int = Field(..., ge=18, le=100,
                      description="Edad del cliente en años")
    antiguedad_empleado: float = Field(..., ge=0,
                                       description="Años de antigüedad laboral")
    situacion_vivienda: Literal["ALQUILER", "HIPOTECA", "OTROS", "PROPIA"] = Field(
        ..., description="Situación de vivienda del cliente")
    ingresos: int = Field(..., ge=0,
                          description="Ingresos anuales del cliente")
    objetivo_credito: Literal[
        "EDUCACIÓN", "INVERSIONES", "MEJORAS_HOGAR",
        "PAGO_DEUDAS", "PERSONAL", "SALUD"
    ] = Field(..., description="Propósito del crédito solicitado")
    pct_ingreso: float = Field(..., ge=0, le=1,
                               description="Porcentaje del ingreso destinado al crédito")
    tasa_interes: float = Field(..., ge=0,
                                description="Tasa de interés del crédito (%)")
    estado_credito: int = Field(..., ge=0, le=1,
                                description="Historial crediticio: 1=bueno, 0=malo")
 
    # --- Variables de productos del cliente (tarjeta de crédito) ---
    antiguedad_cliente: float = Field(..., ge=0,
                                      description="Meses como cliente de la entidad")
    estado_civil: Literal["CASADO", "DESCONOCIDO", "DIVORCIADO", "SOLTERO"] = Field(
        ..., description="Estado civil del cliente")
    estado_cliente: Literal["ACTIVO", "PASIVO"] = Field(
        ..., description="Estado actual del cliente")
    gastos_ult_12m: float = Field(..., ge=0,
                                  description="Gastos totales con tarjeta en los últimos 12 meses")
    genero: Literal["M", "F"] = Field(..., description="Género del cliente")
    limite_credito_tc: float = Field(..., ge=0,
                                     description="Límite de crédito de la tarjeta")
    nivel_educativo: Literal[
        "DESCONOCIDO", "POSGRADO_COMPLETO", "POSGRADO_INCOMPLETO",
        "SECUNDARIO_COMPLETO", "UNIVERSITARIO_COMPLETO", "UNIVERSITARIO_INCOMPLETO"
    ] = Field(..., description="Nivel educativo del cliente")
    personas_a_cargo: float = Field(..., ge=0,
                                    description="Número de personas a cargo del cliente")
 
    # --- Variables derivadas (fase de preparación de datos) ---
    capacidad_pago: float = Field(..., ge=0,
                                  description="Ratio importe_solicitado / ingresos")
    operaciones_mensuales: float = Field(..., ge=0,
                                         description="Frecuencia mensual de operaciones del cliente")
    presion_financiera: float = Field(..., ge=0,
                                      description="Ratio gastos mensuales totales / ingresos mensuales")
    gasto_prom_por_operacion: float = Field(..., ge=0,
                                             description="Gasto promedio por operación con tarjeta")
    operaciones_mensuales_tarjeta: float = Field(..., ge=0,
                                                  description="Frecuencia mensual de uso de la tarjeta")
    estabilidad_laboral: float = Field(..., ge=0, le=1,
                                       description="Ratio antiguedad_empleado / edad")
 
    class Config:
        json_schema_extra = {
            "example": {
                "edad": 21,
                "antiguedad_empleado": 5.0,
                "situacion_vivienda": "PROPIA",
                "ingresos": 9600,
                "objetivo_credito": "EDUCACIÓN",
                "pct_ingreso": 0.1,
                "tasa_interes": 11.14,
                "estado_credito": 0,
                "antiguedad_cliente": 39.0,
                "estado_civil": "CASADO",
                "estado_cliente": "ACTIVO",
                "gastos_ult_12m": 1144.0,
                "genero": "M",
                "limite_credito_tc": 12691.0,
                "nivel_educativo": "SECUNDARIO_COMPLETO",
                "personas_a_cargo": 3.0,
                "capacidad_pago": 0.104167,
                "operaciones_mensuales": 3.5,
                "presion_financiera": 0.17125,
                "gasto_prom_por_operacion": 27.238095,
                "operaciones_mensuales_tarjeta": 3.5,
                "estabilidad_laboral": 0.238095,
            }
        }
 
 
# ---------------------------------------------------------------------------
# Esquema de respuesta
# ---------------------------------------------------------------------------
class PredictionResponse(BaseModel):
    prediction: str
    probability: Dict[str, float]
    class_labels: Dict[str, str]
    model_info: Dict[str, str]
 
 
# ---------------------------------------------------------------------------
# Carga del modelo (no bloqueante: la API arranca aunque el modelo falle)
# ---------------------------------------------------------------------------
MODEL_PATH = "models/prod_model.pkl"
 
try:
    model = joblib.load(MODEL_PATH)
    print("Modelo cargado exitosamente.")
except FileNotFoundError:
    print(f"Error: No se encontró el modelo en '{MODEL_PATH}'. "
          "Ejecutar primero: python src/train_model.py")
    model = None
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    model = None
 
 
# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", summary="Bienvenida y descripción de endpoints")
def read_root():
    return {
        "message": "Bienvenido a la API de Predicción de Mora en Créditos",
        "endpoints": {
            "/predict": "POST - Realiza una predicción de mora para un crédito",
            "/docs":    "GET  - Documentación interactiva de la API (Swagger UI)",
            "/health":  "GET  - Verifica el estado de la API y del modelo",
        }
    }
 
 
@app.get("/health", summary="Health check")
def health_check():
    """Verifica que la API está activa y el modelo está cargado."""
    if model is not None:
        return {"status": "ok", "message": "La API está funcionando correctamente."}
    return {"status": "error", "message": "El modelo no está cargado. Verifica el estado del modelo."}
 
 
@app.post("/predict", response_model=PredictionResponse, summary="Predecir mora de un crédito")
def predict(request: PredictionRequest):
    """
    Recibe las características de un crédito y devuelve:
    - **prediction**: clase predicha (0 = no mora, 1 = mora)
    - **probability**: probabilidad para cada clase
    - **class_labels**: descripción legible de cada clase
    - **model_info**: metadatos del modelo utilizado
    """
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="El modelo no está disponible. Ejecutar primero: python src/train_model.py"
        )
 
    try:
        # Convertir el input a DataFrame (orden de columnas respetado por Pydantic)
        input_data = pd.DataFrame([request.dict()])
 
        # Predicción y probabilidades
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
 
        # Las clases del modelo (0 y 1 tras la codificación N->0, Y->1)
        class_labels = model.named_steps["model"].classes_
 
        # Conversión a str para garantizar serialización JSON correcta
        probability_dict = {
            str(class_labels[i]): float(probability[i])
            for i in range(len(class_labels))
        }
 
        model_info = {
            "model_version": "1.0.0",
            # Se obtiene dinámicamente para reflejar el modelo real cargado
            "model_type": type(model.named_steps["model"]).__name__,
        }
 
        return PredictionResponse(
            prediction=str(prediction),
            probability=probability_dict,
            class_labels={
                "0": "No entra en mora (N)",
                "1": "Entra en mora (Y)",
            },
            model_info=model_info,
        )
 
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error al realizar la predicción: {e}")
 
 
# ---------------------------------------------------------------------------
# Arranque local: uvicorn src.api:app --reload
# ---------------------------------------------------------------------------