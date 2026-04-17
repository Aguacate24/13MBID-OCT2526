import pandas as pd
from pathlib import Path
import pytest
import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*`Number` field should not be instantiated.*",
)

import great_expectations as ge


pytestmark = [
    pytest.mark.filterwarnings("ignore:.*Number.*should not be instantiated.*"),
    pytest.mark.filterwarnings("ignore:.*result_format.*Validator-level.*"),
    pytest.mark.filterwarnings("ignore:.*result_format.*Expectation-level.*"),
]


# Paths
PROJECT_DIR = Path(".").resolve()
DATA_DIR = PROJECT_DIR / "data"


def test_great_expectations():
    """ Prueba para validar la calidad de los datos utilizando Great Expectations.
    """
    # Cargar los datos de créditos y tarjetas
    df_creditos = pd.read_csv(DATA_DIR / "raw/datos_creditos.csv", sep=";")
    df_tarjetas = pd.read_csv(DATA_DIR / "raw/datos_tarjetas.csv", sep=";")

    results = {
        "success": True,
        "expectations": [],
        "statistics": {"success_count": 0, "total_count": 0}
    }

    def add_expectation(expectation_name, condition, message=""):
        results["statistics"]["total_count"] += 1
        if condition:
            results["statistics"]["success_count"] += 1
            results["expectations"].append({
                "expectation": expectation_name,
                "success": True
            })
        else:
            results["success"] = False
            results["expectations"].append({
                "expectation": expectation_name,
                "success": False,
                "message": message
            })

    
    # Atributo a analizar: Exactitud (rangos de valores en datos)
    # Validación 1 (créditos): Rango de edad (18-100 años)
    edad_valida = df_creditos["edad"].between(18, 100).all()
    mensaje_edad = ""
    if not edad_valida:
        edades_fuera = df_creditos[(df_creditos["edad"] < 18) | (df_creditos["edad"] > 100)]["edad"].unique()
        mensaje_edad = f"Edades fuera de rango encontradas: {sorted(edades_fuera)}"
    add_expectation(
        "rango_edad",
        edad_valida,
        f"La edad debe estar entre 18 y 100 años. {mensaje_edad}"
    )
   
    # Validación 2 (créditos): Rango de valores para situación de vivienda (ALQUILER, PROPIA, OTROS, HIPOTECA)
    vivienda_valida = df_creditos["situacion_vivienda"].isin(["ALQUILER", "PROPIA", "OTROS", "HIPOTECA"]).all()
    mensaje_vivienda = ""
    if not vivienda_valida:
        viviendas_fuera = df_creditos[~df_creditos["situacion_vivienda"].isin(["ALQUILER", "PROPIA", "OTROS", "HIPOTECA"])]["situacion_vivienda"].unique()
        mensaje_vivienda = f"Situaciones de vivienda no válidas encontradas: {sorted(viviendas_fuera)}" 
    add_expectation(
        "situacion_vivienda",
        vivienda_valida,
        f"La situación de vivienda no se encuentra en el rango válido. {mensaje_vivienda}"
    )

    # Validación 3 (créditos): Rango de valores para objetivo_credito
    objetivos_validos = {"PERSONAL", "EDUCACIÓN", "SALUD", "INVERSIONES", "MEJORAS_HOGAR", "PAGO_DEUDAS"}
    objetivo_valido = df_creditos["objetivo_credito"].isin(objetivos_validos).all()
    mensaje_objetivo = ""
    if not objetivo_valido:
        objetivos_fuera = df_creditos[~df_creditos["objetivo_credito"].isin(objetivos_validos)]["objetivo_credito"].unique()
        mensaje_objetivo = f"Objetivos de crédito no válidos encontrados: {sorted(objetivos_fuera)}"
    add_expectation(
        "objetivo_credito",
        objetivo_valido,
        f"El objetivo del crédito no se encuentra en el rango válido. {mensaje_objetivo}"
    )

    # Validación 4 (tarjetas): Rango de valores para estado_civil
    estados_civiles_validos = {"CASADO", "SOLTERO", "DIVORCIADO", "DESCONOCIDO"}
    estado_civil_valido = df_tarjetas["estado_civil"].isin(estados_civiles_validos).all()
    mensaje_estado_civil = ""
    if not estado_civil_valido:
        estados_fuera = df_tarjetas[~df_tarjetas["estado_civil"].isin(estados_civiles_validos)]["estado_civil"].unique()
        mensaje_estado_civil = f"Estados civiles no válidos encontrados: {sorted(estados_fuera)}"
    add_expectation(
        "estado_civil",
        estado_civil_valido,
        f"El estado civil no se encuentra en el rango válido. {mensaje_estado_civil}"
    )

    # Validación 5 (tarjetas): Rango de valores para nivel_educativo
    niveles_educativos_validos = {
        "UNIVERSITARIO_COMPLETO", "UNIVERSITARIO_INCOMPLETO",
        "SECUNDARIO_COMPLETO", "POSGRADO_COMPLETO",
        "POSGRADO_INCOMPLETO", "DESCONOCIDO"
    }
    nivel_educativo_valido = df_tarjetas["nivel_educativo"].isin(niveles_educativos_validos).all()
    mensaje_nivel_educativo = ""
    if not nivel_educativo_valido:
        niveles_fuera = df_tarjetas[~df_tarjetas["nivel_educativo"].isin(niveles_educativos_validos)]["nivel_educativo"].unique()
        mensaje_nivel_educativo = f"Niveles educativos no válidos encontrados: {sorted(niveles_fuera)}"
    add_expectation(
        "nivel_educativo",
        nivel_educativo_valido,
        f"El nivel educativo no se encuentra en el rango válido. {mensaje_nivel_educativo}"
    )

    # Validación 6 (tarjetas): Rango de valores para nivel_tarjeta
    niveles_tarjeta_validos = {"Blue", "Gold", "Silver", "Platinum"}
    nivel_tarjeta_valido = df_tarjetas["nivel_tarjeta"].isin(niveles_tarjeta_validos).all()
    mensaje_nivel_tarjeta = ""
    if not nivel_tarjeta_valido:
        niveles_fuera = df_tarjetas[~df_tarjetas["nivel_tarjeta"].isin(niveles_tarjeta_validos)]["nivel_tarjeta"].unique()
        mensaje_nivel_tarjeta = f"Niveles de tarjeta no válidos encontrados: {sorted(niveles_fuera)}"
    add_expectation(
        "nivel_tarjeta",
        nivel_tarjeta_valido,
        f"El nivel de tarjeta no se encuentra en el rango válido. {mensaje_nivel_tarjeta}"
    )

    # Resumen y validación final
    print("\n" + "="*70)
    print("RESUMEN DE VALIDACIONES")
    print("="*70)
    for exp in results["expectations"]:
        status = "✓ PASS" if exp["success"] else "✗ FAIL"
        print(f"{status}: {exp['expectation']}")
        if not exp["success"] and "message" in exp:
            print(f"       Detalle: {exp['message']}")
    print(f"\nTotal: {results['statistics']['success_count']}/{results['statistics']['total_count']} validaciones pasaron")
    print("="*70 + "\n")

    # El test falla si alguna validación no pasó
    assert results["success"], f"Se encontraron {results['statistics']['total_count'] - results['statistics']['success_count']} validaciones fallidas"
