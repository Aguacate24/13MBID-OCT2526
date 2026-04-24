import streamlit as st
import requests
 
st.set_page_config(
    page_title="Predicción de Mora en Créditos",
    page_icon=":credit_card:",
    layout="wide"
)
 
with st.sidebar:
    st.header("Instrucciones")
    st.write("""
    1. Ingrese los datos del cliente en el formulario.
    2. Haga clic en "Predecir" para obtener la probabilidad de mora.
    3. Las variables derivadas se calculan automáticamente.
    4. Revise los resultados y la información del modelo.
    """)
    st.divider()
    st.header("Configuración de la API")
    api_url = st.text_input(
        "URL de la API",
        value="http://localhost:8000",
        help="Ingrese la URL donde está alojada la API de predicción."
    )
    st.divider()
    if st.button("Probar Conexión a la API"):
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            if response.status_code == 200:
                st.success("Conexión exitosa a la API.")
            else:
                st.error(f"Error al conectar con la API: {response.status_code}")
        except Exception as e:
            st.error(f"Error al conectar con la API: {e}")
 
st.title("Predicción de Mora en Créditos")
st.write("Ingrese los datos del cliente. Las variables derivadas se calcularán automáticamente antes de enviar la predicción.")
 
with st.form("prediction_form"):
 
    # ------------------------------------------------------------------
    # Sección 1: Datos personales
    # ------------------------------------------------------------------
    st.subheader("Datos del Cliente")
    col1, col2, col3 = st.columns(3)
 
    with col1:
        edad = st.number_input("Edad", min_value=18, max_value=100, value=30)
        genero = st.selectbox("Género", options=["M", "F"])
        estado_civil = st.selectbox(
            "Estado Civil",
            options=["CASADO", "SOLTERO", "DIVORCIADO", "DESCONOCIDO"]
        )
 
    with col2:
        nivel_educativo = st.selectbox(
            "Nivel Educativo",
            options=[
                "SECUNDARIO_COMPLETO",
                "UNIVERSITARIO_INCOMPLETO",
                "UNIVERSITARIO_COMPLETO",
                "POSGRADO_INCOMPLETO",
                "POSGRADO_COMPLETO",
                "DESCONOCIDO",
            ]
        )
        situacion_vivienda = st.selectbox(
            "Situación de Vivienda",
            options=["ALQUILER", "HIPOTECA", "PROPIA", "OTROS"]
        )
        personas_a_cargo = st.number_input(
            "Personas a Cargo", min_value=0.0, max_value=20.0, value=0.0
        )
 
    with col3:
        estado_cliente = st.selectbox("Estado del Cliente", options=["ACTIVO", "PASIVO"])
        estado_credito = st.number_input(
            "Estado del Crédito (1=bueno, 0=malo)",
            min_value=0, max_value=1, value=1, step=1
        )
 
    st.divider()
 
    # ------------------------------------------------------------------
    # Sección 2: Información financiera y laboral
    # ------------------------------------------------------------------
    st.subheader("Información Financiera y Laboral")
    col4, col5, col6 = st.columns(3)
 
    with col4:
        ingresos = st.number_input("Ingresos anuales", min_value=1, value=50000)
        antiguedad_empleado = st.number_input(
            "Antigüedad del Empleado (años)", min_value=0.0, max_value=50.0, value=1.0
        )
 
    with col5:
        objetivo_credito = st.selectbox(
            "Objetivo del Crédito",
            options=["PERSONAL", "EDUCACIÓN", "SALUD", "INVERSIONES", "MEJORAS_HOGAR", "PAGO_DEUDAS"]
        )
        tasa_interes = st.number_input(
            "Tasa de Interés (%)", min_value=0.0, value=15.0, step=0.01, format="%.2f"
        )
        pct_ingreso = st.number_input(
            "Porcentaje del Ingreso destinado al crédito",
            min_value=0.0, max_value=1.0, value=0.1, step=0.01, format="%.2f"
        )
 
    with col6:
        antiguedad_cliente = st.number_input(
            "Antigüedad como Cliente (meses)", min_value=0.0, value=36.0, step=1.0
        )
        importe_solicitado = st.number_input(
            "Importe Solicitado", min_value=0.0, value=5000.0, step=100.0,
            help="Importe del crédito solicitado por el cliente."
        )
        duracion_credito = st.number_input(
            "Duración del Crédito (años)", min_value=1, max_value=30, value=3, step=1,
            help="Plazo de devolución del crédito en años."
        )
 
    st.divider()
 
    # ------------------------------------------------------------------
    # Sección 3: Datos de tarjeta de crédito
    # ------------------------------------------------------------------
    st.subheader("Datos de Tarjeta de Crédito")
    col7, col8 = st.columns(2)
 
    with col7:
        limite_credito_tc = st.number_input(
            "Límite de Crédito de la Tarjeta", min_value=0.0, value=10000.0, step=100.0
        )
        gastos_ult_12m = st.number_input(
            "Gastos con Tarjeta (últimos 12 meses)",
            min_value=0.0, value=1200.0, step=0.01, format="%.2f"
        )
 
    with col8:
        operaciones_ult_12m = st.number_input(
            "Operaciones con Tarjeta (últimos 12 meses)",
            min_value=1.0, value=42.0, step=1.0,
            help="Total de operaciones realizadas con tarjeta en el último año."
        )
 
    st.divider()
    submit_button = st.form_submit_button(
        "Predecir",
        use_container_width=True,
        type="primary"
    )
 
# ------------------------------------------------------------------
# Cálculo automático de variables derivadas (EXTRA)
# Fórmulas exactas del script data_prep.py
# ------------------------------------------------------------------
if submit_button:
 
    capacidad_pago               = importe_solicitado / ingresos
    operaciones_mensuales        = operaciones_ult_12m / 12
    presion_financiera           = (
        (gastos_ult_12m / 12 + importe_solicitado / (duracion_credito * 12))
        / (ingresos / 12)
    )
    gasto_prom_por_operacion     = gastos_ult_12m / operaciones_ult_12m
    operaciones_mensuales_tarjeta = operaciones_ult_12m / 12
    estabilidad_laboral          = antiguedad_empleado / edad
 
    input_data = {
        "edad": edad,
        "antiguedad_empleado": antiguedad_empleado,
        "situacion_vivienda": situacion_vivienda,
        "ingresos": ingresos,
        "objetivo_credito": objetivo_credito,
        "pct_ingreso": pct_ingreso,
        "tasa_interes": tasa_interes,
        "estado_credito": estado_credito,
        "antiguedad_cliente": antiguedad_cliente,
        "estado_civil": estado_civil,
        "estado_cliente": estado_cliente,
        "gastos_ult_12m": gastos_ult_12m,
        "genero": genero,
        "limite_credito_tc": limite_credito_tc,
        "nivel_educativo": nivel_educativo,
        "personas_a_cargo": personas_a_cargo,
        "capacidad_pago": capacidad_pago,
        "operaciones_mensuales": operaciones_mensuales,
        "presion_financiera": presion_financiera,
        "gasto_prom_por_operacion": gasto_prom_por_operacion,
        "operaciones_mensuales_tarjeta": operaciones_mensuales_tarjeta,
        "estabilidad_laboral": estabilidad_laboral,
    }
 
    # Mostrar variables calculadas automáticamente
    with st.expander("Variables derivadas calculadas automáticamente"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Capacidad de pago",            f"{capacidad_pago:.4f}")
        c1.metric("Operaciones mensuales",         f"{operaciones_mensuales:.4f}")
        c2.metric("Presión financiera",            f"{presion_financiera:.4f}")
        c2.metric("Gasto prom. por operación",     f"{gasto_prom_por_operacion:.4f}")
        c3.metric("Op. mensuales tarjeta",         f"{operaciones_mensuales_tarjeta:.4f}")
        c3.metric("Estabilidad laboral",           f"{estabilidad_laboral:.4f}")
 
    try:
        resp = requests.post(f"{api_url}/predict", json=input_data, timeout=10)
        resp.raise_for_status()
        result = resp.json()
 
        st.divider()
        st.subheader("📊 Resultado de la predicción")
 
        prediction = result["prediction"]
        prob       = result.get("probability", {})
        labels     = result.get("class_labels", {"0": "No entra en mora", "1": "Entra en mora"})
        label_text = labels.get(str(prediction), prediction)
 
        col_res1, col_res2 = st.columns(2)
 
        with col_res1:
            if str(prediction) == "1":
                st.error(f"**Predicción: {label_text}**")
            else:
                st.success(f"**Predicción: {label_text}**")
 
        with col_res2:
            prob_mora    = prob.get("1", prob.get(str(prediction), 0))
            prob_no_mora = prob.get("0", 1 - prob_mora)
            st.metric("Probabilidad de mora",    f"{prob_mora * 100:.1f}%")
            st.metric("Probabilidad de no mora", f"{prob_no_mora * 100:.1f}%")
 
        with st.expander("Ver respuesta completa de la API"):
            st.json(result)
 
    except requests.exceptions.ConnectionError:
        st.error("No se pudo conectar con la API. Verificá la URL en el panel lateral.")
    except requests.exceptions.HTTPError as e:
        st.error(f"Error de la API ({resp.status_code}): {resp.json().get('detail', str(e))}")
    except Exception as e:
        st.error(f"Error inesperado: {e}")