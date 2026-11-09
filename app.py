"""Aplicación Streamlit · Predicción del nivel de riesgo cardiometabólico.

Proyecto de aula de Minería de Datos (Universidad de la Costa, CUC).
Integrantes: Iván David Pertuz · Diego Llanos.

La aplicación carga el pipeline entrenado en el notebook (modelo/pipeline_riesgo.joblib)
y NO vuelve a entrenar el modelo.
"""

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------------------
BASE = Path(__file__).parent
RUTA_MODELO = BASE / "modelo" / "pipeline_riesgo.joblib"
RUTA_META = BASE / "modelo" / "metadatos.json"
RUTA_PRUEBA = BASE / "modelo" / "conjunto_prueba.csv"
RUTA_DATOS = BASE / "Dataset_Sintetico_Salud_EDA.xlsx"

ORDEN_RIESGO = ["Bajo", "Medio", "Alto"]
COLORES_RIESGO = {"Bajo": "#59A14F", "Medio": "#EDC948", "Alto": "#E15759"}

st.set_page_config(
    page_title="Riesgo cardiometabólico · Minería de Datos CUC",
    page_icon="🩺",
    layout="wide",
)

# Etiquetas, unidades y rangos de validación de cada variable del modelo.
# "rango" = valores fisiológicamente admisibles en el formulario (fuera de ellos la entrada se rechaza).
# El rango observado en los datos de entrenamiento se lee de metadatos.json y se usa para advertir
# cuando el modelo tendría que extrapolar.
VARIABLES_NUMERICAS = {
    "Edad": {"etiqueta": "Edad", "unidad": "años", "rango": (18, 100), "entero": True, "paso": 1,
             "ayuda": "Edad en años cumplidos. El dataset solo incluye adultos (18 años o más)."},
    "IMC": {"etiqueta": "Índice de masa corporal (IMC)", "unidad": "kg/m²", "rango": (10.0, 70.0),
            "entero": False, "paso": 0.1, "ayuda": "Peso (kg) dividido por la estatura al cuadrado (m²)."},
    "Glucosa_mg_dL": {"etiqueta": "Glucosa en sangre", "unidad": "mg/dL", "rango": (40.0, 400.0),
                      "entero": False, "paso": 1.0, "ayuda": "Glucosa medida en ayunas."},
    "Presion_Sistolica_mmHg": {"etiqueta": "Presión arterial sistólica", "unidad": "mmHg", "rango": (70, 250),
                               "entero": True, "paso": 1, "ayuda": "Valor 'alto' de la presión arterial."},
    "Colesterol_mg_dL": {"etiqueta": "Colesterol total", "unidad": "mg/dL", "rango": (80.0, 400.0),
                         "entero": False, "paso": 1.0, "ayuda": "Colesterol total en sangre."},
    "Horas_Sueno": {"etiqueta": "Horas de sueño", "unidad": "horas por noche", "rango": (0.0, 14.0),
                    "entero": False, "paso": 0.1, "ayuda": "Promedio de horas dormidas por noche."},
    "Nivel_Estres": {"etiqueta": "Nivel de estrés", "unidad": "escala 1 a 10", "rango": (1, 10),
                     "entero": True, "paso": 1, "ayuda": "Autorreporte: 1 = muy bajo, 10 = muy alto."},
    "Consultas_Medicas": {"etiqueta": "Consultas médicas", "unidad": "en los últimos 12 meses", "rango": (0, 30),
                          "entero": True, "paso": 1, "ayuda": "Número de consultas médicas en el último año."},
}
VARIABLES_CATEGORICAS = {
    "Genero": {"etiqueta": "Género", "opciones": ["Femenino", "Masculino", "Otro/No reporta"]},
    "Actividad_Fisica": {"etiqueta": "Actividad física", "opciones": ["Baja", "Moderada", "Alta"]},
    "Fumador": {"etiqueta": "¿Fuma?", "opciones": ["No", "Sí"]},
}
ETIQUETAS = {**{k: v["etiqueta"] for k, v in VARIABLES_NUMERICAS.items()},
             **{k: v["etiqueta"] for k, v in VARIABLES_CATEGORICAS.items()}}

CASO_NORMAL = {"Edad": 45, "IMC": 26.5, "Glucosa_mg_dL": 100.0, "Presion_Sistolica_mmHg": 120,
               "Colesterol_mg_dL": 190.0, "Horas_Sueno": 7.0, "Nivel_Estres": 5, "Consultas_Medicas": 2,
               "Genero": "Femenino", "Actividad_Fisica": "Moderada", "Fumador": "No"}


# ---------------------------------------------------------------------------
# Carga de recursos (en caché: se cargan una sola vez)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando el modelo…")
def cargar_modelo():
    return joblib.load(RUTA_MODELO)


@st.cache_data
def cargar_metadatos():
    return json.loads(RUTA_META.read_text(encoding="utf-8"))


@st.cache_data
def cargar_datos():
    df = pd.read_excel(RUTA_DATOS, sheet_name="Datos").drop_duplicates()
    df["Nivel_Riesgo"] = pd.Categorical(df["Nivel_Riesgo"], categories=ORDEN_RIESGO, ordered=True)
    return df


@st.cache_data
def cargar_prueba():
    return pd.read_csv(RUTA_PRUEBA)


# ---------------------------------------------------------------------------
# Validación de la entrada
# ---------------------------------------------------------------------------
def validar_entrada(entrada: dict, rangos_observados: dict):
    """Comprueba tipos, rangos y categorías. Devuelve (errores, advertencias)."""
    errores, advertencias = [], []
    for var, cfg in VARIABLES_NUMERICAS.items():
        valor = entrada.get(var)
        nombre = f"**{cfg['etiqueta']}**"
        if valor is None or (isinstance(valor, float) and math.isnan(valor)):
            errores.append(f"{nombre}: falta el valor.")
            continue
        if isinstance(valor, bool) or not isinstance(valor, (int, float, np.integer, np.floating)):
            errores.append(f"{nombre}: debe ser un número (se recibió `{valor!r}`).")
            continue
        if cfg["entero"] and float(valor) != int(valor):
            errores.append(f"{nombre}: debe ser un número entero (se recibió {valor}).")
            continue
        minimo, maximo = cfg["rango"]
        if not minimo <= valor <= maximo:
            errores.append(f"{nombre}: el valor {valor:g} está fuera del rango válido "
                           f"({minimo:g} a {maximo:g}; {cfg['unidad']}).")
            continue
        obs_min, obs_max = rangos_observados[var]
        if not obs_min <= valor <= obs_max:
            advertencias.append(f"{nombre}: {valor} {cfg['unidad']} está fuera del rango visto en el "
                                f"entrenamiento ({obs_min:g} a {obs_max:g}). El modelo extrapola y la "
                                f"predicción es menos confiable.")
    for var, cfg in VARIABLES_CATEGORICAS.items():
        valor = entrada.get(var)
        if valor not in cfg["opciones"]:
            errores.append(f"**{cfg['etiqueta']}**: categoría no válida `{valor!r}`. "
                           f"Opciones: {', '.join(cfg['opciones'])}.")
    return errores, advertencias


def fijar_formulario(valores: dict):
    """Copia un registro en los widgets del formulario."""
    for var, cfg in VARIABLES_NUMERICAS.items():
        v = valores[var]
        if pd.isna(v):
            v = None
        elif cfg["entero"]:
            v = int(v)
        else:
            v = float(v)
        st.session_state[f"in_{var}"] = v
    for var, cfg in VARIABLES_CATEGORICAS.items():
        v = valores[var]
        st.session_state[f"in_{var}"] = v if v in cfg["opciones"] else None


# ---------------------------------------------------------------------------
# Comprobación de archivos
# ---------------------------------------------------------------------------
faltantes = [p.name for p in (RUTA_MODELO, RUTA_META, RUTA_PRUEBA, RUTA_DATOS) if not p.exists()]
if faltantes:
    st.error("Faltan archivos necesarios: " + ", ".join(faltantes)
             + ". Ejecute primero el notebook (sección 6.4) para exportar el modelo.")
    st.stop()

modelo = cargar_modelo()
meta = cargar_metadatos()
datos = cargar_datos()
prueba = cargar_prueba()
CLASES_MODELO = list(modelo.classes_)

# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🩺 Riesgo cardiometabólico")
    st.caption("Minería de Datos · Universidad de la Costa (CUC)")
    st.markdown("**Integrantes**  \nIván David Pertuz  \nDiego Llanos")
    st.markdown("**Profesor**  \nJosé Escorcia-Gutierrez")
    st.divider()
    st.markdown(f"**Modelo:** {meta['modelo']}  \n"
                f"**C** = {meta['hiperparametros']['C']:g} · pesos = {meta['hiperparametros']['class_weight']}")
    st.markdown(f"**F1-macro (prueba):** {meta['metricas_prueba']['f1_macro']:.3f}  \n"
                f"**F1-macro (CV 5 particiones):** {meta['validacion_cruzada']['f1_macro_media']:.3f}")
    st.divider()
    st.warning("**Prototipo académico.** Los datos son sintéticos. Esta aplicación **no** realiza "
               "diagnósticos ni sustituye la valoración de un profesional de la salud.", icon="⚠️")

# ---------------------------------------------------------------------------
# Encabezado y pestañas
# ---------------------------------------------------------------------------
st.title("Predicción del nivel de riesgo cardiometabólico")
st.markdown("Clasificación supervisada del nivel de riesgo **(Bajo, Medio, Alto)** a partir de indicadores "
            "clínicos y de estilo de vida, con un conjunto de datos **sintético** de salud.")

tab_proy, tab_eda, tab_pred, tab_metr, tab_lim = st.tabs(
    ["📘 Proyecto", "📊 Exploración de datos", "🔮 Predicción", "📈 Desempeño del modelo", "⚠️ Limitaciones"]
)

# ---------------------------------------------------------------------------
# 1. Proyecto
# ---------------------------------------------------------------------------
with tab_proy:
    col1, col2 = st.columns([3, 2], gap="large")
    with col1:
        st.subheader("Objetivo")
        st.markdown(
            "Responder la pregunta: *¿es posible clasificar el nivel de riesgo (Bajo, Medio, Alto) de una persona "
            "a partir de sus indicadores clínicos y de estilo de vida, y cuáles de esas características se asocian "
            "más con la clasificación?*\n\n"
            "Se compararon tres clasificadores (regresión logística, árbol de decisión y random forest). Se "
            "seleccionó la **regresión logística multinomial** por su mejor F1-macro en prueba y en validación "
            "cruzada, su baja brecha entre entrenamiento y prueba, su recall en la clase Alto y la "
            "interpretabilidad de sus coeficientes. En el tercer corte se ajustó su regularización con validación "
            "cruzada y el pipeline completo se exportó con `joblib` para esta aplicación."
        )
        st.subheader("Fuente de los datos")
        st.markdown(
            "*Dataset Sintético de Salud para EDA* (`Dataset_Sintetico_Salud_EDA.xlsx`), proporcionado por el "
            "docente en el aula virtual (Moodle) de la asignatura Minería de Datos de la CUC. Contiene 500 "
            "registros **generados sintéticamente** (enero–diciembre de 2025); no corresponden a pacientes reales. "
            "Tras retirar 3 duplicados exactos quedan **497 registros**. La variable objetivo `Nivel_Riesgo` es "
            "una etiqueta didáctica construida a partir de las demás variables."
        )
        st.subheader("Cómo usar la aplicación")
        st.markdown(
            "1. **Exploración de datos:** visualizaciones clave del análisis exploratorio.\n"
            "2. **Predicción:** ingrese los datos de una persona (o cargue un registro del conjunto de prueba) "
            "y obtenga la clase predicha con sus probabilidades.\n"
            "3. **Desempeño del modelo:** métricas finales, matriz de confusión y validación cruzada.\n"
            "4. **Limitaciones:** qué no puede hacer el modelo."
        )
    with col2:
        st.subheader("Integrantes")
        st.markdown("- Iván David Pertuz\n- Diego Llanos")
        st.markdown("**Profesor:** José Escorcia-Gutierrez  \n**Universidad de la Costa (CUC)** · Departamento "
                    "de Ciencias de la Computación y Electrónica")
        st.subheader("Datos en cifras")
        a, b = st.columns(2)
        a.metric("Registros", f"{len(datos)}")
        b.metric("Variables predictoras", "11")
        a.metric("Entrenamiento", f"{meta['n_entrenamiento']}")
        b.metric("Prueba", f"{meta['n_prueba']}")
        st.subheader("Variables del modelo")
        st.dataframe(
            pd.DataFrame(
                [{"Variable": v["etiqueta"], "Unidad / valores": v["unidad"]} for v in VARIABLES_NUMERICAS.values()]
                + [{"Variable": v["etiqueta"], "Unidad / valores": ", ".join(v["opciones"])}
                   for v in VARIABLES_CATEGORICAS.values()]
            ),
            hide_index=True, width="stretch",
        )

# ---------------------------------------------------------------------------
# 2. Exploración de datos (EDA)
# ---------------------------------------------------------------------------
with tab_eda:
    st.subheader("1. Distribución de la variable objetivo")
    conteo = datos["Nivel_Riesgo"].value_counts().reindex(ORDEN_RIESGO).rename_axis("Nivel").reset_index(name="n")
    conteo["%"] = (conteo["n"] / conteo["n"].sum() * 100).round(1)
    fig = px.bar(conteo, x="Nivel", y="n", color="Nivel", color_discrete_map=COLORES_RIESGO,
                 text=conteo.apply(lambda r: f"{r['n']} ({r['%']} %)", axis=1),
                 labels={"n": "Número de registros", "Nivel": "Nivel de riesgo"})
    fig.update_layout(showlegend=False, height=360, margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")
    st.caption("Desbalance leve (razón 1.48 entre la clase mayoritaria y la minoritaria). Por eso se usó "
               "división estratificada, pesos balanceados y métricas *macro*.")

    st.subheader("2. Variables clínicas según el nivel de riesgo")
    var = st.selectbox("Variable", list(VARIABLES_NUMERICAS),
                       format_func=lambda v: f"{ETIQUETAS[v]} ({VARIABLES_NUMERICAS[v]['unidad']})", index=0)
    fig = px.box(datos, x="Nivel_Riesgo", y=var, color="Nivel_Riesgo", color_discrete_map=COLORES_RIESGO,
                 category_orders={"Nivel_Riesgo": ORDEN_RIESGO}, points="outliers",
                 labels={"Nivel_Riesgo": "Nivel de riesgo",
                         var: f"{ETIQUETAS[var]} ({VARIABLES_NUMERICAS[var]['unidad']})"})
    fig.update_layout(showlegend=False, height=420, margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")
    st.caption("Edad, IMC, colesterol, presión y glucosa aumentan de forma ordenada de Bajo a Alto (diferencias "
               "estandarizadas Alto–Bajo entre 1.2 y 1.8 desviaciones estándar). Sueño y estrés casi no "
               "diferencian las clases.")

    st.subheader("3. Composición del riesgo según variables categóricas")
    cat = st.radio("Variable categórica", list(VARIABLES_CATEGORICAS), index=1, horizontal=True,
                   format_func=lambda v: ETIQUETAS[v])
    tabla = (pd.crosstab(datos[cat], datos["Nivel_Riesgo"], normalize="index") * 100).round(1)
    tabla = tabla.reindex([o for o in VARIABLES_CATEGORICAS[cat]["opciones"] if o in tabla.index])
    larga = tabla.reset_index().melt(id_vars=cat, var_name="Nivel de riesgo", value_name="%")
    fig = px.bar(larga, x=cat, y="%", color="Nivel de riesgo", color_discrete_map=COLORES_RIESGO,
                 category_orders={"Nivel de riesgo": ORDEN_RIESGO}, text="%",
                 labels={cat: ETIQUETAS[cat], "%": "% dentro de la categoría"})
    fig.update_layout(barmode="stack", height=400, margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")
    st.caption("La proporción de riesgo Alto baja de 42.4 % con actividad física baja a 9.3 % con actividad "
               "alta, y es mayor entre fumadores (37.8 %) que entre no fumadores (23.3 %). Son asociaciones en "
               "datos sintéticos, no relaciones causales.")

# ---------------------------------------------------------------------------
# 3. Predicción
# ---------------------------------------------------------------------------
with tab_pred:
    if "inicializado" not in st.session_state:
        fijar_formulario(CASO_NORMAL)
        st.session_state["inicializado"] = True
        st.session_state["registro_cargado"] = None

    st.subheader("Cargar datos de ejemplo (opcional)")
    c1, c2, c3 = st.columns([3, 1, 1], vertical_alignment="bottom")
    ids = prueba["ID_Paciente"].tolist()
    id_sel = c1.selectbox(
        "Registro del conjunto de prueba",
        ids,
        format_func=lambda i: f"{i} · real: {prueba.set_index('ID_Paciente').loc[i, 'Nivel_Riesgo_real']}",
        help="Los 125 registros que el modelo no vio durante el entrenamiento.",
    )
    if c2.button("Cargar registro", width="stretch"):
        fijar_formulario(prueba.set_index("ID_Paciente").loc[id_sel].to_dict())
        st.session_state["registro_cargado"] = id_sel
    if c3.button("Caso normal", width="stretch"):
        fijar_formulario(CASO_NORMAL)
        st.session_state["registro_cargado"] = None

    st.subheader("Datos de la persona")
    with st.form("formulario"):
        cols = st.columns(3)
        entrada = {}
        for i, (var, cfg) in enumerate(VARIABLES_NUMERICAS.items()):
            # Sin min/max en el widget: la validación propia muestra mensajes claros si el valor es inválido.
            entrada[var] = cols[i % 3].number_input(
                f"{cfg['etiqueta']} ({cfg['unidad']})",
                step=cfg["paso"],
                format="%d" if cfg["entero"] else ("%.1f" if cfg["paso"] < 1 else "%.0f"),
                help=f"{cfg['ayuda']} Rango válido: {cfg['rango'][0]} a {cfg['rango'][1]}.",
                key=f"in_{var}",
            )
        for j, (var, cfg) in enumerate(VARIABLES_CATEGORICAS.items()):
            entrada[var] = cols[j % 3].selectbox(cfg["etiqueta"], cfg["opciones"], key=f"in_{var}",
                                                 placeholder="Seleccione una opción")
        enviado = st.form_submit_button("Predecir nivel de riesgo", type="primary", width="stretch")

    if enviado:
        errores, advertencias = validar_entrada(entrada, meta["rangos_observados"])
        if errores:
            st.error("**No se puede predecir. Corrija los siguientes datos:**\n\n"
                     + "\n".join(f"- {e}" for e in errores))
        else:
            for a in advertencias:
                st.warning(a)
            X_nuevo = pd.DataFrame([entrada])[meta["variables_numericas"] + meta["variables_categoricas"]]
            clase = modelo.predict(X_nuevo)[0]
            proba = dict(zip(CLASES_MODELO, modelo.predict_proba(X_nuevo)[0]))

            st.divider()
            r1, r2 = st.columns([1, 2], gap="large")
            with r1:
                st.markdown("#### Resultado")
                st.markdown(
                    f"<div style='padding:1rem;border-radius:0.6rem;background:{COLORES_RIESGO[clase]}22;"
                    f"border-left:8px solid {COLORES_RIESGO[clase]}'>"
                    f"<div style='font-size:0.95rem'>Nivel de riesgo predicho</div>"
                    f"<div style='font-size:2.2rem;font-weight:700'>{clase}</div>"
                    f"<div>Probabilidad: <b>{proba[clase]:.1%}</b></div></div>",
                    unsafe_allow_html=True,
                )
                orden = sorted(proba.values(), reverse=True)
                if orden[0] - orden[1] < 0.15:
                    st.info("Las dos clases más probables están cerca: el caso está en una zona de transición "
                            "y la predicción es incierta.")
            with r2:
                st.markdown("#### Probabilidad de cada clase (`predict_proba`)")
                dfp = pd.DataFrame({"Nivel": ORDEN_RIESGO, "Probabilidad": [proba[c] for c in ORDEN_RIESGO]})
                fig = px.bar(dfp, x="Probabilidad", y="Nivel", orientation="h", color="Nivel",
                             color_discrete_map=COLORES_RIESGO, text=dfp["Probabilidad"].map(lambda p: f"{p:.1%}"),
                             category_orders={"Nivel": ORDEN_RIESGO[::-1]}, range_x=[0, 1])
                fig.update_layout(showlegend=False, height=230, margin=dict(t=10, b=10), xaxis_tickformat=".0%")
                st.plotly_chart(fig, width="stretch")

            id_cargado = st.session_state.get("registro_cargado")
            if id_cargado is not None:
                fila = prueba.set_index("ID_Paciente").loc[id_cargado]
                original = {v: fila[v] for v in list(VARIABLES_NUMERICAS) + list(VARIABLES_CATEGORICAS)}
                sin_cambios = all(
                    (pd.isna(original[v]) and entrada[v] is None)
                    or (not pd.isna(original[v]) and entrada[v] is not None
                        and (entrada[v] == original[v] if v in VARIABLES_CATEGORICAS
                             else math.isclose(float(entrada[v]), float(original[v]))))
                    for v in original
                )
                if sin_cambios:
                    st.markdown(f"#### Verificación con el notebook · registro `{id_cargado}`")
                    comp = pd.DataFrame({
                        "": ["Clase real", "Predicción en el notebook", "Predicción en Streamlit"]
                            + [f"P({c}) notebook / Streamlit" for c in ORDEN_RIESGO],
                        "Valor": [fila["Nivel_Riesgo_real"], fila["Prediccion_notebook"], clase]
                            + [f"{fila[f'Prob_{c}']:.4f} / {proba[c]:.4f}" for c in ORDEN_RIESGO],
                    })
                    st.dataframe(comp, hide_index=True, width="stretch")
                    coincide = (clase == fila["Prediccion_notebook"]) and all(
                        abs(proba[c] - fila[f"Prob_{c}"]) < 1e-4 for c in ORDEN_RIESGO)
                    if coincide:
                        st.success("La predicción y las probabilidades coinciden con las del notebook.")
                    else:
                        st.error("La predicción no coincide con la del notebook. Revise las versiones de las librerías.")
                    if fila["Nivel_Riesgo_real"] != clase:
                        st.info("Este registro es uno de los errores del modelo en el conjunto de prueba.")

            with st.expander("Ver los datos enviados al modelo"):
                st.dataframe(X_nuevo.rename(columns=ETIQUETAS), hide_index=True, width="stretch")

    st.caption("⚠️ Prototipo académico con datos sintéticos. El resultado no es un diagnóstico y no sustituye la "
               "valoración de un profesional de la salud.")

# ---------------------------------------------------------------------------
# 4. Desempeño del modelo
# ---------------------------------------------------------------------------
with tab_metr:
    m = meta["metricas_prueba"]
    vc = meta["validacion_cruzada"]
    st.subheader(f"Métricas finales en el conjunto de prueba (n = {meta['n_prueba']})")
    k = st.columns(5)
    k[0].metric("Accuracy", f"{m['accuracy']:.3f}")
    k[1].metric("Precision (macro)", f"{m['precision_macro']:.3f}")
    k[2].metric("Recall (macro)", f"{m['recall_macro']:.3f}")
    k[3].metric("F1-score (macro)", f"{m['f1_macro']:.3f}")
    k[4].metric("F1 macro · CV 5 particiones", f"{vc['f1_macro_media']:.3f}", f"± {vc['f1_macro_desv']:.3f}",
                delta_color="off")

    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.markdown("#### Matriz de confusión")
        cm = np.array(meta["matriz_confusion"])
        fig = go.Figure(go.Heatmap(
            z=cm, x=[f"Pred. {c}" for c in ORDEN_RIESGO], y=[f"Real {c}" for c in ORDEN_RIESGO],
            text=cm, texttemplate="%{text}", textfont={"size": 18}, colorscale="Blues", showscale=False,
            hovertemplate="%{y} → %{x}: %{z}<extra></extra>",
        ))
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(height=380, margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")
        alto_no = int(cm[2, 0] + cm[2, 1])
        bajo_alto = int(cm[0, 2] + cm[2, 0])
        st.caption(f"Filas = clase real; columnas = clase predicha. {int(np.trace(cm))} de {int(cm.sum())} aciertos. "
                   f"Pacientes Alto no detectados: **{alto_no}**. Confusiones graves Bajo ↔ Alto: **{bajo_alto}**.")
    with c2:
        st.markdown("#### Métricas por clase")
        pc = pd.DataFrame(meta["metricas_por_clase"]).T.loc[ORDEN_RIESGO]
        pc.columns = ["Precision", "Recall", "F1-score", "Soporte"]
        pc["Soporte"] = pc["Soporte"].astype(int)
        st.dataframe(pc.style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1-score": "{:.3f}"}),
                     width="stretch")
        st.markdown("#### Validación cruzada estratificada (5 particiones)")
        fig = px.bar(x=[f"P{i + 1}" for i in range(len(vc["f1_macro_particiones"]))], y=vc["f1_macro_particiones"],
                     labels={"x": "Partición", "y": "F1-macro"}, text=[f"{v:.3f}" for v in vc["f1_macro_particiones"]])
        fig.add_hline(y=vc["f1_macro_media"], line_dash="dash", annotation_text=f"media {vc['f1_macro_media']:.3f}")
        fig.add_hline(y=0.70, line_dash="dot", line_color="red", annotation_text="alerta 0.70",
                      annotation_position="bottom right")
        fig.update_layout(height=260, margin=dict(t=10), yaxis_range=[0, 1])
        st.plotly_chart(fig, width="stretch")

    st.markdown("#### Optimización del tercer corte: antes y después")
    ad = pd.DataFrame(meta["antes_despues"]).set_index("Versión")
    cols_ad = ["Accuracy", "Precision_macro", "Recall_macro", "F1_macro", "Recall_Alto", "Accuracy_train",
               "F1_macro_CV", "F1_macro_CV_desv", "Alto_no_detectados"]
    st.dataframe(ad[cols_ad].style.format("{:.3f}", subset=cols_ad[:-1]), width="stretch")
    st.caption(f"Versión usada en la aplicación: **{meta['version_seleccionada']}** "
               f"(C = {meta['hiperparametros']['C']:g}, class_weight = {meta['hiperparametros']['class_weight']}). "
               "La regularización se ajustó con validación cruzada solo sobre el entrenamiento; se adopta la versión "
               "optimizada únicamente si mejora el F1-macro de CV sin reducir el recall de la clase Alto.")

    with st.expander("Comparación de los tres modelos del segundo corte"):
        cm2 = pd.DataFrame(meta["comparacion_modelos_corte2"]).set_index("Modelo")
        st.dataframe(cm2.style.format("{:.3f}"), width="stretch")
        st.caption("La regresión logística obtuvo el mejor F1-macro en prueba y en validación cruzada. El random "
                   "forest sobreajusta (F1 de entrenamiento ≈ 0.985) y el árbol de decisión rinde menos.")

# ---------------------------------------------------------------------------
# 5. Limitaciones
# ---------------------------------------------------------------------------
with tab_lim:
    st.subheader("Limitaciones del modelo")
    st.markdown(f"""
1. **Datos sintéticos.** Los 497 registros fueron generados por simulación y la etiqueta `Nivel_Riesgo` se construyó
   a partir de las mismas variables. El desempeño mide qué tan bien el modelo recupera esa regla didáctica, **no** su
   validez clínica. No debe usarse con pacientes reales.
2. **Errores entre clases vecinas.** Todos los errores relevantes ocurren entre Bajo–Medio y Medio–Alto. La clase
   **Medio** es la más difícil y algunos pacientes **Alto** se clasifican como Medio ({int(np.array(meta['matriz_confusion'])[2, 1])}
   en el conjunto de prueba). Por eso la app muestra siempre las probabilidades de las tres clases.
3. **Tamaño de muestra moderado** (372 registros de entrenamiento). El grupo `Otro/No reporta` en género tiene solo
   12 personas, así que las predicciones para ese grupo son poco fiables.
4. **Extrapolación.** Fuera de los rangos observados en el entrenamiento (por ejemplo, edades mayores de
   {meta['rangos_observados']['Edad'][1]:g} años) el modelo extrapola; la app lo advierte.
5. **Asociación no es causalidad.** Que la glucosa o la actividad física pesen en el modelo no significa que
   modificarlas cambie el riesgo real de una persona.
6. **Modelo lineal.** La regresión logística supone efectos aditivos en la escala log-odds; no captura interacciones
   complejas. No se exploró un modelo ordinal ni el ajuste del umbral de decisión.
""")
    st.error("**Esta aplicación es un prototipo académico.** No realiza diagnósticos médicos y no sustituye la "
             "decisión de un profesional de la salud.", icon="⚠️")
    st.caption(f"Versiones con las que se entrenó el modelo: "
               + " · ".join(f"{k} {v}" for k, v in meta["versiones"].items()))
