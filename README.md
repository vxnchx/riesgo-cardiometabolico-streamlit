# Predicción del nivel de riesgo cardiometabólico · App en Streamlit

Proyecto de aula de **Minería de Datos**, Universidad de la Costa (CUC). Tercer corte, actividad 1: integración y despliegue del modelo con Streamlit.

**Integrantes:** Iván David Pertuz · Diego Llanos
**Profesor:** José Escorcia-Gutierrez

🔗 **Aplicación pública:** `https://<pendiente>.streamlit.app` <!-- reemplazar por la URL de Streamlit Community Cloud -->
🔗 **Repositorio:** https://github.com/vxnchx/riesgo-cardiometabolico-streamlit

> ⚠️ **Prototipo académico.** Los datos son sintéticos. La aplicación no realiza diagnósticos médicos y no sustituye la valoración de un profesional de la salud.

## Descripción

La aplicación clasifica el **nivel de riesgo cardiometabólico (Bajo, Medio, Alto)** de una persona a partir de 11 variables clínicas y de estilo de vida: edad, género, IMC, glucosa, presión sistólica, colesterol, actividad física, tabaquismo, horas de sueño, nivel de estrés y consultas médicas.

- **Datos:** *Dataset Sintético de Salud para EDA* (`Dataset_Sintetico_Salud_EDA.xlsx`), proporcionado por el docente en Moodle. Tiene 500 registros sintéticos; quedan 497 tras retirar 3 duplicados exactos. No contiene datos de personas reales.
- **Modelo:** regresión logística multinomial dentro de un `Pipeline` de scikit-learn: imputación por mediana o moda, estandarización, *one-hot encoding* y clasificador. En el tercer corte se ajustó la regularización `C` con validación cruzada sobre el entrenamiento (notebook, sección 6).
- **La app no reentrena el modelo.** Carga el pipeline serializado `modelo/pipeline_riesgo.joblib`.

### Contenido de la aplicación

| Pestaña | Contenido |
|---|---|
| Proyecto | Objetivo, integrantes, fuente de los datos y variables |
| Exploración de datos | Distribución del objetivo, variables clínicas por nivel de riesgo (boxplots) y composición por variables categóricas |
| Predicción | Formulario con validación de tipos, rangos y categorías; clase predicha y probabilidades (`predict_proba`); carga de registros del conjunto de prueba para comparar con el notebook |
| Desempeño del modelo | Accuracy, precision, recall y F1 macro, matriz de confusión, métricas por clase, validación cruzada y comparación antes/después de la optimización |
| Limitaciones | Alcance y limitaciones del modelo |

## Estructura del repositorio

```
├── app.py                                 # aplicación Streamlit
├── requirements.txt                       # dependencias (versiones fijadas)
├── README.md
├── Avance3_Proyecto_Pertuz_Llanos.ipynb   # notebook actualizado (secciones 1-5 del corte 2 + sección 6 nueva)
├── Avance2_Proyecto_Pertuz_Llanos.ipynb   # notebook del segundo corte (referencia)
├── Dataset_Sintetico_Salud_EDA.xlsx       # datos (sintéticos)
├── modelo/
│   ├── pipeline_riesgo.joblib             # pipeline completo serializado
│   ├── metadatos.json                     # métricas, matriz de confusión, rangos y versiones
│   └── conjunto_prueba.csv                # 125 registros de prueba con la predicción del notebook
└── figuras/                               # figuras del EDA y la evaluación
```

## Ejecución local

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Para regenerar el modelo, ejecute el notebook `Avance3_Proyecto_Pertuz_Llanos.ipynb` de principio a fin. La sección 6.4 vuelve a crear los archivos de la carpeta `modelo/`.

## Despliegue en Streamlit Community Cloud

1. Subir este repositorio a GitHub como **público**.
2. En <https://share.streamlit.io>, elegir **Create app**, luego el repositorio, la rama `main` y el archivo `app.py`.
3. En *Advanced settings*, elegir **Python 3.12**, la misma versión con la que se entrenó y serializó el pipeline.
4. Abrir la URL pública y comprobar las pruebas de la tabla siguiente.

## Pruebas de funcionamiento

| Prueba | Entrada | Resultado esperado |
|---|---|---|
| Caso normal | Botón **Caso normal** (45 años, IMC 26.5, glucosa 100, presión 120, colesterol 190, 7 h de sueño, estrés 5, 2 consultas, Femenino, actividad Moderada, no fuma) | **Bajo** (P = 0.505; Medio 0.492; Alto 0.003), igual que la celda "Caso normal" del notebook (sección 6.5). Como las dos clases más probables están cerca, la app avisa que el caso está en una zona de transición |
| Entrada inválida | Por ejemplo, glucosa = 900, edad = 10 o nivel de estrés = 15 | Mensaje de error por cada campo fuera de rango; **no** se predice |
| Registro conocido del conjunto de prueba | Seleccionar un `ID_Paciente` y pulsar **Cargar registro**, luego **Predecir** | La tabla de verificación muestra que la clase y las probabilidades coinciden con las del notebook (`modelo/conjunto_prueba.csv`) |

Además, el notebook (sección 6.5) comprueba con `assert` que el pipeline cargado desde disco reproduce exactamente las predicciones y probabilidades de los 125 registros de prueba.

## Resultados del modelo final

Regresión logística multinomial (`C = 1`, `class_weight="balanced"`), evaluada en el conjunto de prueba (n = 125):

| Métrica | Valor |
|---|---|
| Accuracy | 0.856 |
| Precision / Recall / F1 (macro) | 0.854 / 0.858 / 0.856 |
| Recall de la clase Alto | 0.848 (5 de 33 pacientes Alto quedaron como Medio) |
| F1-macro, validación cruzada de 5 particiones | 0.844 ± 0.030 |
| Brecha entre entrenamiento y prueba (accuracy) | 0.015 |
| Confusiones Bajo ↔ Alto | 0 |

**Optimización del tercer corte (notebook, sección 6):** se buscaron `C` y los pesos de clase con `GridSearchCV`, usando solo el entrenamiento. La mejor combinación (`C = 0.3`) **no mejoró** al modelo original en la validación cruzada completa (0.835 frente a 0.844) ni en prueba (F1 0.820 frente a 0.856), así que se conservó el modelo del segundo corte. Ninguna métrica queda por debajo de la alerta del 70 %.

## Uso de inteligencia artificial

| Aspecto | Detalle |
|---|---|
| **Herramienta** | Claude Code (modelo Claude Opus 5.5, Anthropic) |
| **Propósito** | Apoyo para estructurar la sección 6 del notebook (búsqueda de hiperparámetros, comparación antes/después, exportación con `joblib`), escribir `app.py` (formulario, validación, visualizaciones con Plotly) y redactar este README. |
| **Prompt principal** | "Realiza esta actividad", adjuntando el enunciado *Actividad 1 – Tercer corte: Integración y despliegue del modelo con Streamlit*. La herramienta trabajó sobre el notebook, el dataset y las figuras del segundo corte. |
| **Cómo se verificó** | (1) Se ejecutó el notebook completo y se comprobó que las métricas del segundo corte se reproducen. (2) Un `assert` comprueba que el pipeline exportado da las mismas predicciones y probabilidades que el notebook en los 125 registros de prueba. (3) Se probó la app con un caso normal, entradas inválidas y registros del conjunto de prueba, y se comparó con `conjunto_prueba.csv`. (4) Revisamos el código línea por línea para entender cada parte. |
| **Problemas encontrados** | El intérprete del sistema (Python 3.9) no tenía las librerías y Streamlit Cloud usa una versión más reciente, así que se creó un entorno con Python 3.12 y se fijaron las versiones en `requirements.txt` para que el `.joblib` cargue sin incompatibilidades. Con `min_value`/`max_value` en los campos numéricos, Streamlit impide escribir valores inválidos y no se podía mostrar el mensaje de error, por eso la validación se hace con una función propia (`validar_entrada`). |
| **Ajustes realizados** | La versión optimizada se adopta solo si mejora el F1-macro de validación cruzada sin reducir el recall de Alto (no solo por el accuracy). Se añadió una tabla en la app que compara la predicción con la del notebook para el mismo registro. Se agregaron advertencias de extrapolación fuera del rango de entrenamiento y avisos de prototipo académico. |

Todos los integrantes revisaron y comprenden el código entregado.
