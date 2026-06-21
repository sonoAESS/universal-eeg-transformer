# Universal EEG Transformer: Multi-Input, Multi-Output Linear Autoencoder for Cross-Referencing Estandardization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Este repositorio contiene la implementación oficial del **Universal EEG Transformer**, una arquitectura basada en redes neuronales diseñada para resolver el problema de la interoperabilidad de bases de datos de EEG. A diferencia de los métodos de conversión tradicionales restrictivos ($X \rightarrow Y$), este modelo utiliza un **Autoencoder Lineal Multicabezal All-to-All** capaz de mapear y transformar señales de manera bidireccional y simultánea entre múltiples referencias físicas y algebraicas a través de un **Espacio Latente Central Universal**.

---

## 🚀 Características Clave

* **Mapeo Simétrico Completo ($5 \times 5$):** Un único modelo compacto que aprende de forma simultánea las 25 combinaciones posibles de transformación entre referencias de EEG.
* **Modelado Biofísico Integrado:** Generación automática de matrices *Lead Field* mediante simulación geométrica tridimensional (Sistema 10-20) con modelos de esferas concéntricas homogéneas.
* **Soporte de Montajes Críticos:** Procesamiento integrado de datos en crudo (Ground Truth), Unipolar, Bipolar, Centering (CAR) y la Referencia al Infinito Virtual (REST).
* **Función de Pérdida Estandarizada:** Optimización balanceada vía Z-Score interno por lote para neutralizar discrepancias de amplitud entre escalas microvoltio ($10^{-5}\text{ V}$) y voltio ($10^{-3}\text{ V}$).

---

## 🛠️ Estructura del Repositorio

```text
├── data/                             # Archivos CSV generados con los montajes (Ignorados en Git)
├── src/
│   ├── data_generation.py            # Pipeline de preprocesamiento biofísico y MNE
│   └── model_training.ipynb             # Arquitectura del transformador y plots de evaluación
├── README.md                         # Documentación del proyecto
└── requirements.txt                  # Dependencias del entorno

```

---

## 💻 Requisitos e Instalación

Se recomienda el uso de un entorno virtual limpio (`venv` o `conda`) ejecutando sobre **Python 3.8+**.

1. **Clonar el repositorio:**
```bash
git clone [https://github.com/sonoAESS/universal-eeg-transformer.git](https://github.com/sonoAESS/universal-eeg-transformer.git)
cd universal-eeg-transformer

```


2. **Instalar dependencias:**
```bash
pip install -r requirements.txt

```


*Nota: El archivo `requirements.txt` incluye principalmente: `mne`, `numpy`, `pandas`, `scikit-learn`, `tensorflow` y `matplotlib`.*

---

## ⚙️ Flujo de Ejecución

El pipeline de trabajo está dividido en dos fases secuenciales e independientes:

### Paso 1: Generación de Montajes Biofísicos

Ejecuta el script de preparación de datos. Este proceso descargará de forma automática el dataset público `eegbci` de PhysioNet, alineará el mapa 3D estándar 10-20, simulará 500 fuentes dipolares internas y exportará los 5 archivos estructurados dentro del directorio `data/`.

```bash
python src/data_generation.py

```

### Paso 2: Entrenamiento y Evaluación Cruzada

Corre el script del modelo para instanciar la red neuronal, compilarla con la función de pérdida estandarizada y entrenarla. Al finalizar las épocas, el script desplegará en pantalla la curva de aprendizaje y la **Matriz de Transferencia Completa** sobre datos de test ciegos.

```bash
python src/model_training.py

```

---

## 📊 Arquitectura y Pérdida

La arquitectura consiste en compuertas de codificación y decodificación puramente lineales, garantizando que el modelo respete las propiedades algebraicas subyacentes de los campos eléctricos del cuero cabelludo:

```
       [ Entrada: Cualquier Referencia ]
                       │
                       ▼ (Encoder Dedicado)
         [ Espacio Latente: Ground Truth ]
          /        │         │        \
         ▼         ▼         ▼         ▼ (Decoder Heads)
  [Unipolar]   [Bipolar]   [CAR]   [Infinito]

```

Para asegurar que las señales más pequeñas de microvoltios no queden sepultadas numéricamente por montajes de mayor amplitud como REST, la retropropagación evalúa la pérdida en una escala adimensional utilizando Z-Score instantáneo:

$$\text{Pérdida} = \text{MSE}\left(\frac{Y_{\text{true}} - \mu_{\text{true}}}{\sigma_{\text{true}}}, \frac{Y_{\text{pred}} - \mu_{\text{true}}}{\sigma_{\text{true}}}\right)$$

---

## 📈 Resultados Obtenidos

Tras estabilizar la convergencia, el modelo alcanza un ajuste morfológico milimétrico (sub-nanovoltiométrico) en el conjunto de prueba ciego:

* **Auto-reconstrucción (Diagonal):** MSE residual $\approx 1.19 \times 10^{-10}\text{ V}$.
* **Transformación Cruzada (Fuera de diagonal):** MSE promedio $\approx 1.45 \times 10^{-10}\text{ V}$.

El acoplamiento temporal estricto entre las señales reales y las predicciones de la red confirma empíricamente la existencia de un *Hub* topológico latente capaz de unificar de punta a punta la adquisición heterogénea de EEG clínico.

---

## 📜 Licencia

Este proyecto está bajo la licencia MIT. Consulta el archivo `LICENSE` para obtener más detalles.