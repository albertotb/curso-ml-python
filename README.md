# Introducción al Machine Learning con Python

## Requisitos

- Instalación de Anaconda3. Como mínimo son necesarias las librerías:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn

- Conocimientos de Python 3.9+

## Temario

### 1. Introducción a Python para Data Science (aprox. 1h)

  1. Entorno de trabajo
  2. Repaso conceptos básicos
  3. Estructuras de datos: listas, tuplas y diccionarios

### 2. EDA con Pandas (aprox. 7 horas)

  1. Estructuras de datos: `DataFrame` y `Series`
  2. Importar datos
  3. Filtrado de filas y columnas
  4. Estadísticas descriptivas
  5. Operaciones agrupadas
  6. Visualización

### 3. Machine Learning práctico con scikit-learn (aprox. 10 horas)

  1. Introducción al Machine Learning [[presentación]](https://albertotb.com/curso-inap/big_data.html)
  2. Introducción a scikit-learn
  3. Conjuntos de datos
  4. Preproceso
  5. Modelos lineales de regresión: Ridge, Lasso, Elastic Net.
  6. Modelos lineales de clasificación: regresión logística
  7. K-Vecinos próximos
  8. Árboles de decisión
  9. Ensembles: bagging y boosting
      - Random Forest
      - Gradient Boosting
  10. Métricas para evaluar modelos
  11. Introducción a modelos no supervisados: K-means

### 4.Introducción a transformers (aprox. 2 horas)

  1. Modelos pre-entrenados
  2. Ejemplos de uso

### 5. Despliegue de modelos (aprox. 3 horas)

  1. Combinación de modelos
  2. Persistencia de modelos
  3. Gestión de ciclo vida de modelos:
     - Despliegue de modelos con FastAPI y Docker [[repositorio]](https://github.com/albertotb/sklearn_fastapi_docker)
     - Registro y análisis de parámetros y métricas
     - Caso práctico guiado: Kaggle

## Referencias

General

- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)
- [Machine Learning tutorials](https://github.com/ethen8181/machine-learning)
- [scikit-learn MOOC](https://www.fun-mooc.fr/en/courses/machine-learning-python-scikit-learn/)
- [Machine Learning (Loyola University Chicago)](https://github.com/dmitriydligach/PyMLSlides)
- [Machine Learning (University Wisconsin-Madison)](https://github.com/rasbt/stat479-machine-learning-fs19)
- [Applied Machine Learning (Columbia University)](https://github.com/amueller/COMS4995-s20)
- [Applied Machine Learning in Python](https://amueller.github.io/aml/)
- [Introduction to Machine Learning in Python (workshop)](https://github.com/amueller/ml-workshop-1-of-4)
- [Pandas Cookbook](https://github.com/jvns/pandas-cookbook)
- [Curso numpy y pandas básico](https://github.com/guiwitz/NumpyPandas_course)
- [Python for Data Analysis (github)](https://github.com/wesm/pydata-book)
- [machine learning tutorials](https://github.com/ethen8181/machine-learning)

Transformers

- Hugging Face. [NLP Course](https://huggingface.co/learn/nlp-course/chapter1/1)
- Hugging Face. [The transformer model family](https://huggingface.co/docs/transformers/en/model_summary)
- Hugging Face. [Transformers](https://huggingface.co/docs/transformers/en/index)
- Hugging Face. [Models](https://huggingface.co/models?sort=trending)
- Hugging Face. [Pretrained models](https://huggingface.co/transformers/v3.3.1/pretrained_models.html)
- [The most popular HuggingFace models](https://medium.com/@nzungize.lambert/the-most-popular-huggingface-models-d67eaaea392c)
- [Hugging Face Pre-trained Models: Find the Best One for Your Task](https://neptune.ai/blog/hugging-face-pre-trained-models-find-the-best)
