## Requisitos

* Instalación de Anaconda3. Como mínimo son necesarias las librerías:
    + numpy
    + pandas
    + scipy
    + matplotlib
    + seaborn
    + mlxtend
    + scikit-learn

* Conocimientos de Python 3.6+

* Conocimientos básicos de numpy, pandas y matplotlib

## Temario

1. Introducción al Machine Learning [[presentación]](./src/slides_supervised/supervised.html)

   - Introducción al aprendizaje automático
   - Definición y flujo de un proceso de Machine Learning
   - Tipos de aprendizaje automático
   - Aprendizaje supervisado vs no supervisado
   - Problema del sobreentrenamiento

2. Análisis de datos con pandas
   - Introducción y despliegue de entorno de pruebas
     + numpy
     + pandas
     + anaconda
     + jupyter
   - Operaciones básicas de manipulación de datos (numpy, pandas)

3. Machine Learning práctico con scikit-Learn
   - Introducción a scikit-learn
   - Datasets sklearn
   - Preproceso
   - Métricas para evaluar modelos y selección hiper-parámetros
      + Hands-On Python: Cross-Validation
      + Hands-On Python: GridSearchCV
   - Introducción a modelos supervisados para problemas de regresión:
      + Ajuste por mínimos cuadrados: Ridge, Lasso, Elastic Net.
      + Hands-On Python : Regresión
   - Introducción a modelos supervisados para problemas de clasificación:
      + Regresión logística
         + Hands-On Python : Clasificación
      + Introducción a métodos Bayesianos
         + Hands-On Python: Naive Bayes
      + Introducción a k-vecinos
         + Hands-On Python : KNeighborsClassifier
   - Introducción a máquinas de soporte vectorial (SVM)
      + Tipos: Clasificación y Regresión
      + Tipos de funciones Kernel
      + Consejos prácticos
      + Hands-On Python : SVC
      + Hands-On Python : SVR
   - Introducción a los árboles de decisión
      + Tipos: Clasificación, Regresión
      + Consejos prácticos
      + Hands-On Python : DecissionTreeClassifier
      + Hands-On Python : DecissionTreeRegressor
   - Introducción a Random Forest
      + Hands-On Python: RandomForestClassifier
      + Hands-On Python: RandomForestRegressor
   - Gradient Boosting algorithms
      + Hands-On Python: GradientBoostingClassifier
      + Hands-On Python: GradientBoostingRegressor
   - Introducción a las redes neuronales
      + Hands-On Python: MLP
   - Introducción a métodos de selección de variables
      + Hands-On Python: Feature-Selection
      + Hands-On Python: Dimensionality reduction
   - Introducción a modelos no supervisados
      + Clustering: K-Means
   - Optimización de hyper-parámetros avanzada
      + Hands-On Python: RandomSearchCV
      + Hands-On Python: BayesSearchCV (`scikit-optimize`)
   -  Introducción a las redes neuronales profundas     
      + Hands-On Python: Redes Neuronales Convolucionales
      + Hands-On Python: Redes Neuronales Recurrentes

4. Despliegue de modelos
   - Combinación de modelos
   - Persistencia de modelos
   - Gestión de ciclo vida de modelos:
      + Despliegue de modelos con FastAPI y Docker [[repositorio]](https://github.com/albertotb/sklearn_fastapi_docker)
      + Registro y análisis de parámetros y métricas
   - Caso práctico guiado: Kaggle


## Referencias

  * [Curso numpy y pandas básico](https://github.com/guiwitz/NumpyPandas_course)
  * [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)
  * [Python for Data Analysis (github)](https://github.com/wesm/pydata-book)
  * [scipy-lectures](https://www.scipy-lectures.org/)
  * [scipy.stats](https://docs.scipy.org/doc/scipy/reference/tutorial/stats.html)
  * [statmodels](https://www.statsmodels.org/stable/index.html)
  * [machine learning tutorials](https://github.com/ethen8181/machine-learning)
  * [Machine Learning (Loyola University Chicago)](https://github.com/dmitriydligach/PyMLSlides)
  * [Machine Learning (University Wisconsin-Madison)](https://github.com/rasbt/stat479-machine-learning-fs19)
  * [Applied Machine Learning (Columbia University)](https://github.com/amueller/COMS4995-s20)
  * [Pandas Cookbook](https://github.com/jvns/pandas-cookbook)
  * [Applied Machine Learning in Python](https://amueller.github.io/aml/)
  * [Introduction to Machine Learning in Python (workshop)](https://github.com/amueller/ml-workshop-1-of-4)
  * [Choosing the right estimator](https://scikit-learn.org/stable/_static/ml_map.png)
  * [Esquema aprendizaje supervisado](esquema.md)
