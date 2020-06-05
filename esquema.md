## Aprendizaje supervisado 
 
 1. Obtener datos
 2. Transformar a formato tabular, con una fila por observación y una columna
    por variable
 3. Dividir datos en *entrenamiento* y *test* (opcionalmente *validación*)
 4. Crear nuevas variables (*feature engineering*)
 5. Transformar todas las variables a numéricas (*one-hot encoding*)
 6. (Opcional) Reducir número de variables:
    * Selección de variables (*feature selection*), métodos filtro, wrapper, ...
    * Reducción de dimensionalidad (*dimensionality reduction*), PCA
 7. Modelizar:
    1. Definir tipo de problema: clasificación, regresión, clustering, ...
    2. Elegir modelo:
       * Linear/Logistic regression
       * Ridge Regression, Lasso, Elastic Net
       * SVM
       * Neural Networks
       * Random Forest
       * Gradient Boosting, XGBoost, LightBoost, CatBoost
    3. Seleccionar hyper-parametros
       * Grid Search
       * Random Search
       * Bayesian Optimization
 8. Análisis resultados
    * Regresión: MAE, MSE
    * Clasificación: confussion matrix, accuracy, sensitivity, specificity, ...
 9. Volver a **4**. Importante a la hora de comparar ya sea para elegir modelos,
    hyper-parametros, variables a usar, etc. usar siempre el error sobre el
    conjunto de validación o bien validación cruzada
10. Reportar el error en *test* de la configuración final y a producción  
