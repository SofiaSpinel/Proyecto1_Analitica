# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 16:08:46 2023

@author: sofia
"""

# Importe el conjunto de datos de diabetes y divÃ­dalo en entrenamiento y prueba usando scikit-learn
from sklearn.model_selection import train_test_split
import pandas as pd

db = pd.read_excel("C:/Users/sofia/OneDrive/Documents/GitHub/Proyecto1_Analitica/Parte 3/DatosFINALproyecto3_SinPeriodo.xlsx")
db.head()
y = db['puntaje']
X = db.drop('puntaje', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y)

#Importe MLFlow para registrar los experimentos, el regresor de bosques aleatorios y la mÃ©trica de error cuadrÃ¡tico medio
import mlflow
import mlflow.sklearn
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.metrics import mean_squared_error
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from sklearn.metrics import accuracy_score, confusion_matrix
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import K2Score
from pgmpy.estimators import BicScore

# defina el servidor para llevar el registro de modelos y artefactos
# mlflow.set_tracking_uri('http://0.0.0.0:5000')
# registre el experimento
experiment = mlflow.set_experiment("exp-efecto_Pandemia-K2")

# AquÃ­ se ejecuta MLflow sin especificar un nombre o id del experimento. MLflow los crea un experimento para este cuaderno por defecto y guarda las caracterÃ­sticas del experimento y las mÃ©tricas definidas. 
# Para ver el resultado de las corridas haga click en Experimentos en el menÃº izquierdo. 
with mlflow.start_run(experiment_id=experiment.experiment_id):
    # defina los parÃ¡metros del modelo
    #n_estimators = 200 
    #max_depth = 6
    #max_features = 3
    Black_list=""
    # Cree el modelo con los parámetros definidos y entrÃ©nelo
    #rf = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, max_features = max_features)
    #rf.fit(X_train, y_train)
    scoring_method = BicScore(data=X_train)
    esth_bic = HillClimbSearch(data=X_train)
    estimated_modelh_bic = esth_bic.estimate(
        scoring_method=scoring_method, max_indegree=4, max_iter=int(1e4))
    
    Modelo_BicScore = BayesianNetwork(estimated_modelh_bic)
    emv_2= MaximumLikelihoodEstimator(model= Modelo_BicScore, data=X_train)
    
    Modelo_BicScore.fit(data=X_train, estimator = MaximumLikelihoodEstimator)
    # Realice predicciones de prueba
    y_pred = Modelo_BicScore.predict(X_test)
  
    # Registre los parámetros
    mlflow.log_param("lista.negra", Black_list)
    #mlflow.log_param("num_trees", n_estimators)
    #mlflow.log_param("maxdepth", max_depth)
    #mlflow.log_param("max_feat", max_features)
  
    # Registre el modelo
    mlflow.sklearn.log_model(Modelo_BicScore, "Modelo-k2")
  
    # Cree y registre la mtrica de interÃ©s
    #mse = mean_squared_error(y_test, predictions)
    #mlflow.log_metric("mse", mse)
    #print(mse)# Calcula la exactitud del modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Exactitud del modelo: {accuracy}')
    