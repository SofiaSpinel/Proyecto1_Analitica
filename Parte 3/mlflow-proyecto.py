# Importe el conjunto de datos de diabetes y divídalo en entrenamiento y prueba usando scikit-learn
from sklearn.model_selection import train_test_split
import pandas as pd

db = pd.read_excel("C:/Users/sofia/OneDrive/Documents/GitHub/Proyecto1_Analitica/Parte 3/DatosFINALproyecto3_SinPeriodo.xlsx")
db.head()
y = db['puntaje']
X = db.drop('puntaje', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y)

#Importe MLFlow para registrar los experimentos, el regresor de bosques aleatorios y la métrica de error cuadrático medio
import mlflow
import mlflow.sklearn
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.metrics import mean_squared_error
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from sklearn.metrics import accuracy_score, confusion_matrix

# defina el servidor para llevar el registro de modelos y artefactos
# mlflow.set_tracking_uri('http://0.0.0.0:5000')
# registre el experimento
experiment = mlflow.set_experiment("exp-efecto_Pandemia-logica")

# Aquí se ejecuta MLflow sin especificar un nombre o id del experimento. MLflow los crea un experimento para este cuaderno por defecto y guarda las características del experimento y las métricas definidas. 
# Para ver el resultado de las corridas haga click en Experimentos en el menú izquierdo. 
with mlflow.start_run(experiment_id=experiment.experiment_id):
    # defina los parámetros del modelo
    #n_estimators = 200 
    #max_depth = 6
    #max_features = 3
    # Cree el modelo con los par�metros definidos y entrénelo
    #rf = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, max_features = max_features)
    #rf.fit(X_train, y_train)
    modelo = BayesianNetwork([('COLE_BILINGUE', 'puntaje'),
                              ('COLE_JORNADA','puntaje'),
                              ('COLE_NATURALEZA','puntaje'),
                              ('ESTU_GENERO','puntaje'),
                              ('FAMI_ESTRATOVIVIENDA','puntaje'),
                              ('FAMI_TIENEINTERNET','puntaje'),
                              ('FAMI_TIENECOMPUTADOR','puntaje'),])
    emv = MaximumLikelihoodEstimator(model= modelo, data=db)
    modelo.fit(data=db, estimator = MaximumLikelihoodEstimator)
    # Realice predicciones de prueba
    y_pred = modelo.predict(X_test)
  
    # Registre los par�metros
    #mlflow.log_param("num_trees", n_estimators)
    #mlflow.log_param("maxdepth", max_depth)
    #mlflow.log_param("max_feat", max_features)
  
    # Registre el modelo
    mlflow.sklearn.log_model(modelo, "modelo-1-l�gica")
  
    # Cree y registre la mtrica de interés
    #mse = mean_squared_error(y_test, predictions)
    #mlflow.log_metric("mse", mse)
    #print(mse)# Calcula la exactitud del modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Exactitud del modelo: {accuracy}')
    