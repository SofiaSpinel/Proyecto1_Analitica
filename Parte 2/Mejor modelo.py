# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 07:30:29 2023

@author: sofia spinel, sara grijalba y valeria diaz
"""
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling
from pgmpy.estimators import MaximumLikelihoodEstimator
import numpy
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import K2Score

df = pd.read_excel("D:/7. Septimo semestre/Analítica Computacional/Proyecto/data_variables_proy2.xlsx")
df.head()
#crear modelo
model_2 = BayesianNetwork([('tuition','target'),('age','target'),('course','target'),('mquali','target'),('grade','target'),('mocup','target'),('target','unrate')])

#maxima verosimilitud
emv_2 = MaximumLikelihoodEstimator(model= model_2, data=df)

#ejemplo de cpd
cpd_target_2 = emv_2.estimate_cpd(node="target")
print(cpd_target_2)

cpd_course_2 = emv_2.estimate_cpd(node='course')
print(cpd_course_2)

#estimar todo el modelo
model_2.fit(data=df, estimator = MaximumLikelihoodEstimator)

 # Las características predictoras, excluyendo 'target'
X_train, X_test= train_test_split(df, test_size=0.2, random_state=42)

#Metodos aplicados en el modelo K2 --------------------------------------------------------------------
for i in model_2.nodes():
    print(model_2.get_cpds(i))
    
scoring_method = K2Score(data=X_train)
esth = HillClimbSearch(data=X_train)
estimated_modelh = esth.estimate(
    scoring_method=scoring_method, max_indegree=6, max_iter=int(1e4) 
)
print("------K2--------")
print(estimated_modelh)
print("--------------")
print(estimated_modelh.nodes())
print("--------------")
print(estimated_modelh.edges())

print(scoring_method.score(estimated_modelh))

Modelo_k2 = BayesianNetwork([('mquali', 'age'), ('mocup', 'mquali'), ('mocup', 'unrate'), ('target', 'tuition'), ('target', 'age'), ('target', 'course'), ('target', 'mocup'), ('age', 'course'), ('grade', 'target'), ('grade', 'course')])

#maxima verosimilitud
emv_2 = MaximumLikelihoodEstimator(model= Modelo_k2, data=X_train)

#ejemplo de cpd
cpd_target_2 = emv_2.estimate_cpd(node="target")
print(cpd_target_2)

cpd_course_2 = emv_2.estimate_cpd(node='course')
print(cpd_course_2)

#estimar todo el modelo
Modelo_k2.fit(data=X_train, estimator = MaximumLikelihoodEstimator)
#Metodos aplicados en el modelo K2 --------------------------------------------------------------------

from pgmpy.estimators import BicScore
scoring_method_bic = BicScore(data=X_train)
esth_bic = HillClimbSearch(data=X_train)
estimated_modelh_bic = esth_bic.estimate(
    scoring_method=scoring_method_bic, max_indegree=6, max_iter=int(1e4) #black_list=Name
)

print("------BIc-------")
print(estimated_modelh_bic)
print("--------------")
print(estimated_modelh_bic.nodes())
print("--------------")
print(estimated_modelh_bic.edges())
print("--------------")
print(scoring_method.score(estimated_modelh_bic))


#Evaluacion----------------------------------------------------------------------
y_test= X_test['target']
X_test=X_test.drop('target', axis=1)

label_encoder = LabelEncoder()

# Aplicar la codificación a las columnas categóricas
categorical_columns = ['course', 'mquali', 'mocup', 'tuition', 'unrate', 'age', 'grade']
for col in categorical_columns:
    X_train[col] = label_encoder.fit_transform(X_train[col])
    X_test[col] = label_encoder.transform(X_test[col])

y_pred_K2 = Modelo_k2.predict(X_test)

# Calcula la exactitud del modelo
accuracy = accuracy_score(y_test, y_pred_K2)
print(f'Exactitud del modelo: {accuracy}')

# Calcula y reporta Verdaderos Positivos, Falsos Positivos, Verdaderos Negativos y Falsos Negativos
confusion = confusion_matrix(y_test, y_pred_K2)
true_positives = confusion[1, 1]
false_positives = confusion[0, 1]
true_negatives = confusion[0, 0]
false_negatives = confusion[1, 0]

print(f'Verdaderos Positivos: {true_positives}')
print(f'Falsos Positivos: {false_positives}')
print(f'Verdaderos Negativos: {true_negatives}')
print(f'Falsos Negativos: {false_negatives}')

