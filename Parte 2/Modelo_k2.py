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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

df = pd.read_excel("D:/7. Septimo semestre/Analítica Computacional/Proyecto/data_variables_proy2.xlsx")
df.head()
#crear modelo
model_2 = BayesianNetwork([('tuition','target'),('age','target'),('course','target'),('mquali','target'),('grade','target'),('mocup','target'),('target','unrate')])

#maxima verosimilitud
emv_2 = MaximumLikelihoodEstimator(model= model_2, data=df)

#ejemplo de cpd
cpd_target_2 = emv_2.estimate_cpd(node="target")
#print(cpd_target_2)

cpd_course_2 = emv_2.estimate_cpd(node='course')
#print(cpd_course_2)

#estimar todo el modelo
model_2.fit(data=df, estimator = MaximumLikelihoodEstimator)

 # Las características predictoras, excluyendo 'target'
X_train, X_test= train_test_split(df, test_size=0.2, random_state=42)

#Metodos aplicados en el modelo K2 
#Primera Prueba--------------------------------------------------------------------
#----------------------------------------------------------------------------------
for i in model_2.nodes():
    print(model_2.get_cpds(i))
    
scoring_method = K2Score(data=X_train)
esth = HillClimbSearch(data=X_train)
estimated_modelh = esth.estimate(
    scoring_method=scoring_method, max_indegree=2, max_iter=int(1e4) 
)
print("------K2---#1-----------------------------------------------------")
#print(estimated_modelh)
print("--------------")
#print(estimated_modelh.nodes())
print("--------------")
#print(estimated_modelh.edges())

print(scoring_method.score(estimated_modelh))

Modelo_k2 = BayesianNetwork([('mquali', 'age'), ('mocup', 'mquali'), ('mocup', 'unrate'), ('target', 'tuition'), ('target', 'age'), ('target', 'course'), ('target', 'mocup'), ('age', 'course'), ('grade', 'target'), ('grade', 'course')])

#maxima verosimilitud
emv_2 = MaximumLikelihoodEstimator(model= Modelo_k2, data=X_train)

#ejemplo de cpd
cpd_target_2 = emv_2.estimate_cpd(node="target")
#print(cpd_target_2)

cpd_course_2 = emv_2.estimate_cpd(node='course')
#print(cpd_course_2)

#estimar todo el modelo
Modelo_k2.fit(data=X_train, estimator = MaximumLikelihoodEstimator)

#Segunda Prueba--------------------------------------------------------------------
#----------------------------------------------------------------------------------
 # Las características predictoras, excluyendo 'target'
X_train, X_test= train_test_split(df, test_size=0.2, random_state=42)

for i in model_2.nodes():
    print(model_2.get_cpds(i))
    
scoring_method = K2Score(data=X_train)
esth = HillClimbSearch(data=X_train)
estimated_modelh_2 = esth.estimate(
    scoring_method=scoring_method, max_indegree=4, max_iter=int(1e4),
    black_list=[('mocup', 'mquali'),('target', 'age'),('target', 'mocup'),('target', 'tuition'),
                ('target', 'grade'),('age', 'mquali'),('grade', 'tuition'),('grade', 'course'),
                ('mquali', 'age'),('unrate', 'course'),('target', 'course'),('mocup', 'age'),
                ('mocup', 'unrate'),('mquali', 'unrate'),('unrate', 'mquali')]
)
print("------K2---#2-----------------------------------------------------")
#print(estimated_modelh)
print("--------------")
#print(estimated_modelh_2.nodes())
print("--------------")
#print(estimated_modelh_2.edges())

print(scoring_method.score(estimated_modelh_2))

Modelo_k2_2 = BayesianNetwork(estimated_modelh_2)
#maxima verosimilitud
emv_2_2 = MaximumLikelihoodEstimator(model= Modelo_k2_2, data=X_train)

#ejemplo de cpd
cpd_target_2_2 = emv_2_2.estimate_cpd(node="target")
#print(cpd_target_2)

cpd_course_2_2 = emv_2_2.estimate_cpd(node='course')
#print(cpd_course_2_2)

#estimar todo el modelo
Modelo_k2_2.fit(data=X_train, estimator = MaximumLikelihoodEstimator)

#Segunda Prueba--------------------------------------------------------------------
#----------------------------------------------------------------------------------
 # Las características predictoras, excluyendo 'target'
X_train, X_test= train_test_split(df, test_size=0.2, random_state=42)

for i in model_2.nodes():
    print(model_2.get_cpds(i))
    
scoring_method = K2Score(data=X_train)
esth = HillClimbSearch(data=X_train)
estimated_modelh_3 = esth.estimate(
    scoring_method=scoring_method, max_indegree=8, max_iter=int(1e4),
    black_list=[('mocup', 'mquali'),('target', 'age'),('target', 'mocup'),('target', 'tuition'),
                ('target', 'grade'),('age', 'mquali'),('grade', 'tuition'),('grade', 'course'),
                ('mquali', 'age'),('unrate', 'course'),('target', 'course'),('mocup', 'age'),
                ('mocup', 'unrate'),('mquali', 'unrate'),('unrate', 'mquali')]
)
print("------K2---#3-----------------------------------------------------")
#print(estimated_modelh)
print("--------------")
#print(estimated_modelh_3.nodes())
print("--------------")
#print(estimated_modelh_3.edges())

print(scoring_method.score(estimated_modelh_3))

Modelo_k2_3 = BayesianNetwork(estimated_modelh_3)

#maxima verosimilitud
emv_2_3 = MaximumLikelihoodEstimator(model= Modelo_k2_3, data=X_train)

#ejemplo de cpd
cpd_target_2_3 = emv_2_3.estimate_cpd(node="target")
#print(cpd_target_2_3)

cpd_course_2_3 = emv_2_3.estimate_cpd(node='course')
#print(cpd_course_2_3)

#estimar todo el modelo
Modelo_k2_3.fit(data=X_train, estimator = MaximumLikelihoodEstimator)

print("-------Comparacion-------")
print(scoring_method.score(estimated_modelh_2))
print(scoring_method.score(estimated_modelh_3))
#Evaluacion----------------------------------------------------------------------
y_test= X_test['target']
X_test=X_test.drop('target', axis=1)

label_encoder = LabelEncoder()

# Aplicar la codificación a las columnas categóricas
categorical_columns = ['course', 'mquali', 'mocup', 'tuition', 'unrate', 'age', 'grade']
for col in categorical_columns:
    X_train[col] = label_encoder.fit_transform(X_train[col])
    X_test[col] = label_encoder.transform(X_test[col])

y_pred_K2_3 = Modelo_k2_3.predict(X_test)

# Calcula la exactitud del modelo
accuracy = accuracy_score(y_test, y_pred_K2_3)
print(f'Exactitud del modelo: {accuracy}')

# Calcula y reporta Verdaderos Positivos, Falsos Positivos, Verdaderos Negativos y Falsos Negativos
confusion = confusion_matrix(y_test, y_pred_K2_3)
true_positives = confusion[1, 1]
false_positives = confusion[0, 1]
true_negatives = confusion[0, 0]
false_negatives = confusion[1, 0]
sensibilidad=true_positives/(true_positives+false_negatives)

print(f'Verdaderos Positivos: {true_positives}')
print(f'Falsos Positivos: {false_positives}')
print(f'Verdaderos Negativos: {true_negatives}')
print(f'Falsos Negativos: {false_negatives}')

print(f'Sensibilidad del modelo: {sensibilidad}')

from pgmpy.readwrite import BIFWriter
# write model to a BIF file 
filename='monty.bif'
writer = BIFWriter(Modelo_k2_3)
writer.write_bif(filename=filename)

# Obtener las probabilidades de pertenencia a la clase positiva
# Predecir las probabilidades de pertenencia a la clase positiva
y_predicted = Modelo_k2_3.predict(X_test)


# Obtener las probabilidades de pertenencia a la clase positiva (asumiendo que la clase positiva es 1)
# Convertir las etiquetas de clase a valores numéricos binarios
label_encoder = LabelEncoder()
y_test_binary = label_encoder.fit_transform(y_test)
y_predicted_binary = label_encoder.transform(y_predicted)

# Calcular la curva ROC y el AUC
fpr, tpr, _ = roc_curve(y_test_binary, y_predicted_binary)
roc_auc = auc(fpr, tpr)

#print("Etiquetas codificadas:", y_test_binary)
#print("Clases originales:", label_encoder.inverse_transform(y_test_binary))


# Graficar la curva ROC
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC - K2Score')
plt.legend(loc="lower right")
plt.show()

