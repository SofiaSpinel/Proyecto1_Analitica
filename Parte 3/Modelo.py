# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 08:20:58 2023

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

df = pd.read_excel("C:/Users/sofia/OneDrive/Documents/GitHub/Proyecto1_Analitica/Parte 3/DatosFINALproyecto3.xlsx")
df.head()

model = BayesianNetwork([('COLE_BILINGUE', 'puntaje'),
                          ('COLE_JORNADA','puntaje'),
                          ('COLE_NATURALEZA','puntaje'),
                          ('ESTU_GENERO','puntaje'),
                          ('FAMI_ESTRATOVIVIENDA','puntaje'),
                          ('FAMI_TIENEINTERNET','puntaje'),
                          ('FAMI_TIENECOMPUTADOR','puntaje'),])

#maxima verosimilitud
emv = MaximumLikelihoodEstimator(model= model, data=df)

#ejemplo de cpd
cpd_puntaje = emv.estimate_cpd(node="puntaje")
print(cpd_puntaje)

cpd_course = emv.estimate_cpd(node='ESTU_GENERO')
print(cpd_course)

#estimar todo el modelo
model.fit(data=df, estimator = MaximumLikelihoodEstimator)

for i in model.nodes():
    print(model.get_cpds(i))

y = df['puntaje'] # La variable objetivo 'lung'    
X = df.drop('puntaje', axis=1) # Las características predictoras, excluyendo 'lung'


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Crear una instancia de LabelEncoder
label_encoder = LabelEncoder()

# Aplicar la codificación a las columnas categóricas
categorical_columns = ['COLE_JORNADA', 'COLE_NATURALEZA', 'COLE_BILINGUE', 'ESTU_GENERO', 'FAMI_ESTRATOVIVIENDA', 'FAMI_TIENEINTERNET','FAMI_TIENECOMPUTADOR']
for col in categorical_columns:
    X_train[col] = label_encoder.fit_transform(X_train[col])
    X_test[col] = label_encoder.transform(X_test[col])


# Crea y ajusta el modelo de red neuronal
# Realiza predicciones en los datos de prueba
y_pred = model.predict(X_test)

# Calcula la exactitud del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Exactitud del modelo: {accuracy}')

# Calcula y reporta Verdaderos Positivos, Falsos Positivos, Verdaderos Negativos y Falsos Negativos
confusion = confusion_matrix(y_test, y_pred)
true_positives = confusion[1, 1]
false_positives = confusion[0, 1]
true_negatives = confusion[0, 0]
false_negatives = confusion[1, 0]

print(f'Verdaderos Positivos: {true_positives}')
print(f'Falsos Positivos: {false_positives}')
print(f'Verdaderos Negativos: {true_negatives}')
print(f'Falsos Negativos: {false_negatives}')


  
#Segunda opción de modelo
df = pd.read_excel("C:/Users/sofia/OneDrive/Documents/GitHub/Proyecto1_Analitica/Parte 3/DatosFINALproyecto3.xlsx")
df.head()

model2 = BayesianNetwork([('COLE_BILINGUE', 'puntaje'),
                          ('COLE_JORNADA','puntaje'),
                          ('COLE_NATURALEZA','COLE_JORNADA'),
                          ('ESTU_GENERO','puntaje'),
                          ('FAMI_ESTRATOVIVIENDA','FAMI_TIENEINTERNET'),
                          ('FAMI_ESTRATOVIVIENDA','FAMI_TIENECOMPUTADOR'),
                          ('FAMI_TIENEINTERNET','FAMI_TIENECOMPUTADOR'),
                          ('FAMI_TIENEINTERNET','puntaje'),
                          ('FAMI_TIENECOMPUTADOR','puntaje'),])

#maxima verosimilitud
emv = MaximumLikelihoodEstimator(model= model2, data=df)

#ejemplo de cpd
cpd_puntaje = emv.estimate_cpd(node="puntaje")
print(cpd_puntaje)

cpd_course = emv.estimate_cpd(node='ESTU_GENERO')
print(cpd_course)

#estimar todo el modelo
model2.fit(data=df, estimator = MaximumLikelihoodEstimator)

for i in model2.nodes():
    print(model2.get_cpds(i))

y = df['puntaje'] # La variable objetivo 'lung'    
X = df.drop('puntaje', axis=1) # Las características predictoras, excluyendo 'lung'


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Crear una instancia de LabelEncoder
label_encoder = LabelEncoder()

# Aplicar la codificación a las columnas categóricas
categorical_columns = ['COLE_JORNADA', 'COLE_NATURALEZA', 'COLE_BILINGUE', 'ESTU_GENERO', 'FAMI_ESTRATOVIVIENDA', 'FAMI_TIENEINTERNET','FAMI_TIENECOMPUTADOR']
for col in categorical_columns:
    X_train[col] = label_encoder.fit_transform(X_train[col])
    X_test[col] = label_encoder.transform(X_test[col])


# Crea y ajusta el modelo de red neuronal
# Realiza predicciones en los datos de prueba
y_pred = model2.predict(X_test)

# Calcula la exactitud del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Exactitud del modelo: {accuracy}')

# Calcula y reporta Verdaderos Positivos, Falsos Positivos, Verdaderos Negativos y Falsos Negativos
confusion = confusion_matrix(y_test, y_pred)
true_positives = confusion[1, 1]
false_positives = confusion[0, 1]
true_negatives = confusion[0, 0]
false_negatives = confusion[1, 0]

print(f'Verdaderos Positivos: {true_positives}')
print(f'Falsos Positivos: {false_positives}')
print(f'Verdaderos Negativos: {true_negatives}')
print(f'Falsos Negativos: {false_negatives}')




