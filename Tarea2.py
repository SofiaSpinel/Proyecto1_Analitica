# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 12:50:19 2023

@author: sofia
"""

from ucimlrepo import fetch_ucirepo 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
 
# fetch dataset 
predict_students_dropout_and_academic_success = fetch_ucirepo(id=697) 
  
# data (as pandas dataframes) 
X = predict_students_dropout_and_academic_success.data.features 
y = predict_students_dropout_and_academic_success.data.targets 
  
# metadata 
print(predict_students_dropout_and_academic_success.metadata) 
  
# variable information 
print(predict_students_dropout_and_academic_success.variables) 


# Convertir los datos a un dataframe 
data = pd.concat([X, y], axis=1)

# Entender los datos datos --------------------------------------------
print(data.head())
print(data.info())
print(data.describe()) # Estadísticas descriptivas
print(data.isnull().sum()) #Datos flatantes?

#Lo estoy editando 

plt.subplot(1, 2, 1)
sns.histplot(data['Age at enrollment'], bins=20, kde=True)
plt.title('Distribución de Edades')

plt.figure(figsize=(8, 6))
sns.boxplot(y=data['Admission grade'])
plt.title('Boxplot de la variable "grade"')
plt.ylabel('Calificación')
plt.show()




