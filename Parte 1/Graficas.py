#!/usr/bin/env python
# coding: utf-8

# In[2]:


from ucimlrepo import fetch_ucirepo 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
 


# In[4]:


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


# In[3]:


# Entender los datos datos --------------------------------------------
print(data.head())
print(data.info())
print(data.describe()) # Estadísticas descriptivas
print(data.isnull().sum()) #Datos flatantes?


# In[14]:


#Lo estoy editando 


sns.histplot(data['Age at enrollment'], bins=20, kde=True)
plt.title('Distribución de Edades')


# In[5]:


plt.scatter(data['Tuition fees up to date'], data['unrate'])
plt.xlabel('Tuition')
plt.ylabel('Unemployment Rate')
plt.title('Diagrama de Dispersión entre Tuition y Unemployment Rate')
plt.show()


# In[9]:



data['Target'].value_counts().plot(kind='bar')
plt.xlabel('Target')
plt.ylabel('Frecuencia')
plt.title('Distribución del Target')
plt.xticks(rotation=0)
plt.show()


# In[12]:


data["Mother's qualification"].value_counts().plot(kind='bar')
plt.xlabel('Calificación de la Madre')
plt.ylabel('Frecuencia')
plt.title('Distribución de Calificación de la Madre')
plt.show()


# In[15]:


data["Mother's occupation"].value_counts().plot(kind='bar')
plt.xlabel('Ocupación de la Madre')
plt.ylabel('Frecuencia')
plt.title('Distribución de Ocupación de la Madre')
plt.xticks(rotation=45)
plt.show()


# In[19]:


sns.violinplot(x='Tuition fees up to date', data=data)
plt.xlabel('Tuition fees up to date')
plt.title('Diagrama de Violín para Tuition')
plt.show()


# In[20]:


data['Course'].value_counts().plot(kind='bar')
plt.xlabel('Curso')
plt.ylabel('Frecuencia')
plt.title('Distribución de Cursos')
plt.xticks(rotation=45)
plt.show()


# In[21]:


sns.violinplot(x='Unemployment rate', data=data)
plt.xlabel('Tasa de Desempleo (Unemployment Rate)')
plt.title('Diagrama de Violín para Tasa de Desempleo')
plt.show()


# In[23]:


sns.violinplot(x='Curricular units 2nd sem (grade)', data=data)
plt.xlabel('Unidades Curriculares del Segundo Semestre')
plt.title('Diagrama de Violín para Unidades Curriculares del Segundo Semestre')
plt.show()


# In[25]:


data['Curricular units 2nd sem (grade)'].value_counts().plot(kind='bar')
plt.xlabel('Unidades Curriculares del Segundo Semestre')
plt.ylabel('Frecuencia')
plt.title('Diagrama de Barras para Unidades Curriculares del Segundo Semestre')
plt.show()


# In[5]:


sns.violinplot(x='Unemployment rate', y='Target', data=data)
plt.xlabel('Tasa de Desempleo (Unemployment Rate)')
plt.ylabel('Distribución del Target')
plt.title('Diagrama de Violín para Tasa de Desempleo y Target')
plt.show()


# In[7]:


sns.violinplot(x="Mother's occupation",y="Mother's qualification", data=data)
plt.xlabel('Ocupación de la Madre')
plt.ylabel('Calificación de la Madre')
plt.title('Diagrama de Violín para Unidades Curriculares del Segundo Semestre')
plt.show()


# In[10]:


sns.violinplot(x='Curricular units 2nd sem (grade)', y='Age at enrollment', data=data)
plt.xlabel('Unidades Curriculares del Segundo Semestre')
plt.ylabel('Age at enrollment')
plt.title('Gráfico de Violín Agrupado para Unidades Curriculares del Segundo Semestre y Grados')
plt.show()


# In[ ]:




