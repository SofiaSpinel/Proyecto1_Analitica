#!/usr/bin/env python
# coding: utf-8

# In[1]:


from ucimlrepo import fetch_ucirepo 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 


# In[3]:


data = pd.read_excel("D:/7. Septimo semestre/Analítica Computacional/Proyecto/data_variables.xlsx")
data.head()


# In[6]:


sns.violinplot(x='grade', y='age', data=data)
plt.xlabel('Unidades Curriculares del Segundo Semestre')
plt.ylabel('Age at enrollment')
plt.title('Gráfico de Violín Agrupado para Unidades Curriculares del Segundo Semestre y Grados')
plt.show()


# In[7]:


ax = sns.violinplot(x="target", y="tuition", data=df)

# Añade etiquetas y título
ax.set_xlabel("Objetivo del estudiante")
ax.set_ylabel("Matrícula")
ax.set_title("Distribución de Matrícula por Objetivo del Estudiante")

# Muestra el gráfico
plt.show()


# In[14]:


sns.violinplot(x='target', y='unrate', data=data)
plt.ylabel('Tasa de Desempleo (Unemployment Rate)')
plt.xlabel('Distribución del Target')
plt.title('Diagrama de Violín para Tasa de Desempleo y Target')
plt.show()


# In[10]:


sns.histplot(data['age'], bins=20, kde=True)
plt.title('Distribución de Edades')


# In[11]:


data['target'].value_counts().plot(kind='bar')
plt.xlabel('Target')
plt.ylabel('Frecuencia')
plt.title('Distribución del Target')
plt.xticks(rotation=0)
plt.show()


# In[12]:


ax2 = sns.violinplot(x="target", y="unrate", data=df, inner="stick", palette="Set2")
ax2.set_xlabel("Objetivo del estudiante")
ax2.set_ylabel("Tasa de Desempleo")
ax2.set_title("Comparación de Tasa de Desempleo por Objetivo del Estudiante")

# Ajusta el diseño de los gráficos
plt.tight_layout()

# Muestra los gráficos
plt.show()


# In[ ]:




