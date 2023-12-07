
import pandas as pd
import matplotlib.pyplot as plt
 

##HISTOGRAMA PUNTAJE GLOBAL---------------------------------------------------------------------------------------------------------------------------------------------------------------
df = pd.read_csv("C:/Users/valer/OneDrive/Documentos/Analitica de datos/Proyecto 3/puntaje global bolivar.csv")

# Crear un histograma
plt.hist(df, bins=5, color='blue', edgecolor='black')

# Agregar etiquetas y título
plt.xlabel('Rangos')
plt.ylabel('Frecuencia')
plt.title('Histograma puntajes globales Bolívar')

# Mostrar el histograma
plt.show()

###FALLIDA----------------------------------------------------------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt

df_deptos = pd.read_csv("C:/Users/valer/OneDrive/Documentos/Analitica de datos/Proyecto 3/punt_global dept mañana.csv")
# Datos de ejemplo
x = df_deptos['cole_depto_ubicacion']
y = df_deptos['punt_global']

# Crear la gráfica de dispersión
plt.scatter(x, y, color='blue', marker='o', label='Datos')

# Agregar etiquetas y título
plt.xlabel('Eje X')
plt.ylabel('Eje Y')
plt.title('Gráfica de Dispersión')

# Mostrar la leyenda
plt.legend()

# Mostrar la gráfica
plt.show()

#PUNTAJE GLOBAL PROMEDIO POR DEPARTAMENTO----------------------------------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt

dfavg = pd.read_csv("C:/Users/valer/OneDrive/Documentos/Analitica de datos/Proyecto 3/avg.csv")

print(dfavg)

dfavg.rename(columns={'_col0': 'avg'}, inplace=True)

dfavg = dfavg.drop(20)
dfavg = dfavg.drop(35)
print(dfavg['avg'])
# Datos de ejemplo

categorias = dfavg['cole_depto_ubicacion'] 
print(categorias)
valores = dfavg['avg']

# Crear el gráfico de barras
bars = plt.bar(categorias,valores, color='blue')

plt.xticks(categorias, fontsize=4, fontname = "Times New Roman")

# Cambiar el color de las columnas resaltadas
bars[14].set_color('orange')
# Agregar etiquetas y título

plt.xlabel('Categorías')
plt.ylabel('Valores')
plt.title('Promedio de puntaje global por departamento en Colombia')

# Mostrar el gráfico
plt.show()


############Gráfico de barras compu-global------------------------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt


# Datos de ejemplo

categorias = ["No", "Si"]

valores = [220.63301670488045,251.89337250496845]

# Crear el gráfico de barras
plt.bar(categorias,valores, color='lightblue')

plt.xticks(categorias, fontsize=12, fontname = "Times New Roman")
# Agregar etiquetas y título

plt.xlabel('Categorías')
plt.ylabel('Valores')
plt.title('Promedio de puntaje global de acuerdo si se tiene computador o no')

# Mostrar el gráfico
plt.show()

###GRÁFICO DE DISPERSIÓN---------------------------------------------------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt

dfim = pd.read_csv("C:/Users/valer/OneDrive/Documentos/Analitica de datos/Proyecto 3/ingles mate.csv")
# Datos de ejemplo
x = dfim['punt_ingles']
y = dfim['punt_matematicas']

# Crear la gráfica de dispersión
plt.scatter(x, y, color=(0.6, 0.4, 0.4), marker='o', label='Puntajes')

# Agregar etiquetas y título
plt.xlabel('Inglés')
plt.ylabel('Matemáticas')
plt.title('Puntajes de inglés vs matemáticas en Cartagena de indias')

# Mostrar la leyenda
plt.legend()

# Mostrar la gráfica
plt.show()

#Genero-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dfavg = pd.read_csv("C:/Users/valer/OneDrive/Documentos/Analitica de datos/Proyecto 3/genero puntajes.csv")

print(dfavg)

dfavg = dfavg.drop("estu_genero", axis=1)

fila = dfavg.iloc[1]
print(fila)

fila2 = dfavg.iloc[0]
print(fila2)


# Datos de ejemplo
categorias = dfavg.columns.values
datos_f = fila2  # Primer conjunto de datos
datos_m = fila    # Segundo conjunto de datos

# Configuración de barras dobles
ancho_barra = 0.35  # Ancho de cada barra
indice = np.arange(len(categorias))  # Índices para las ubicaciones de las barras

# Crear el gráfico de barras dobles
plt.bar(indice - ancho_barra/2, datos_m, width=ancho_barra, label='Masculino', color = "lightblue")
plt.bar(indice + ancho_barra/2, datos_f, width=ancho_barra, label='Femenino', color = "teal")

# Configurar el gráfico
plt.xlabel('Categorías')
plt.ylabel('Valores')
plt.title('Gráfico de Barras Dobles por Categoría')
plt.xticks(indice, categorias)
plt.legend()

# Mostrar el gráfico
plt.show()


#Naturaleza------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt


categorias = ["NO OFICIAL", "OFICIAL"]

valores = [269.03389830508473,221.16894446059902]

# Crear el gráfico de barras
barsi =plt.bar(categorias,valores, color="lightblue")

barsi[0].set_color('blue')

plt.xticks(categorias, fontsize=12, fontname = "Times New Roman")
# Agregar etiquetas y título

plt.xlabel('Categorías')
plt.ylabel('Valores')
plt.title('Puntaje promedio según la naturaleza del colegio')

# Mostrar el gráfico
plt.show()


#Jornada--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


import matplotlib.pyplot as plt

dfjor = pd.read_csv("C:/Users/valer/OneDrive/Documentos/Analitica de datos/Proyecto 3/puntaje global promedio por jornada.csv")
print(dfjor)
categorias = dfjor['cole_jornada']

valores = dfjor['_col0']

# Crear el gráfico de barras
barsi2 = plt.bar(categorias,valores, color=(0.5,0,0.5))

barsi2[0].set_color('#FFC0CB')
#(0.784, 0.627, 1.0)

plt.xticks(categorias, fontsize=12, fontname = "Times New Roman")
# Agregar etiquetas y título

plt.xlabel('Categorías')
plt.ylabel('Valores')
plt.title('Puntaje promedio según la jornada del colegio')

# Mostrar el gráfico
plt.show()



