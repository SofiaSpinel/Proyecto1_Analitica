import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling
from pgmpy.estimators import MaximumLikelihoodEstimator


#cargar dataframe
df = pd.read_excel("C:/Users/sarit/OneDrive/Documentos/GitHub/Proyecto1_Analitica/Parte 1/data_variables.xlsx")
df.head()

#crear modelo
model = BayesianNetwork([('tuition','target'),('age','target'),('course','grade'),('mquali','mocup'),('grade','target'),('mocup','target'),('target','unrate')])

#maxima verosimilitud
emv = MaximumLikelihoodEstimator(model= model, data=df)

#ejemplo de cpd
cpd_target = emv.estimate_cpd(node="target")
print(cpd_target)

cpd_course = emv.estimate_cpd(node='course')
print(cpd_course)

#estimar todo el modelo
model.fit(data=df, estimator = MaximumLikelihoodEstimator)

for i in model.nodes():
    print(model.get_cpds(i))
    
import numpy
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

X = df.drop('target', axis=1) # Las características predictoras, excluyendo 'lung'
y = df['target'] # La variable objetivo 'lung'

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Crear una instancia de LabelEncoder
label_encoder = LabelEncoder()

# Aplicar la codificación a las columnas categóricas
categorical_columns = ['course', 'mquali', 'mocup', 'tuition', 'unrate', 'age', 'grade']
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
  
  
  
    
df = pd.read_excel("C:/Users/valer/OneDrive/Documentos/GitHub/Proyecto1_Analitica/data_variables.xlsx")
df.head()

#crear modelo
model_2 = BayesianNetwork([('tuition','target'),('age','target'),('course','target'),('mquali','target'),('grade','target'),('mocup','target'),('target','unrate')])

#maxima verosimilitud
emv_2 = MaximumLikelihoodEstimator(model= model_2, data=df)

#ejemplo de cpd
cpd_target_2 = emv.estimate_cpd(node="target")
print(cpd_target_2)

cpd_course_2 = emv.estimate_cpd(node='course')
print(cpd_course_2)

#estimar todo el modelo
model_2.fit(data=df, estimator = MaximumLikelihoodEstimator)

for i in model_2.nodes():
    print(model_2.get_cpds(i))


#Accuracy y verdaderos positivos 

X = df.drop('target', axis=1)  # Las características predictoras, excluyendo 'target'
y = df['target']  # La variable objetivo 'target'

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Crear una instancia de LabelEncoder
label_encoder = LabelEncoder()

# Aplicar la codificación a las columnas categóricas
categorical_columns = ['course', 'mquali', 'mocup', 'tuition', 'unrate', 'age', 'grade']
for col in categorical_columns:
    X_train[col] = label_encoder.fit_transform(X_train[col])
    X_test[col] = label_encoder.transform(X_test[col])


# Crea y ajusta el modelo de red neuronal
# Realiza predicciones en los datos de prueba
y_pred = model_2.predict(X_test)

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