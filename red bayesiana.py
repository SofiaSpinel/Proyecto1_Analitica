import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling
from pgmpy.estimators import MaximumLikelihoodEstimator
import itertools
from itertools import permutations

#cargar dataframe
df = pd.read_excel("C:/Users/valer/OneDrive/Documentos/Analitica de datos/Proyecto 1/data_variables.xlsx")

#crear modelo
model = BayesianNetwork([('mquali','tuition'),('mocup','tuition'),('age','target'),('course','grade'),('mquali','mocup'),('grade','target'),('mocup','target'),('target','unrate')])

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



    

