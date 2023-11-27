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

# Las características predictoras, excluyendo 'target'
X_train, X_test= train_test_split(df, test_size=0.2, random_state=42)



