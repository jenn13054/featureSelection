import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import missingno as msno
from plotnine import *
import seaborn as sns
import pyarrow.feather as feather



read_df = feather.read_feather('dataset')

print(read_df.info())

read_df.target= read_df['retention'].astype('category')
read_df['target']=read_df['retention'].astype('category')
read_df.target
print(read_df)


read_var=pd.read_csv('variables.csv')

#almacemar diccionario de datos del conjunto
read_df.variables=read_var

X=read_df

#Se elminan los outlier para mejorar la minería de datos
Q1 = X['age'].quantile(0.25) #definir los limites superior e inferior para la edad, eliminando outliers
Q3 = X['age'].quantile(0.75)
IQR = Q3 - Q1

# Definir límites
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR
df_filtrado = X[(X['age'] >= limite_inferior) & (X['age'] <= limite_superior)]
df_filtrado.age.plot(kind='box')
X['age']=df_filtrado.age

# Selección de caracteristicas para set de datos tec de monterrey, preprocesado
