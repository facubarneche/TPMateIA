import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#MIRAR PREECONDICIONES QUE FALTAN ALGUNAS PERO NO ESTOY SEGURO
"""
REGRESION .............
Hoy, sabado 5 de noviembre de 2022, estamos a nada mas y nada menos que a 15 dias para el comienzo de la copa del mundo en Qatar. En consecuencia al peso de este maravilloso evento, se pensó sobre la posibilidad de hacer predicciones de los partidos del mundial.

Se cuenta con un dataSet de mas de 40000 datos, que contiene información sobre partidos entre selecciones de futbol desde 1872 hasta la actualidad. La información incluye la fecha, la selección local y visitante, los goles del local y de la visita, el torneo (amistoso, copa america, copa del mundo, etc), la cuidad y el respectivo pais en donde se disputo el partido, y neutral (un valor booleano por si el partido entre las selecciones es en un pais neutral).

Vale destacar que los datos fueron obtenidos de https://www.kaggle.com
"""

#Importacion de datos
df = pd.read_csv('results.csv', low_memory=False)
df = df.fillna(method='ffill')#PROBANDO EL LLENADO DE VARIABLES VACIAS
#Impresión de las primeras 5 filas del dataSet con sus respectivas columnas
print(df.head())

#Impresión de la cantidad de filas y columnas (en ese sentido) del dataSet
print(df.shape)

#Detalles estadísticos del conjunto de datos:
print(df.describe())

"""
Analicemos los datos!

En consola se puede observar 9 columnas (excluyendo el id de la columna): 
    * Fecha: Fecha donde el encuentro tuvo lugar
    * Seleccion Local: Seleccion que juega en su territorio
    * Seleccion Visitante: Seleccion que juega en el territorio del oponente
    * Goles Local: Goles de la seleccion que juega en su pais
    * Goles Visita: Goles de la seleccion que juega en el pais del oponente
    * Torneo: Nombre del torneo al que pertenece ese partido
    * Ciudad: Nombre de la ciudad donde se juega el encuentro
    * Pais: Nombre del pais donde se juega el encuentro
    * Neutral: Valor booleano que indica si el partido se juega en territorio neutral, en esa ocacion la seleccion local y visitante son asignados a esa posicion de forma arbitraria

Distingamos las variables...

Las columnas fecha, seleccion local, seleccion visitante, torneo, ciudad, pais y neutral, tienen informacion propia que son totalmente independientes.
Por otro lado, las columnas goles local y goles visitas son las 2 variables dependientes de las otras, las cuales vamos a intentar predecir con el algoritmo de machine learning
"""

#def isNull(self):
 #   if(df.isnull().any()):
  


