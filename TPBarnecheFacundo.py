import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
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
#########################################################################################
################################       FUNCIONES       ##################################
#########################################################################################

#Recibe un dataSet y si hay algun campo vacio, lo llena con el valor de la fila anterior
def isNull(dataSet):
    print(f'Puede haber minimas variaciones en el caso que haya valores vacios, estos seran reemplazados por el resultado anterior.\nEl DataSet tiene {df.isnull().sum().sum()} valores vacios\n')
    return dataSet.fillna(method='ffill')

#Recibe un dataSet e imprime detalles estadisticos
def statistics(dataSet):
    return(f'\n\
Maxima goleada de local: {dataSet["Goles Local"].max()}\n\
Maxima goleada de visitante:  {dataSet["Goles Visita"].max()}\n\
Minimo goles de local: {dataSet["Goles Local"].min()}\n\
Minimo goles de visitante: {dataSet["Goles Visita"].min()}\n\
Promedio goles de local: {dataSet["Goles Local"].mean()}\n\
Promedio goles de visitante: {dataSet["Goles Visita"].mean()}\n\
Total de goles de local {dataSet["Goles Local"].sum()}\n\
Total de goles de visita {dataSet["Goles Visita"].sum()}\n\
Total de partidos ganados por locales: {(dataSet["Goles Local"] > dataSet["Goles Visita"]).sum()}\n\
Total de partidos ganados por visitantes: {(dataSet["Goles Local"] < dataSet["Goles Visita"]).sum()}\n\
Total de partidos empatados: {(dataSet["Goles Local"] == dataSet["Goles Visita"]).sum()}\n')

#Separa las variables independientes
def independentVar():
    x1 = df.iloc[:,0:2].values
    x2 = df.iloc[:,4:].values
    return np.concatenate((x1,x2), axis=1)

#Separa las variables dependientes
def dependentVar():
    return df.iloc[:,2:4].values

#Utilizamos LabelEnconder para convertir los datos categoricos a numericos
def transform(dataSet):
    labelencoder_x = LabelEncoder()
    
    for i in range(dataSet.shape[1]):    
        dataSet[:,i] = labelencoder_x.fit_transform(dataSet[:,i])

#########################################################################################
################################       PROGRAMA       ###################################
#########################################################################################

#Importacion de datos
df = pd.read_csv('results.csv', low_memory=False)

#Valida que no haya espacios vacios y si hay los completa con el valor de la fila anterior
df = isNull(df)

#Impresión de las primeras 5 filas del dataSet con sus respectivas columnas
#print(df.head())

#Impresión de la cantidad de filas y columnas (en ese sentido) del dataSet
#print(df.shape)

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
Por otro lado, las columnas goles local y goles visitas son las 2 variables dependientes de las otras, las cuales vamos a intentar predecir con el algoritmo de machine learning.

Luego de analizar los datos y distinguir las variables vamos a limpiar el dataSet de informacion sin relevancia para los fines de esta prediccion como por ejemplo la fecha, la ciudad y el pais ya que con saber si es local suponemos que se jugó en su pais o viceverza a menos que neutral este en true (cancha neutral)
"""

#Detalles estadísticos del conjunto de datos:
#print(statistics(df))

#Limpiamos las columnas que no tienen peso en la prediccion
df.drop(["Fecha"], axis = 1, inplace = True)
df.drop(["Ciudad"], axis = 1, inplace = True)
df.drop(["Pais"], axis = 1, inplace = True)

#Separamos las variables independientes por un lado y las dependientes por otro
x = independentVar()
y = dependentVar()

"""
Hagamos una limpieza!

Una vez separadas las variables dependientes de las independientes necesitamos saber que existen 2 tipos de variables "Categoricas" y "Numericas". Las variables categoricas basicamente son las NO numericas, el problema es que para la predicción necesitamos que estas variables sean del tipo numericas, por eso una parte muy importante es traducir las variables categoricas a numericas. 
"""

#Pasamos todos los datos categoricos a numericos
transform(x)

print(x)
    
#onehotencoder = make_column_transformer((OneHotEncoder(), [0]), remainder = "passthrough")
#x = onehotencoder.fit_transform(x)

"""
A entrenar se ha dicho!

Llega el momento mas esperado, es el momento de usar los datos para entrenar. Generalmente entre el 70% y 80% se usa para la fase de entrenamiento mientras que la parte restante se usa para evaluar los resultados.
Para lograr esta fase del proyecto es necesario dividir los datos en 4 partes:
    * Variables independientes para entrenar (entre 70% y 80%)
    * Variables independientes para testear (resto)
    * Variables dependientes o llamadas conjunto de prediccion para entrenar (igual a la variable independiente)
    * Variables dependientes o llamadas conjunto de prediccion para testear (igual a la variable independiente)
"""

#Dividimos el dataSet en bloques, que usaremos para entrenamiento y validacion
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


regressor = LinearRegression() 
regressor.fit(x_train, y_train) 
y_pred = regressor.predict(x_test)

df_aux = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})

df1 = df_aux.head(50)
df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='darkgreen')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()