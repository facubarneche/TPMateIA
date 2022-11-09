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
from sklearn.preprocessing import StandardScaler

"""
Nombre y Apellido: Facundo Hernán Barneche
DataSet: Resultados de Selecciones de futbol
Fecha de Entrega: 11/11/2022


Hoy, sabado 5 de noviembre de 2022, estamos a nada mas y nada menos que a 15 dias para el comienzo de la copa del mundo en Qatar. En consecuencia al peso de este maravilloso evento, se pensó sobre la posibilidad de hacer predicciones de los partidos de la seleccion nacional en el mundial.

Se cuenta con un dataSet de mas de 40000 datos, que contiene información sobre partidos entre selecciones de futbol desde 1872 hasta la actualidad. La información incluye la fecha, la selección local y visitante, los goles del local y de la visita, el torneo (amistoso, copa america, copa del mundo, etc), la cuidad y el respectivo pais en donde se disputo el partido, neutral (un valor booleano por si el partido entre las selecciones es en un pais neutral) y por otro lado se agrego con una formula una columna con el ganador ya que al dataSet original se lo va a utilizar para hacer un proceso de limpieza en donde se va a precisar esa columna y asi crear un dataFrame mas efectivo para la informacion que se desea mostrar.

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
Maxima goleada a favor: {dataSet["Goles a Favor"].max()}\n\
Maxima goleada en contra:  {dataSet["Goles en Contra"].max()}\n\
Promedio goles a favor: {dataSet["Goles a Favor"].mean()}\n\
Promedio goles en contra: {dataSet["Goles en Contra"].mean()}\n\
Total de goles a favor {dataSet["Goles a Favor"].sum()}\n\
Total de goles en contra {dataSet["Goles en Contra"].sum()}\n\
Total de partidos ganados: {(dataSet["Resultado"] == 2).sum()}\n\
Total de partidos perdido: {(dataSet["Resultado"] == 0).sum()}\n\
Total de partidos empatados: {(dataSet["Resultado"] == 1).sum()}\n\\n')

#Separa las variables independientes
def independentVar():
    print(newDF)
    return newDF.iloc[:,0:7].values

#Separa las variables dependientes
def dependentVar():
    return newDF.iloc[:,-1].values

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

df['Condicion'] = np.where(df['Seleccion Visitante'] == 'Argentina', "Visitante", "Local")
df_local = df[df['Seleccion Local'].isin(['Argentina'])]
df_visita = df[df['Seleccion Visitante'].isin(['Argentina'])]

#Creo un nuevo dataframe para aislar al rival de la argentina
newDF = pd.DataFrame()

newDF['Fecha'] = pd.concat([df_local['Fecha'], df_visita['Fecha']])
newDF['Torneo'] = pd.concat([df_local['Torneo'], df_visita['Torneo']])

#Concateno el rival cuando jugamos de visitante y al rival cuando jugamos de local
newDF['Rival'] = pd.concat([df_local['Seleccion Visitante'], df_visita['Seleccion Local']])
#Agrego la columna de si fue en cancha neutral o no
newDF['Neutral'] = pd.concat([df_local['Neutral'], df_visita['Neutral']])
newDF['Condicion'] = pd.concat([df_local['Condicion'], df_visita['Condicion']])
newDF['Goles a Favor'] = pd.concat([df_local['Goles Local'], df_visita['Goles Visita']])
newDF['Goles en Contra'] = pd.concat([df_local['Goles Visita'], df_visita['Goles Local']])
newDF['Resultado'] = np.where(newDF['Goles a Favor'] > newDF['Goles en Contra'], "GANO", np.where(newDF['Goles a Favor'] < newDF['Goles en Contra'], "PERDIO", "EMPATO"))

ganador = {'PERDIO':0, 'EMPATO':1, 'GANO':2}
newDF['Resultado'] = newDF['Resultado'].map(ganador)

#Ordeno el dataFrame por index
newDF = newDF.sort_index()



newDF.groupby('Condicion')['Resultado'].mean().plot(kind='barh', figsize=(10,8), color='skyblue', label='Promedios por condicion')

plt.xlabel('Resultado', weight='bold')
plt.ylabel('Condicion')
plt.title('Promedios Argentina Local/Visitante', weight='bold', size=10)
plt.legend(title='Derrota = 0, Empate = 1, Victoria = 2')
plt.plot(data=None)


prom_gaf = pd.Series(newDF.groupby('Condicion')['Goles a Favor'].mean())
prom_gec = pd.Series(newDF.groupby('Condicion')['Goles en Contra'].mean())

#Obtenemos la posicion de cada etiqueta en el eje de X
cond = ['Local', 'Visitante']
x = np.arange(len(cond))
fig, ax = plt.subplots()
width=0.25

#Generamos las barras para el conjunto de promedios de goles a favor
ax.bar(x - width/2, prom_gaf, width, label='Promedio de goles a favor',color='yellow')

#Generamos las barras para el conjunto de promedios de goles en contra
ax.bar(x + width/2, prom_gec, width, label='Promedio de goles en contra',color='lightgreen')

#Agregamos las etiquetas de identificación de valores en el gráfico
ax.set_ylabel('Goles')
ax.set_title('Relación Promedio de goles de Local y Visitante')
ax.set_xticks(x)
ax.set_xticklabels(cond)

#Agregamos legen() para mostrar con colores a que pertenece cada valor.
ax.legend()
#fig.tight_layout()

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
print(statistics(newDF))
print(df)



#Limpiamos las columnas que no tienen peso en la prediccion
#df.drop(["Fecha"], axis = 1, inplace = True)
#df.drop(["Ciudad"], axis = 1, inplace = True)
#df.drop(["Pais"], axis = 1, inplace = True)
#df.drop(["Torneo"], axis = 1, inplace = True)

#Separamos las variables independientes por un lado y las dependientes por otro
x = independentVar()
y = dependentVar()


"""
Hagamos una limpieza!

Una vez separadas las variables dependientes de las independientes necesitamos saber que existen 2 tipos de variables "Categoricas" y "Numericas". Las variables categoricas basicamente son las NO numericas, el problema es que para la predicción necesitamos que estas variables sean del tipo numericas, por eso una parte muy importante es traducir las variables categoricas a numericas. 
"""

#Pasamos todos los datos categoricos a numericos
transform(x)


#Utilizamos OneHotEncoder para codificar características categóricas como una matriz y make_column_transformer permite aplicar transformaciones de datos de forma selectiva a diferentes columnas del conjunto de datos. Es decir que calcula y sobreescribe.

#CONSULTAR, ME PARECE MAS APROPIEDADO DEJARLE EL PESO A LAS FECHAS 
onehotencoder = make_column_transformer((OneHotEncoder(), [5,6]), remainder = "passthrough")
x = onehotencoder.fit_transform(x)

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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)






#sc_X = StandardScaler()
#x_train = sc_X.fit_transform(x_train)
#x_test = sc_X.transform(x_test)


regressor = LinearRegression() 
regressor.fit(x_train, y_train) 
#PREGUNTAR SI ESTA BIEN REDONDEAR LA PREDICCION Y CAMBIARLA
y_pred = regressor.predict(x_test)
#y_pred = np.round(regressor.predict(x_test))
#y_pred = y_pred.astype(int)
#i = 0
#for y in y_pred:
#    if(y > 2):
#        y_pred[int(i)] = 2
#    if(y < 0):
#        y_pred[int(i)] = 0
#    i += 1

 



df_aux = pd.DataFrame({'Actual': y_test.flatten(), 'Predicción': y_pred.flatten()})
print(df_aux.head(25))


df_aux.head(30).plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='darkgreen')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
#plt.show()

print('Error Absoluto Medio:',metrics.mean_absolute_error(y_test, y_pred)) 
print('Error Cuadratico Medio:', metrics.mean_squared_error(y_test, y_pred)) 
print('Raíz del error cuadrático medio:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
