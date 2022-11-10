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


#########################################################################################
##############################       PRECONDICIONES       ###############################
#########################################################################################
"""
Nombre y Apellido: Facundo Hernán Barneche
DataSet: Resultados de Selecciones de futbol
Fecha de Entrega: 11/11/2022


Hoy, sabado 5 de noviembre de 2022, estamos a nada mas y nada menos que a 15 dias para el comienzo de la copa del mundo en Qatar. En consecuencia al peso de este maravilloso evento, se pensó sobre la posibilidad de hacer predicciones de los partidos de la seleccion nacional en el mundial.

Se cuenta con un dataSet de mas de 40000 datos, que contiene información sobre partidos entre selecciones de futbol desde 1872 hasta la actualidad. Se hizó un proceso de seleccion y limpieza para dejar un dataFrame unicamente de la seleccion Argentina. La información incluye la fecha, el rival, los goles a favor y en contra, el torneo (amistoso, copa america, copa del mundo, etc), la cuidad y el respectivo pais en donde se disputo el partido, neutral (un valor booleano por si el partido entre las selecciones es en un pais neutral) y por otro lado se agrego con una formula en el csv una columna con el ganador del partido ya que al dataSet original se lo va a utilizar para hacer un proceso de limpieza en donde se va a precisar esa columna y asi crear un dataFrame mas efectivo para la informacion que se desea mostrar.

Para finalizar se podria decir que con ayuda del dataFrame inicial creamos un nuevo dataFrame que contiene la Fecha, Torneo que disputa, el rival, si juega en cancha neutral,la condicion (local o visitante), los goles a Favor, los Goles en Contra, el Ganador  y el Resultado (-1 si pierde, 0 si empata, 1 si gana). De estos datos no se va a desperdiciar ninguna columna ya que ese trabajo de limpieza lo hicimos cuando se creo este nuevo, del dataFrame inicial se desecho la ciudad y pais donde se jugo el partido y todos los partidos de selecciones diferentes a argentina.

En base a lo anteriormente comentado es que se usará la regresion lineal multiple y se esperará que prediga con la mayor precision posible el resultado de la seleccion Argentina.

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
    print(f'\n\
Impresión de las primeras 5 filas del dataSet con sus respectivas columnas:\n\
{dataSet.head()}\n\n\
Impresión de la cantidad de filas y columnas (en ese sentido) del dataSet\n\
{dataSet.shape}\n\n\
{dataSet.describe()}\n\n\
Maxima goleada a favor: {dataSet["Goles a Favor"].max()}\n\
Maxima goleada en contra:  {dataSet["Goles en Contra"].max()}\n\
Promedio goles a favor: {dataSet["Goles a Favor"].mean()}\n\
Promedio goles en contra: {dataSet["Goles en Contra"].mean()}\n\
Total de goles a favor {dataSet["Goles a Favor"].sum()}\n\
Total de goles en contra {dataSet["Goles en Contra"].sum()}\n\
Total de partidos ganados: {(dataSet["Resultado"] == 2).sum()}\n\
Total de partidos perdido: {(dataSet["Resultado"] == 0).sum()}\n\
Total de partidos empatados: {(dataSet["Resultado"] == 1).sum()}\n')

#Separa las variables independientes
def independentVar(dataSet):
    return dataSet.iloc[:,0:7].values

#Separa las variables dependientes
def dependentVar(dataSet):
    return dataSet.iloc[:,-1].values

#Utilizamos LabelEnconder para convertir los datos categoricos a numericos
def transform(dataSet):
    labelencoder_x = LabelEncoder()
    
    for i in range(dataSet.shape[1]):    
        dataSet[:,i] = labelencoder_x.fit_transform(dataSet[:,i])
        
#########################################################################################
################################       PROGRAMA       ###################################
#########################################################################################

#Importacion de datos
df = pd.DataFrame()
df = pd.read_csv('results.csv', low_memory=False)

#Valida que no haya espacios vacios y si hay los completa con el valor de la fila anterior
df = isNull(df)

#Le creo una columna al df inicial para luego pasarlo al nuevo df
df['Condicion'] = np.where(df['Seleccion Visitante'] == 'Argentina', "Visitante", "Local")

#Filtra los partidos como local y como visitante (si existe argentina)
df_local = df[df['Seleccion Local'].isin(['Argentina'])]
df_visita = df[df['Seleccion Visitante'].isin(['Argentina'])]

#Creo un nuevo dataframe para aislar al rival de la argentina
newDF = pd.DataFrame()

#Concateno las los datos de df_local y df_visita
newDF['Fecha'] = pd.concat([df_local['Fecha'], df_visita['Fecha']])
newDF['Torneo'] = pd.concat([df_local['Torneo'], df_visita['Torneo']])

#Concateno el rival cuando jugamos de visitante y al rival cuando jugamos de local
newDF['Rival'] = pd.concat([df_local['Seleccion Visitante'], df_visita['Seleccion Local']])
#Agrego la columna de si fue en cancha neutral o no, la condicion, los goles y el resultado
newDF['Neutral'] = pd.concat([df_local['Neutral'], df_visita['Neutral']])
newDF['Condicion'] = pd.concat([df_local['Condicion'], df_visita['Condicion']])
newDF['Goles a Favor'] = pd.concat([df_local['Goles Local'], df_visita['Goles Visita']])
newDF['Goles en Contra'] = pd.concat([df_local['Goles Visita'], df_visita['Goles Local']])
newDF['Resultado'] = np.where(newDF['Goles a Favor'] > newDF['Goles en Contra'], "GANO", np.where(newDF['Goles a Favor'] < newDF['Goles en Contra'], "PERDIO", "EMPATO"))

#Creo un diccionario para luego mapearle las variables numericas que yo quiero
ganador = {'PERDIO':-1, 'EMPATO':0, 'GANO':1}
newDF['Resultado'] = newDF['Resultado'].map(ganador)

#Ordeno el dataFrame por index
newDF = newDF.sort_index()


#Agrupo la columna condicion para relacionarlo con los resultados en un grafico de barra
newDF.groupby('Condicion')['Resultado'].mean().plot(kind='barh', figsize=(10,8), color='skyblue', label='Promedios por condicion')
plt.xlabel('Resultado', weight='bold')
plt.ylabel('Condicion')
plt.title('Promedios Argentina Local/Visitante', weight='bold', size=10)
plt.legend(title='Derrota = -1, Empate = 0, Victoria = 1')
plt.plot(data=None)
plt.show()


#Agrupamos en 2 series distintas la columna condicion para relacionarlo con goles a favor y en contra
prom_gaf = pd.Series(newDF.groupby('Condicion')['Goles a Favor'].mean())
prom_gec = pd.Series(newDF.groupby('Condicion')['Goles en Contra'].mean())

#Obtenemos la posicion de cada etiqueta en el eje de X
cond = ['Local', 'Visitante']
x = np.arange(len(cond))
fig, ax = plt.subplots()
width=0.25

#Generamos las barras para el conjunto de promedios de goles a favor
ax.bar(x - width/2, prom_gaf, width, label='Promedio de goles a favor',color='red')

#Generamos las barras para el conjunto de promedios de goles en contra
ax.bar(x + width/2, prom_gec, width, label='Promedio de goles en contra',color='pink')

#Agregamos las etiquetas de identificación de valores en el gráfico
ax.set_ylabel('Goles')
ax.set_title('Relación Promedio de goles de Local y Visitante')
ax.set_xticks(x)
ax.set_xticklabels(cond)

#Agregamos legen() para mostrar con colores a que pertenece cada valor.
ax.legend()
fig.tight_layout()
plt.show()


"""
Analicemos los datos!

En consola se puede observar 8 columnas (excluyendo el id de la columna): 
    * Fecha: Fecha donde el encuentro tuvo lugar
    * Torneo: Nombre del torneo al que pertenece ese partido
    * Rival: Nombre del rival de Argentina
    * Neutral: Valor booleano que indica si el partido se juega en territorio neutral, en esa ocacion la seleccion local y visitante son asignados a esa posicion de forma arbitraria
    * Condicion: Local o Visitante
    * Goles a favor: Goles de la seleccion a favor
    * Goles en contra: Goles de la seleccion en contra
    * Resultado: Resultado final del partido
    
Distingamos las variables...

Las columnas fecha, torneo, rival, neutral, condicion, goles a favor y en contra, tienen informacion propia que son totalmente independientes.
Por otro lado, la columna resultados es una variable dependiente, la cual vamos a intentar predecir con el algoritmo de machine learning.

Hagamos una limpieza!

Una vez separadas las variables dependientes de las independientes necesitamos saber que existen 2 tipos de variables "Categoricas" y "Numericas". Las variables categoricas basicamente son las NO numericas, el problema es que para la predicción necesitamos que estas variables sean del tipo numericas, por eso una parte muy importante del algoritmo de ML es traducir las variables categoricas a numericas. 
"""

#Detalles estadísticos del conjunto de datos:
statistics(newDF)

#Separamos las variables independientes por un lado y las dependientes por otro
x = independentVar(newDF)
y = dependentVar(newDF)

#Pasamos todos los datos categoricos a numericos (la variable independiente no tiene datos categoricos)
transform(x)

#Utilizamos OneHotEncoder para codificar características categóricas como una matriz y make_column_transformer permite aplicar transformaciones de datos de forma selectiva a diferentes columnas del conjunto de datos. Es decir que calcula y sobreescribe.

#CONSULTAR, ASI TENGO MENOS ERROR PERO NO ME PARECE CORRECTO
#onehotencoder = make_column_transformer((OneHotEncoder(), [5,6]), remainder = "passthrough")
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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)

sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)

#Utilizamos la tecnica de los cuadrados minimos, basado en la experiencia de aprendizaje
regressor = LinearRegression() 
regressor.fit(x_train, y_train) 

#PREGUNTAR SI ESTA BIEN REDONDEAR LA PREDICCION Y CAMBIARLA
#Con los datos de test se predice el resultado
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

#Con un nuevo dataFrame se compara los valores actuales con las predicciones
df_aux = pd.DataFrame({'Actual': y_test.flatten(), 'Predicción': y_pred.flatten()})
#print(df_aux.head(25))

#El grafico muestra los valores actuales contra las predicciones
df_aux.head(30).plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='darkgreen')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
#plt.show()

#Mostramos las metricas
print('Error Absoluto Medio:',metrics.mean_absolute_error(y_test, y_pred)) 
print('Error Cuadratico Medio:', metrics.mean_squared_error(y_test, y_pred)) 
print('Raíz del error cuadrático medio:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
