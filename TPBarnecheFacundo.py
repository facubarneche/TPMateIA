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


Hoy, estamos a nada mas y nada menos que a https://www.fifa.com/fifaplus/es/tournaments/mens/worldcup/qatar2022/countdown-to-qatar-2022 (un pequeño scroll hacia abajo xD) para el comienzo de la copa del mundo en Qatar. En consecuencia al peso de este maravilloso evento, se pensó sobre la posibilidad de hacer predicciones de los partidos de la seleccion nacional en el mundial.

Se cuenta con un dataSet de mas de 40000 registros, que contiene información sobre partidos entre selecciones de futbol desde 1872 hasta la actualidad. Se hizó un proceso de seleccion y limpieza para dejar un dataFrame unicamente de la seleccion Argentina y se le agrego posibles partidos de la seleccion argentina. La información incluye la fecha, el rival, los goles a favor y en contra, el torneo (amistoso, copa america, copa del mundo, etc), la cuidad y el respectivo pais en donde se disputo el partido, neutral (un valor booleano por si el partido entre las selecciones es en un pais neutral) y por otro lado se agrego con una formula en el csv una columna con el ganador del partido ya que al dataSet original se lo va a utilizar para hacer un proceso de limpieza en donde se va a precisar esa columna y asi crear un dataFrame mas efectivo para la informacion que se desea mostrar.

Para finalizar se podria decir que con ayuda del dataFrame inicial creamos un nuevo dataFrame que contiene la Fecha, Torneo que disputa, el rival, si juega en cancha neutral,la condicion (local o visitante), los goles a Favor, los Goles en Contra, el Ganador  y el Resultado (-1 si pierde, 0 si empata, 1 si gana). De estos datos no se va a desperdiciar ninguna columna ya que ese trabajo de limpieza lo hicimos cuando se creo este nuevo, del dataFrame inicial se desecho la ciudad y pais donde se jugo el partido y todos los partidos de selecciones diferentes a argentina.

En base a lo anteriormente comentado es que se usará la regresion lineal multiple y se esperará que prediga con la mayor precision posible el resultado de la seleccion Argentina.

Vale destacar que los datos fueron obtenidos de https://www.kaggle.com
"""

#########################################################################################
################################       FUNCIONES       ##################################
#########################################################################################

#Recibe un dataSet y si hay algun campo vacio, lo llena con el valor de la fila anterior
def isNull(dataSet):
    print(f'Puede haber minimas variaciones en el caso que haya valores vacios, estos seran reemplazados por el resultado anterior.\nEl DataSet tiene {dataSet.isnull().sum().sum()} valores vacios\n')
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
#Creo una columna resultado para mejorar la interpretacion de los datos, si los goles a favor es mayor a goles en contra "GANO", siguiendo con la logica el empate y la derrota
#Se podia directamente poner -1, 0, y 1 pero personalmente creo que se hace un poco mas entendible para una persona externa al programa
newDF['Resultado'] = np.where(newDF['Goles a Favor'] > newDF['Goles en Contra'], "GANO", np.where(newDF['Goles a Favor'] < newDF['Goles en Contra'], "PERDIO", "EMPATO"))

#Creo un diccionario para luego mapearle las variables numericas que yo quiero
ganador = {'PERDIO':-1, 'EMPATO':0, 'GANO':1}
newDF['Resultado'] = newDF['Resultado'].map(ganador)

#Ordeno el dataFrame por index ya que al concatenar los rivales de argentina de dos df distintos quedaron desorneados los index
newDF = newDF.sort_index()

"""
Este grafico muestra el promedio de los resultados de los partidos jugados de Argentina, en donde la derrota es -1, el empate 0 y la victoria 1.
Se puede ver que de visitante tiene un promedio de 0.08 aproximadamente lo que da a entender que lo que mas predomina en los partidos de la seleccion como visitante es el empate, por otro lado la cara de la seleccion como local, aproximadamente 0.57, un promedio muy cerca del 1 que seria el 100% de victorias como visitante, algo casi imposible.
"""

#Agrupo la columna condicion para relacionarlo con los resultados en un grafico de barra
newDF.groupby('Condicion')['Resultado'].mean().plot(kind='barh', figsize=(10,8), color='skyblue', label='Promedios por condicion')
plt.xlabel('Resultado', weight='bold')
plt.ylabel('Condicion')
plt.title('Promedios Argentina Local/Visitante', weight='bold', size=10)
plt.legend(title='Derrota = -1, Empate = 0, Victoria = 1')
plt.plot(data=None)
plt.show()

"""
Este grafico muestra el promedio de goles a favor y en contra de argentina como local y visitante, se aprecia como argentina de local tiene un promedio realmente bueno con mas de 2 goles por partido (sinceramente no me lo imaginaba y veo casi todos los partidos de Argentina) y menos de 1 gol por partido en contra (personalmente un promedio aceptable).
Como visitante claramente el juego de la seleccion esta mas opacado que el de local, con casi 1 gol y medio por partido a favor y 1 gol y cuarto aproximadamente de goles en contra por partido, sigue siendo excelente ya que de visitante es normal bajar la efectividad del juego.
"""

#Agrupamos en 2 series distintas la columna condicion para relacionarlo con goles a favor y en contra
prom_gaf = pd.Series(newDF.groupby('Condicion')['Goles a Favor'].mean())
prom_gec = pd.Series(newDF.groupby('Condicion')['Goles en Contra'].mean())

#Obtenemos la posicion de cada etiqueta en el eje de X
cond = ['Local', 'Visitante']
x = np.arange(len(cond))
fig, ax = plt.subplots()
width=0.25

#Generamos las barras para el conjunto de promedios de goles a favor
ax.bar(x - width/2, prom_gaf, width, label='Promedio de goles a favor',color='orange')

#Generamos las barras para el conjunto de promedios de goles en contra
ax.bar(x + width/2, prom_gec, width, label='Promedio de goles en contra',color='gray')

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

#Se podria utilizar OneHotEncoder para codificar características categóricas como una matriz y make_column_transformer permite aplicar transformaciones de datos de forma selectiva a diferentes columnas del conjunto de datos. Es decir que calcula y sobreescribe.
#onehotencoder = make_column_transformer((OneHotEncoder(), []), remainder = "passthrough")
#x = onehotencoder.fit_transform(x)
#De todos modos se inclino por no usar esta funcion ya que las columnas en las cuales serviria aplicar onehotencoder (por ej paises, o fechas) tienen un peso significativo que prefiero que sigan teniendo ya que no es lo mismo (en modo de ejemplo que el 76 es alemania a jugar contra el 15 que es islas fiji)

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

#Escalamos los datos para que ninguna columna tenga mas peso que otra en donde su rango sea muy distinto (por ej, las fechas con los goles a favor y encontra)
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)

#Utilizamos la regresion linea conocida tambien como la tecnica de los cuadrados minimos, basado en la experiencia de aprendizaje
regressor = LinearRegression() 
regressor.fit(x_train, y_train) 

#Con los datos de test se predice el resultado
y_pred = regressor.predict(x_test)

#Estas lineas comentadas son para ver con mas precision por que resultado se inclina el algoritmo ya que si la prediccion es 0.87..... se entiende que la prediccion es "GANO" = 1
#y_pred = np.round(regressor.predict(x_test))
#y_pred = y_pred.astype(int)


#Con un dataFrame auxiliar se compara los valores actuales contra las predicciones y se muestra por consola
df_aux = pd.DataFrame({'Actual': y_test.flatten(), 'Predicción': y_pred.flatten()})
print(df_aux.head(25))

"""
En este grafico se puede ver los valores actuales (reales) contra las predicciones, se aprecia que si bien la gran mayoria se acercan al resultado real no son tan precisas respecto a sus decimales. 
"""

#Se muestra por consola las primeras 25 comparaciones
df_aux.head(25).plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='darkgreen')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

#Mostramos las metricas
print('Error Absoluto Medio:',metrics.mean_absolute_error(y_test, y_pred)) 
print('Error Cuadratico Medio:', metrics.mean_squared_error(y_test, y_pred)) 
print('Raíz del error cuadrático medio:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('mean absolute percentage error:',metrics.mean_absolute_percentage_error(y_test, y_pred))

"""
Para concluir se aprecia como el valor "Raiz del error cuadratico medio" es de 0.5358 lo cual es una pesima prediccion, aunque no fue muy preciso el algoritmo las predicciones son bastante acertadas ya que no es un factor importante el decimal ya que la idea principal era que el valor se acerque al 1 si gana, 0 si empata y -1 si pierde.

Hay muchos factores que pueden haber contribuido a esta inexactitud, por ejemplo:

1. Cantidad de datos: La cantidad de registros de la seleccion Argentina no es una gran cantidad de datos para obtener la mejor predicción posible.

2. Los Atributos: La cantidad de atributos distintos como los paises, fechas, etc puede ser que le jueguen una mala pasada al algoritmo de precision haciendo que este mal interprete los datos.

3. No se puede descartar una mala decision con los datos, o quiza el algoritmo de regresion lineal no era el correcto para este modelo.

Para finalizar, segun los datos que teniamos en nuestro poder las predicciones que logro hacer el algoritmo me dejaron conformes si bien no se pudo mejorar para que la raiz del error cuadratico medio sea aceptable.
"""

#########################################################################################
##############################     PREDICCION MUNDIAL     ###############################
#########################################################################################


#Importacion de datos
df_mundial = pd.read_csv('fixtureMundial.csv', low_memory=False)
print('Apartado de Prediccion partidos del mundial:\n')
print(df_mundial.head(35))
x_mundial = independentVar(df_mundial)
y_mundial = dependentVar(df_mundial)

transform(x_mundial)

x_train_mundial, x_test_mundial, y_train_mundial, y_test_mundial = train_test_split(x_mundial, y_mundial, test_size=0.2)

sc_X_mundial = StandardScaler()
x_train_mundial = sc_X_mundial.fit_transform(x_train_mundial)
x_test_mundial = sc_X.transform(x_test_mundial)

#Utilizamos la regresion linea conocida tambien como la tecnica de los cuadrados minimos, basado en la experiencia de aprendizaje
regressor = LinearRegression() 
regressor.fit(x_train_mundial, y_train_mundial) 

#Con los datos de test se predice el resultado
y_pred_mundial = regressor.predict(x_test_mundial)

df_aux_mundial = pd.DataFrame({'Actual': y_test_mundial.flatten(), 'Predicción': y_pred_mundial.flatten()})

print(df_aux_mundial.head(35))

"""
La idea principal de este apartado es hacer las predicciones de los 7 (en el caso de ganar todos) partidos del mundial, lamentablemente no pude hacer que solo tome los ultimos 7 pero afortunadamente la prediccion dice que argentina gana todos los partidos
"""