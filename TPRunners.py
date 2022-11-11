import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import seaborn as seabornInstance 
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


Se cuenta con un dataSet de menos de 100 registros, que contiene información sobre corredores.
Se hizo un proceso de limpieza en donde se desecha la primera columna del cvs de los ids que no tenia importancia alguna. 
La información incluye el pulso antes y despues de correr, si la persona hace deporte frecuentemente, si fuma, el genero, la altura y que la intensidad de la actividad.

En base a lo anteriormente comentado es que se usará la regresion lineal multiple y se esperará que prediga con la mayor precision posible el peso del corredor segun sus atributos.
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
{dataSet.describe()}')


#Separa las variables independientes
def independentVar(dataSet):
    dataSet = dataSet[['pulso.antes','pulso.despues','hace.deporte','fuma','genero','altura','tipo.actividad','peso']]
    return dataSet.iloc[:,0:7].values

#Separa las variables dependientes
def dependentVar(dataSet):
    return dataSet.iloc[:,6:7].values

#Utilizamos LabelEnconder para convertir los datos categoricos a numericos
def transform(dataSet):
    labelencoder_x = LabelEncoder()
    
    for i in range(dataSet.shape[1]):    
        dataSet[:,i] = labelencoder_x.fit_transform(dataSet[:,i])
        
#########################################################################################
################################       PROGRAMA       ###################################
#########################################################################################

#Importacion de datos
df = pd.read_csv('runners.csv', low_memory=False)

#Valida que no haya espacios vacios y si hay los completa con el valor de la fila anterior
df = isNull(df)

#Desecha la fila de id 
df.drop(['id'], axis=1 , inplace=True)

"""
Matriz de correlación

Muestra el grado de correlaciones, de cada variable en el conjunto de datos, con cada otra variable en el conjunto de datos. Es una representación de todos estos coeficientes de correlación de cada variable individual en los datos con cada otra variable en los datos.

El grado de correlación entre dos variables cualesquiera se representa de dos maneras, el color del cuadro o caja y el número dentro. Cuanto más fuerte sea el color, mayor será la magnitud de la correlación.

Cuanto más cerca esté el número de 1, mayor será la correlación. Si el número es positivo, establece una correlación positiva. Si es negativo establece una correlación negativa. 

1 y -1 establecen correlaciones perfectas entre las variables.

En base a la matriz de correlacion  y el grafico se puede ver que hay una ligera correlatividad entre los pulsos, y una correlatividad ya mas pronunciada respecto a la altura con el peso
"""
corr = df.corr()
plt.subplots(figsize=(10,8))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, fmt='.0%',
            cmap=sns.diverging_palette(240, 10, as_cmap=True)) 

"""
La funcion de este grafico es contar cuantos pesos de cada uno hay, se ve el peso promedio en esta ocasion es de 160. 
"""

plt.tight_layout()
seabornInstance.displot(df['peso'])

"""
Analicemos los datos!

En consola se puede observar 8 columnas: 
    * Pulso Antes: Pulso antes de correr
    * Pulso Despues: Pulso despues de correr
    * Si Hace Deporte: Si es un corredor que hace deporte o no
    * Si Fuma: Si la persona tiene el habito de fumar
    * Genero: Genero del corredor
    * Altura: Altura de la persona
    * Peso: Peso de la persona
    * Tipo de Actividad: La intensidad de la actividad

Distingamos las variables...

Las columnas de los pulsos, deporte, si fuma, genero, altura, y tipo de actividad, tienen informacion propia que son totalmente independientes.
Por otro lado, la columna Peso es una variable dependiente, la cual vamos a intentar predecir con el algoritmo de machine learning.

Hagamos una limpieza!

Una vez separadas las variables dependientes de las independientes necesitamos saber que existen 2 tipos de variables "Categoricas" y "Numericas". Las variables categoricas basicamente son las NO numericas, el problema es que para la predicción necesitamos que estas variables sean del tipo numericas, por eso una parte muy importante del algoritmo de ML es traducir las variables categoricas a numericas. 
"""

#Detalles estadísticos del conjunto de datos:
statistics(df)

#Separamos las variables independientes por un lado y las dependientes por otro
x = independentVar(df)
y = dependentVar(df)

#Pasamos todos los datos categoricos a numericos (la variable independiente no tiene datos categoricos)
transform(x)


"""
A entrenar se ha dicho!

Llega el momento mas esperado, es el momento de usar los datos para entrenar. Generalmente entre el 70% y 80% se usa para la fase de entrenamiento mientras que la parte restante se usa para evaluar los resultados.
Para lograr esta fase del proyecto es necesario dividir los datos en 4 partes:
    * Variables independientes para entrenar (entre 70% y 80%)
    * Variables independientes para testear (resto)
    * Variables dependientes o llamadas conjunto de prediccion para entrenar (igual a la variable independiente)
    * Variables dependientes o llamadas conjunto de prediccion para testear (igual a la variable independiente)

Luego de hacer la division de las variables para el entrenamiento se escalaron las variables (disminuir las variables a su maxima expresion para que la diferencia de rango entre las variables no afecte a las mismas), y nos dimos cuenta que la mejora que tuvimos fue practicamente nula.

Para terminar usamos la tecnica de los cuadrados minimos para hacer las predicciones
"""

#Dividimos el dataSet en bloques, que usaremos para entrenamiento y validacion
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)

#Utilizamos la tecnica de los cuadrados minimos, basado en la experiencia de aprendizaje
regressor = LinearRegression() 
regressor.fit(x_train, y_train) 

#Con los datos de test se predice el resultado
y_pred = regressor.predict(x_test)

#PREGUNTAR COEFICIENTES
#coeff_df = pd.DataFrame(regressor.coef_, df, columns=['Coefficient']) 
#print(coeff_df)

#Con un dataFrame auxiliar se compara los valores actuales contra las predicciones
df_aux = pd.DataFrame({'Actual': y_test.flatten(), 'Predicción': y_pred.flatten()})
print(df_aux.head(25))

"""
En este grafico se puede ver los valores actuales (reales) contra las predicciones, se aprecia que si bien la gran mayoria se acercan al resultado real no son tan precisas respecto a sus decimales. 
"""

df_aux.head(30).plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='darkgreen')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

#Mostramos las metricas
print('Error Absoluto Medio:',metrics.mean_absolute_error(y_test, y_pred)) 
print('Error Cuadratico Medio:', metrics.mean_squared_error(y_test, y_pred)) 
print('Raíz del error cuadrático medio:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('mean absolute percentage error:',metrics.mean_absolute_percentage_error(y_test, y_pred))

""" 
Para concluir se apreciar como el valor "Raiz del error cuadratico medio" es de 12.6360 lo cual es una prediccion bastante acertada, ya que es menor al 10% (8,7% aproximadamente) de la media de peso (variable dependiente).

Hay factores los cuales pueden influir para que el algoritmo sea aun mas eficiente.

1. Cantidad de datos: La cantidad de registros del cvs es muy poco para hacer buenas predicciones, eso igualmente es un punto a favor para el algoritmo ya que tiene una efectividad muy buena con muy pocos datos.
2. Los Atributos: Los atributos son bastante acertados pero no tienen tanta correlacion entre los mismos, personalmente le agregaria atributos como edad, minimo de minutos corridos, maximo de minutos corridos, entre otros para alimentar el algoritmo con mas variables.

Para finalizar, segun los datos que teniamos en nuestro poder las predicciones que logro hacer el algoritmo me dejaron conformes si bien se cree que podria ser muchisimo mejor con mas cantidad de registros (incluso sin agregar los atributos antes mencionados) mejoraria facilmente.
"""