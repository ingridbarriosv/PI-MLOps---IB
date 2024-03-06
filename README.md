
# PI MLOps_Steam 
## Proyecto Individual 1

El siguiente proyecto consiste en desarrollar un modelo de machine learning que simule las funciones de un ingeniero de MLOps (ingeniería de datos y ciencia de datos), el cual está fundamentado en solucionar un problema de negocio con un sistema de recomendación de videojuegos.

El proyecto se enfoca en desarrollar un MVP (Minimun Viable Product) utilizando los datasets proporcionados para la construcción de una API que facilite la recomendación de videojuegos por la similitud entre ellos. 

A través de la API desarrollada se harán las consultas a la base de datos que ya ha sido filtrada y transformada, lo cual garantiza la disponibilidad de datos de calidad para el buen funcionamiento del modelo. 

## Data Utilizada

Para desarrollar este proyecto se utilizaron 3 archivos principales:
- **steam_games.json**: este dataset contiene los datos relacionados con los juegos, sus nombres, sus géneros,  desarrolladores, sus precios, etiquetas, entre otros. 
- **user_reviews.json**: este dataset contiene los datos de los reviews que los usuarios hicieron de los juegos que consumen, estos reviews arrojan varia información de si recomiendan el juego o no, entre otras cosas.
- **user_items.json**: este dataset contiene información sobre todos los juegos que consume cada usuario y el tiempo de juego. 

## Descripción General 
A continuación se verá una descripción general de las tareas desarrolladas a lo largo del proyecto, de forma detallada pero precisa.

### Proceso de ETL
En esta primera fase, como es lo pertinente, se hizo un proceso de ETL (Extracción, Transformación y Carga) de los datasets anteriormente mencionados. 
El objetivo principal de este proceso es adaptar los datos para poder trabajar con ellos en Python encaminados a las necesidades del proyecto.  

Primero que todo, se generan DataFrames de pandas a partir de los archivos json proporcionados para poder visualizar mejor el contenido de los datasets; seguidamente se hace la eliminación de columnas que no necesitamos, filas vacías, datos nulos, entre otras transformaciones básicas. 

En estos datasets se encontraron columnas con datos que estaban anidados, lo que significa que estaban contenidos en diccionarios, listas, o listas de diccionarios.  
Para esto se utilizó la función **explode()**, que toma un iterable como entrada (lista, diccionario, lista de diccionarios, etc), y devuelve un nuevo iterable que contiene cada elemento del iterable original como un elemento individual.  
También se completaron valores nulos, se dividió el dataframe principal en varios dataframes secundarios con el objetivo de seleccionar la data importante y optimizar el rendimiento en la API.

Posteriormente con los dataframes modificados, se convirtieron a formato parquet, para una mejor asimilación de la data en la API.

Los detalles del proceso de ETL están sumados en las carpetas con Jupyter Notebooks como: **ETL games, ETL items y ETL reviews**.

### Feature Engineering

En este apartado del proyecto se debe cumplir con la creación de la columna ***'sentiment_analysis'*** aplicando un análisis de sentimientos con NLP con la siguiente escala: 

- **'0'** si el comentario es **malo** 
- **'1'** si el comentario es **neutral**
- **'2'** si el comentario es **bueno**

Para lograr esto, se utilizó la librería NLTK (Natural Language Toolkit) que es una biblioteca de Python para el procesamiento del lenguaje natural (PLN), de la cual se importó ***SentimentIntensityAnalyzer*** la cual forma parte del módulo nltk.sentiment.vader de NLTK, que se basa en el algoritmo VADER para calcular la puntuación de sentimiento de un texto; ya que VADER es un razonador de sentimientos que se utiliza más precisamente para análisis de sentimientos en texto.

Entonces mediante esta clase: SentimentIntensityAnalyzer, se pudo determinar la polaridad de los textos, que es básicamente asignarle los valores de la escala anteriormente mencionada a los comentarios hechos por los usuarios, para saber si estos son positivos, neutrales o negativos. 

Para este proyecto se tomaron las polaridades por defecto del modelo, que utiliza umbrales de -0.05 y 0.05. Las polaridades por debajo de -0.05 se clasifican como negativas, por encima de 0.05 como positivas, y las polaridades entre ambos umbrales se clasifican como neutrales.

En este apartado también se buscó la optimización de recursos al momento de hacer las consultas, así que se generaron dataframes con los datos relevantes para el análisis y se transformaron a formato parquet para una mayor eficiencia en el rendimiento del modelo. 

Los detalles del sentiment_analysis están sumados en los archivos del proyecto con un notebook llamado **Sentiment Analysis.**

### Desarrollo API

Para el desarrollo de la API se utilizó el framework FastAPI, el cual nos brinda un desarrollo sencillo y eficiente.  
Para esto se realizaron las funciones correspondientes a cada endpoint requerido.


- **Developer(desarrollador : str) :** Esta función recibe como parámetro el nombre de la empresa desarrolladora del juego, y devuelve la cantidad de items de juegos que desarrolla y el porcentaje de contenido free por año. 

- **UserData(user_id : str):** Esta función recibe el id del usuario y devuelve la cantidad de dinero gastado por el usuario, el porcentaje de recomendaciones que realizó y la cantidad de items que consume.

- **UserForGenre(genero : str):** Esta función recibe el nombre del género al que pertenece el juego y devuelve el top 5 de los usuarios con más horas de juego para el genero dado. Así que el de la primera posición será el que más horas de juego tiene acumuladas en dicho género. 

- **BestDeveloperYear(año : int):** En esta función se ingresa como parámetro un año que se quiera consultar y devuelve el top 3 de desarrolladores con juegos MÁS recomendados por usuarios para el año dado.

- **DeveloperReviewsAnalysis(desarrolladora : str):** Esta función recibe como parámetro una desarrolladora y devuelve un diccionario con el nombre de la desarrolladora como llave y una lista con la cantidad total de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento como valor positivo o negativo.

- **RecommendItem(item_id : str):** Esta función recibe como parámetro el nombre de un juego y devuelve una lista con 5 juegos recomendados similares al ingresado.

Los detalles de las funciones utilizadas en la API están sumados en los archivos del proyecto con un Jupyter Notebook llamado **Funciones API.**

### Análisis Exploratorio de Datos EDA 

Para el análisis exploratorio de datos de este proyecto se utilizaron librerías de Python como pandas y numpy para el manejo de los cálculos de datos, estadísticas, limpieza y análisis de los mismos; y las librerías matplotlib y seaborn para la visualización. 

El objetivo principal de la realización de un EDA es el conocimiento profundo de los datos con los cuales vamos a hacer el modelo de machine learning; en esta etapa se identifican patrones en los datos, se visualizan relaciones importantes entre ellos, como están distribuidos y esto nos ayuda también a identificar errores, redundancias o inconsistencias.El EDA es una etapa crucial en el desarrollo de un modelo de machine learning que contribuye a la construcción de un modelo robusto y bien estructurado.

En esta fase se crearon 3 funciones importantes para el desarrollo del análisis: *dataType(df), dataPorcentaje(df, colum) y recommendScore(row)*. Estas funciones permiten hacer análisis más detallados sobre los datos.

También para este sistema de recomendación se construyó un dataframe que contiene el id del usuario que hizo los reviews, los nombres de los juegos a los que se les realizaron comentarios y una columna de calificación que se generó mediante la combinación del análisis de sentimiento y las recomendaciones de juegos.  
Esta combinación se hizo mediante la función **merge()**, lo que permite una mejor selección de variables a partir de los datos en general.

Los detalles completos del EDA también se encuentran sumados en este repositorio en un Jupyter Notebook llamado **EDA.**

### Modelo de Aprendizaje Automático 

El sistema de recomendación modelado se ha dearrollado para tomar como input un juego y que devuelva como output una lista de juegos recomendados. El modelo se basa en una relación ítem-ítem, la cual evalúa la similitud del juego ingestado con los demás, devolviendo aquellos que son los más parecidos a él.

Para hacer este modelo se aplicó la métrica de similitud del coseno. Esta métrica funciona calculando la similitud entre dos vectores, basándose en el concepto del ángulo entre dos vectores en un espacio vectorial; así que el coseno del ángulo entre esos dos vectores muestra cuan alineados están esos vectores. 

Cabe resaltar que los vectores son los datos que ya han sido transformados en el análisis de sentimientos; ya que han sido pasados de ser texto a una escala numérica.

La similitud del coseno es aplicada mediante la librería de sckit-learn, importada de la siguiente manera: **from** sklearn.metrics.pairwise **import cosine_similarity.**

Los detalles del desarrollo del modelo de machine learning están en los archivos del repositorio en un Jupyter Notebook como: **Code ML**

También se encuentra el archivo **main.py** donde está el codigo desarrollado para la creación de la API.


## Autor:

- [@ingridbarriosv](https://www.github.com/ingridbarriosv)
Enlace del repositorio:
