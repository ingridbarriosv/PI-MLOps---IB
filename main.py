from fastapi import FastAPI
import pandas as pd
from fastapi.responses import JSONResponse
import pyarrow

app = FastAPI()


#http://127.0.0.1:8000/

@app.get("/")
async def root():
    return {"Mensaje': '¡Hola! Los endpoints que verán son los siguientes: 'developer':'Desarrollador', 'userData':'Nombre de usuario', 'userForGenre':'Género de juego','BestDeveloperYear':'Año','DeveloperReviewsAnalysis':'Desarrollador','RecommendItem':'Nombre de juego"}

@app.get("/developer/{dev}")
def developer(dev:str):
    # Se leen los datos y se crea el dataFrame a partir del dataframe total anteriormente modificado
    dfGames = pd.read_parquet('DB Steam/steam_games.parquet')
    # Se crea un dataFrame filtrado de acuerdo a la empresa desarrolladora, convirtiendo los str de developer a minusculas
    dfFiltered =  dfGames[dfGames['developer'].str.lower() == dev.lower()]
    # Se guardan los valores del total de juegos y la cantidad de juegos gratuitos agrupados por año
    totalApps = dfFiltered.groupby('release_year').size().reset_index(name = 'Cantidad de Items')
    # Se borran los valores nulos de la columna 'price' del nuevo dataframe filtrado
    dfFiltered = dfFiltered.dropna(subset = ['price'])
    # Se filtra dfFiltered para incluir solo los juegos que sean gratuitos por cada año de lanzamiento en un nuevo dataframe llamado freeApps
    freeApps = dfFiltered[dfFiltered['price'] == 0].groupby('release_year').size().reset_index(name = 'free_items')
    # Se crea el dataFrame resultado con la cantidad de juegos, el porcentaje de los que son gratis, y el año
    # Para esto se combinan los dataframes totalApps y freeApps en función de la columna 'release_year'
    dfResult = pd.merge(totalApps, freeApps, on = 'release_year', how = 'left').fillna(0)
    dfResult.rename(columns = {'release_year': 'Año'}, inplace = True)
    # Se saca el porcentaje de contenido free por año
    dfResult['Contenido Free'] = ((dfResult['free_items'] / dfResult['Cantidad de Items']) * 100).round(2).astype(str) + '%'
    dfResult = dfResult[['Año','Cantidad de Items','Contenido Free']].reset_index(drop = True)
    dfResult = dfResult.to_dict(orient = 'records')
    return JSONResponse(content = dfResult)

@app.get("/userData/{userId}")
def userData(userId:str):
    # Se leen los datos y se crean los DataFrames a partir de los formatos parquet que habíamos creado 
    dfUserRecommend = pd.read_parquet('DB Steam/Reviews_sep_sentiment.parquet')
    userItemsDf = pd.read_parquet('DB Steam/items_sep.parquet')
    userItemCountDf = pd.read_parquet('DB Steam/items_count.parquet')
    dfSteamGamesPrice = pd.read_parquet('DB Steam/GamesPrice.parquet')
    # Se crean DataFrames filtrados por usuario
    dfUserRecommend =  dfUserRecommend[dfUserRecommend['user_id'] == userId].reset_index(drop = True)
    userItemsDf = userItemsDf[userItemsDf['user_id'] == userId].reset_index(drop = True)
    # Se combinan los dataframes userItemsDf y dfSteamGamesPrice en función de la columna 'item_id'
    userItemsDf = pd.merge(userItemsDf, dfSteamGamesPrice, how = 'left', on = 'item_id')
    userItemCountDf = userItemCountDf[userItemCountDf['user_id'] == userId].reset_index(drop = True)
    # Se compone el DataFrame de final conteniendo el ID de usuario, el dinero gastado de acuerdo a sus ítems, el porcentaje de recomendación y la cantidad de ítems
    resultData = {'Usuario': [userId]}
    resultDf = pd.DataFrame(resultData)
    # Se crea una nueva columna para calcular el total del dinero gastado 
    resultDf['Dinero Gastado'] = pd.to_numeric(userItemsDf['price'], errors = 'coerce').sum()
    resultDf['% de recomendación'] = ((dfUserRecommend['recommend'].mean()) * 100).round(2).astype(str) + '%'
    resultDf['Cantidad de Items'] = userItemCountDf['items_count']
    result = resultDf.to_dict(orient = 'records')
    return JSONResponse(content = result)

@app.get("/userForGenre/{genre}")
def userForGenre(genre:str):
    # Se leen los datos y se crea el dataFrame
    dfSteamGamesGenres = pd.read_parquet('DB Steam/GamesGenre_sep.parquet')
    dfUserItems_sep = pd.read_parquet('DB Steam/items_sep.parquet')
    # Se filtra el dataFrame buscando el género solicitado entre los géneros del juego, pasados a minúsculas
    dfSteamGamesGenres = dfSteamGamesGenres[dfSteamGamesGenres['genres'].str.lower() == genre.lower()]
    # Se combina el dataFrame con los géneros 'dfSteamGamesGenres' al dataFrame que contiene a los usuarios con sus ítems 'dfUserItems_sep'
    dfSteamGamesGenres = pd.merge(dfSteamGamesGenres[['item_id','release_year']], dfUserItems_sep, how = 'left', on = 'item_id')
    # Se eliminan todas los valores nulos de la columna del tiempo de juego 
    dfSteamGamesGenres = dfSteamGamesGenres.dropna(subset = 'playtime')
    # Se agrupan de acuerdo al ID de usuario teniendo en cuenta el tiempo total de juego y se ordenan de forma descendente por su posición 
    userId = dfSteamGamesGenres.groupby('user_id')['playtime'].sum().reset_index().sort_values(by = 'playtime',ascending = False).iloc[0,0]
    # Se filtra el dataFrame de acuerdo al usuario recuperado, se agrupa conforme al año de lanzamiento de los juegos y se suma el tiempo jugado
    dfSteamGamesGenres = dfSteamGamesGenres[dfSteamGamesGenres['user_id'] == userId].groupby('release_year')['playtime'].sum().reset_index().sort_values(by = 'release_year',ascending = False)
    # Se hace la transforman los datos y se crea el dataFrame de respuesta
    dfSteamGamesGenres.rename(columns = {'release_year': 'Año'}, inplace = True)
    dfSteamGamesGenres.rename(columns = {'playtime': 'Horas jugadas'}, inplace = True)
    resultPlaytime = dfSteamGamesGenres.to_dict(orient = 'records')
    result_userForGenre = {'Usuario con más horas jugadas para el género ' + genre:userId,'Horas totales':resultPlaytime}
    return JSONResponse(content = result_userForGenre)

@app.get("/BestDeveloperYear/{year}")
def BestDeveloperYear(año:int):
    # Se cargan los datos y se crean los dataFrame
    dfUserReview = pd.read_parquet('DB Steam/Reviews_sep_sentiment.parquet')
    dfGamesBestDev = pd.read_parquet('DB Steam/GamesDevs.parquet')
    # Se filtra el dataFrame de acuerdo al año solicitado, descartando los juegos no recomendados, y con análisis de sentimiento neutral o negativo
    dfUserReview = dfUserReview[(dfUserReview['recommend'] == True) & 
                                            (dfUserReview['sentiment_analysis'] == '2') & 
                                            (dfUserReview['review_year'] == año)]
    dfUserReview['review_year'] = dfUserReview['review_year'].astype(int)
    # Se agrupan las reseñas de acuedo al ID del juego sumando la cantidad de recomendaciones
    dfUserReview = dfUserReview.groupby('item_id')['recommend'].count().reset_index()
    # Se crea un dataFrame que contenga el ID del juego, el desarrollador y la suma de recomendaciones
    dfGamesBestDev = pd.merge(dfGamesBestDev[['item_id','developer']], dfUserReview, how = 'left', on = 'item_id').sort_values(by = 'recommend', ascending = False)
    # Se agrupa de acuerdo al desarrollador y a la suma de recomendaciones
    dfGamesBestDev = dfGamesBestDev.groupby('developer')['recommend'].sum().reset_index().sort_values(by = 'recommend', ascending = False)
    # Se revisa que el primer valor no sea 0 para descartar años en los que no se tenga registro de las reseñas y se recuperan los tres primeros lugares
    if dfGamesBestDev.iloc[1,1]!= 0:
        primero = dfGamesBestDev.iloc[0,0]
        segundo = dfGamesBestDev.iloc[1,0]
        tercero = dfGamesBestDev.iloc[2,0]
        resultBestDev = [{'Puesto 1':primero},{'Puesto 2':segundo},{'Puesto 3':tercero}]
    else:
        resultBestDev = [{'Puesto 1':None},{'Puesto 2':None},{'Puesto 3':None}]
    return JSONResponse(content = resultBestDev)

@app.get("/DeveloperReviewsAnalysis/{dev}")
def DeveloperReviewsAnalysis(dev:str):
    # Se cargan los datos y se crean los DataFrame
    dfUserReview = pd.read_parquet('DB Steam/Reviews_sep_sentiment.parquet')
    dfGamesDev = pd.read_parquet('DB Steam/GamesDevs.parquet')
    # Se filtra el DataFrame de desarrolladores de acuerdo al desarrollador solicitado
    dfGamesDev = dfGamesDev[dfGamesDev['developer'].str.lower() == dev.lower()]
    # Se agrupa el DataFrame de reseñas de acuerdo al ID del juego agregando una columna para la suma de comentarios positivos y otra para la de negativos
    dfUserReview = dfUserReview.groupby('item_id')['sentiment_analysis'].agg([('positive', lambda x: (x == '2').sum()),
        ('negative', lambda x: (x == '0').sum())]).reset_index()
    # Se crea el DataFrame de respuesta
    dfGamesDev = pd.merge(dfGamesDev, dfUserReview, how = 'left', on = 'item_id')
    dfGamesDev = dfGamesDev.groupby('developer')[['positive','negative']].sum().reset_index()
    positive = dfGamesDev.iloc[0,1].astype(int).astype(str)
    negative = dfGamesDev.iloc[0,2].astype(int).astype(str)
    resultDevRev = {dfGamesDev.iloc[0,0]:['Negative = ' + negative,'Positive = ' + positive]}
    return JSONResponse(content = resultDevRev)

@app.get("/recommenditem/{itemId}")
def RecommendItem(itemId: str):
    # Se cargan los datos y se crea el dataFrame
    dfItemSim = pd.read_parquet('DB Steam/item_Sim.parquet')
    counter = 1
    resultRecomm = {'Aquí hay juegos similares a': itemId,'1':'','2':'','3':'','4':'','5':''}
    # Se buscan los ítems con mayor similitud y se añaden al resultado
    for item in dfItemSim.sort_values(by = itemId, ascending = False).index[1:6]:
        resultRecomm[str(counter)] = item
        counter +=1
    return JSONResponse(content = resultRecomm)
