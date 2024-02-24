import urllib.parse
import requests
import json

from flask import Flask
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures

app = Flask(__name__)

API_KEY = "VSJTPURWV477GP6K5L8ZX8UWN"
UNIT_GROUP = "metric"

def getWeatherForecast(LOCATION):
    requestUrl = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/" + urllib.parse.quote_plus(LOCATION)
    requestUrl = requestUrl + "?key=" + API_KEY + "&unitGroup=" + UNIT_GROUP + "&include=days"

    print(requestUrl)
    try:
        req = requests.get(requestUrl)
    except :
        print("error")
        return []
    return req.json()

@app.route('/')
def home():
    message = "Specifier dans l'URL l'un de ces chemin: /predict, /temp, /humidity, /windDir, /solarRadiation, /windSpeed"
    return message

@app.route('/predict')
def predict():
    weather_df = pd.read_csv('VisualData.csv', parse_dates=['datetime'], index_col='datetime')
    weather_df_num = weather_df.loc[:,['tempmax', 'dew', 'humidity', 'precipcover', 'solarradiation', 'solarenergy', 'uvindex']]
    weather_y = weather_df_num.pop("tempmax")
    weather_x = weather_df_num
    train_X, test_X, train_y, test_y = train_test_split(weather_x, weather_y, test_size=0.2, random_state=4)
    model = LinearRegression()
    model.fit(train_X, train_y)
    prediction = model.predict(test_X)

    return pd.DataFrame({'prediction':prediction}).to_json()

@app.route('/temp')
def predictTemp():
    weather_df = pd.read_csv('DataTata.csv', parse_dates=['Datetime'], index_col='Datetime')
    weather_df['Batterie (V)'].fillna(value=6.872137, inplace=True)
    weather_df['Rayonnement global (j/cm²)'].fillna(value=2012.106471, inplace=True)
    weather_df['Température (°C)'].fillna(value=24.747292, inplace=True)
    weather_df['ETo (mm)'].fillna(value=4.700938, inplace=True)
    weather_df['Humidité relative (%)'].fillna(value=12.504310, inplace=True)
    weather_df['Précipitation (mm)'].fillna(value=0.221838, inplace=True)
    weather_df['Vitesse du vent (km/h)'].fillna(value=14.061619, inplace=True)
    weather_df['Direction du vent (°)'].fillna(value=175.754765, inplace=True)
    weather_df_num = weather_df.loc[:,
                     ['Température (°C)', 'tempmax', 'tempmin', 'temp', 'humidity', 'solarradiation', 'solarenergy',
                      'uvindex']]
    weather_y = weather_df_num.pop("Température (°C)")
    weather_x = weather_df_num
    train_X, test_X, train_y, test_y = train_test_split(weather_x, weather_y, test_size=0.2, random_state=4)
    poly_regr = PolynomialFeatures(degree=3)  # our polynomial model is of order
    X_poly = poly_regr.fit_transform(train_X)  # transforms the features to the polynomial form
    polyReg = LinearRegression()  # creates a linear regression object
    polyReg.fit(X_poly, train_y)  # fits the linear regression object to the polynomial features
    weatherForecast = getWeatherForecast("tata%20maroc")

    print('Weather forecast of Temperature for {location}'.format(location=weatherForecast['resolvedAddress']))
    days = weatherForecast['days']

    dataframe = pd.DataFrame(days)

    dataframe_num = dataframe.loc[:, ['tempmax', 'tempmin', 'temp', 'humidity', 'solarradiation', 'solarenergy',
                      'uvindex']]

    prediction = polyReg.predict(poly_regr.fit_transform(dataframe_num))

    return pd.DataFrame({'Date':dataframe['datetime'],'prediction':prediction}).to_json()


@app.route('/humidity')
def predictHumidity():
    weather_df = pd.read_csv('DataTata.csv', parse_dates=['Datetime'], index_col='Datetime')
    weather_df['Batterie (V)'].fillna(value=6.872137, inplace=True)
    weather_df['Rayonnement global (j/cm²)'].fillna(value=2012.106471, inplace=True)
    weather_df['Température (°C)'].fillna(value=24.747292, inplace=True)
    weather_df['ETo (mm)'].fillna(value=4.700938, inplace=True)
    weather_df['Humidité relative (%)'].fillna(value=12.504310, inplace=True)
    weather_df['Précipitation (mm)'].fillna(value=0.221838, inplace=True)
    weather_df['Vitesse du vent (km/h)'].fillna(value=14.061619, inplace=True)
    weather_df['Direction du vent (°)'].fillna(value=175.754765, inplace=True)
    weather_df_num=weather_df.loc[:,['Humidité relative (%)','tempmax','tempmin','temp','humidity','precipcover','solarradiation','solarenergy','uvindex']]
    weather_y = weather_df_num.pop("Humidité relative (%)")
    weather_x = weather_df_num
    train_X, test_X, train_y, test_y = train_test_split(weather_x, weather_y, test_size=0.2, random_state=4)
    regr = RandomForestRegressor(max_depth=90, random_state=0, n_estimators=100)
    regr.fit(train_X, train_y)

    weatherForecast = getWeatherForecast("tata%20maroc")

    #print('Weather forecast of Humidity for {location}'.format(location=weatherForecast['resolvedAddress']))
    days = weatherForecast['days']

    dataframe = pd.DataFrame(days)
    #dataframe.fillna(dataframe.mean(), inplace=True)
    dataframe_num = dataframe.loc[:, ['tempmax','tempmin','temp','humidity','precipcover','solarradiation','solarenergy','uvindex']]
    prediction = regr.predict(dataframe_num)
    return pd.DataFrame({'Date':dataframe['datetime'],'prediction':prediction}).to_json()


@app.route('/precip')
def predictPrecip():
    weather_df = pd.read_csv('DataTata.csv', parse_dates=['Datetime'], index_col='Datetime')
    weather_df['Batterie (V)'].fillna(value=6.872137, inplace=True)
    weather_df['Rayonnement global (j/cm²)'].fillna(value=2012.106471, inplace=True)
    weather_df['Température (°C)'].fillna(value=24.747292, inplace=True)
    weather_df['ETo (mm)'].fillna(value=4.700938, inplace=True)
    weather_df['Humidité relative (%)'].fillna(value=12.504310, inplace=True)
    weather_df['Précipitation (mm)'].fillna(value=0.221838, inplace=True)
    weather_df['Vitesse du vent (km/h)'].fillna(value=14.061619, inplace=True)
    weather_df['Direction du vent (°)'].fillna(value=175.754765, inplace=True)
    weather_df_num = weather_df.loc[:, ['Précipitation (mm)', 'precip', 'precipcover']]
    weather_y = weather_df_num.pop("Précipitation (mm)")
    weather_x = weather_df_num
    train_X, test_X, train_y, test_y = train_test_split(weather_x, weather_y, test_size=0.2, random_state=4)
    poly_regr = PolynomialFeatures(degree=2)
    X_poly = poly_regr.fit_transform(train_X)
    model = LinearRegression()
    model.fit(X_poly, train_y)

    weatherForecast = getWeatherForecast("tata%20maroc")

    print('Weather forecast of Precipitation for {location}'.format(location=weatherForecast['resolvedAddress']))
    days = weatherForecast['days']

    dataframe = pd.DataFrame(days)
    dataframe.fillna(dataframe.mean(), inplace=True)

    dataframe_num = dataframe.loc[:, [ 'precip', 'precipcover']]

    prediction = model.predict(poly_regr.fit_transform(dataframe_num))

    return pd.DataFrame({'Date':dataframe['datetime'],'prediction':prediction}).to_json()


@app.route('/solarRadiation')
def predictSolarRadiation():
    weather_df = pd.read_csv('DataTata.csv', parse_dates=['Datetime'], index_col='Datetime')
    weather_df['Batterie (V)'].fillna(value=6.872137, inplace=True)
    weather_df['Rayonnement global (j/cm²)'].fillna(value=2012.106471, inplace=True)
    weather_df['Température (°C)'].fillna(value=24.747292, inplace=True)
    weather_df['ETo (mm)'].fillna(value=4.700938, inplace=True)
    weather_df['Humidité relative (%)'].fillna(value=12.504310, inplace=True)
    weather_df['Précipitation (mm)'].fillna(value=0.221838, inplace=True)
    weather_df['Vitesse du vent (km/h)'].fillna(value=14.061619, inplace=True)
    weather_df['Direction du vent (°)'].fillna(value=175.754765, inplace=True)
    weather_df_num = weather_df.loc[:,
                     ['Rayonnement global (j/cm²)', 'tempmax', 'solarradiation', 'solarenergy', 'uvindex', 'humidity']]
    weather_y = weather_df_num.pop("Rayonnement global (j/cm²)")
    weather_x = weather_df_num
    train_X, test_X, train_y, test_y = train_test_split(weather_x, weather_y, test_size=0.2, random_state=4)
    model = LinearRegression()
    model.fit(train_X, train_y)

    weatherForecast = getWeatherForecast("tata%20maroc")

    print('Weather forecast of Solar radiation for {location}'.format(location=weatherForecast['resolvedAddress']))
    days = weatherForecast['days']

    dataframe = pd.DataFrame(days)
    dataframe.fillna(dataframe.mean(), inplace=True)
    dataframe_num = dataframe.loc[:, ['tempmax', 'solarradiation', 'solarenergy', 'uvindex', 'humidity']]

    prediction = model.predict(dataframe_num)

    return pd.DataFrame({'Date':dataframe['datetime'],'prediction':prediction}).to_json()


@app.route('/windDir')
def predictWindDir():
    weather_df = pd.read_csv('DataTata.csv', parse_dates=['Datetime'], index_col='Datetime')
    weather_df['Batterie (V)'].fillna(value=6.872137, inplace=True)
    weather_df['Rayonnement global (j/cm²)'].fillna(value=2012.106471, inplace=True)
    weather_df['Température (°C)'].fillna(value=24.747292, inplace=True)
    weather_df['ETo (mm)'].fillna(value=4.700938, inplace=True)
    weather_df['Humidité relative (%)'].fillna(value=12.504310, inplace=True)
    weather_df['Précipitation (mm)'].fillna(value=0.221838, inplace=True)
    weather_df['Vitesse du vent (km/h)'].fillna(value=14.061619, inplace=True)
    weather_df['Direction du vent (°)'].fillna(value=175.754765, inplace=True)
    weather_df_num = weather_df.loc[:, ['Direction du vent (°)', 'winddir', 'windspeed']]
    weather_y = weather_df_num.pop("Direction du vent (°)")
    weather_x = weather_df_num
    train_X, test_X, train_y, test_y = train_test_split(weather_x, weather_y, test_size=0.2, random_state=4)
    poly_regr = PolynomialFeatures(degree=5)  # our polynomial model is of order
    X_poly = poly_regr.fit_transform(train_X)  # transforms the features to the polynomial form
    polyReg = LinearRegression()  # creates a linear regression object
    polyReg.fit(X_poly, train_y)  # fits the linear regression object to the polynomial features

    weatherForecast = getWeatherForecast("tata%20maroc")

    print('Weather forecast of Wind Direction for {location}'.format(location=weatherForecast['resolvedAddress']))
    days = weatherForecast['days']

    dataframe = pd.DataFrame(days)
    dataframe.fillna(dataframe.mean(), inplace=True)
    dataframe_num = dataframe.loc[:, [ 'winddir', 'windspeed']]
    prediction = polyReg.predict(poly_regr.fit_transform(dataframe_num))

    return pd.DataFrame({'Date':dataframe['datetime'],'prediction':prediction}).to_json()


@app.route('/windSpeed')
def predictWindSpeed():
    weather_df = pd.read_csv('DataTata.csv', parse_dates=['Datetime'], index_col='Datetime')
    weather_df['Batterie (V)'].fillna(value=6.872137, inplace=True)
    weather_df['Rayonnement global (j/cm²)'].fillna(value=2012.106471, inplace=True)
    weather_df['Température (°C)'].fillna(value=24.747292, inplace=True)
    weather_df['ETo (mm)'].fillna(value=4.700938, inplace=True)
    weather_df['Humidité relative (%)'].fillna(value=12.504310, inplace=True)
    weather_df['Précipitation (mm)'].fillna(value=0.221838, inplace=True)
    weather_df['Vitesse du vent (km/h)'].fillna(value=14.061619, inplace=True)
    weather_df['Direction du vent (°)'].fillna(value=175.754765, inplace=True)
    weather_df_num = weather_df.loc[:,
                     ['Vitesse du vent (km/h)', 'windspeed', 'winddir', 'solarradiation', 'solarenergy']]
    weather_y = weather_df_num.pop("Vitesse du vent (km/h)")
    weather_x = weather_df_num
    train_X, test_X, train_y, test_y = train_test_split(weather_x, weather_y, test_size=0.2, random_state=4)
    regr = RandomForestRegressor(max_depth=90, random_state=0, n_estimators=100)
    regr.fit(train_X, train_y)

    weatherForecast = getWeatherForecast("tata%20maroc")

    print('Weather forecast of Wind Speed for {location}'.format(location=weatherForecast['resolvedAddress']))
    days = weatherForecast['days']

    dataframe = pd.DataFrame(days)

    dataframe_num = dataframe.loc[:, ['windspeed', 'winddir', 'solarradiation', 'solarenergy']]
    prediction = regr.predict(dataframe_num)

    return pd.DataFrame({'Date':dataframe['datetime'],'prediction':prediction}).to_json()

if __name__ == '__main__':
    app.run()
