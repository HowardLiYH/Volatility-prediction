
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from matplotlib.pyplot import figure
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.graphics.tsaplots import plot_predict
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
import pmdarima as pm
from ThymeBoost import ThymeBoost as tb
import math
import sklearn.metrics as metrics
from datetime import timedelta, datetime



def log_price(list_stock_prices):
    return np.log(list_stock_prices)

def realized_volatility(data):
    list_vol = []
    i = 0
    for i in data.index:
        x = np.std(data.log_returns.iloc[:i])
        i += 1
        list_vol.append(x)

    return list_vol

def preprocessing(data):
    # Weighted Average Price
    data['WAP'] = (data['bp_1']*data['bz_1']
                   +data['bp_2']*data['bz_2']
                   +data['ap_1']*data['az_1']
                   +data['ap_2']*data['az_2'])/(data['bz_1']+
                                             data['bz_2']+
                                             data['az_1']+
                                             data['az_2'])
    ## Log price
    data.insert(0, 'log_price', log_price(data['WAP']))
    data['log_returns'] = data.log_price.diff()

    ## Realized Volatility

    data['realized_volatility'] = realized_volatility(data)

    ## Previous volatility

    data['volatility_t+1'] = data['realized_volatility'].shift(-1)

    y = data['realized_volatility'].to_frame()

    ## Drop the target y
    data.drop(['realized_volatility'], axis = 1, inplace = True)

    df = data
    ## Imputer work on df['volatility_t+1']

    imputer = SimpleImputer(strategy="constant", fill_value = 0) # Instantiate a SimpleImputer object with your strategy of choice

    imputer.fit(df[['volatility_t+1']]) # Call the "fit" method on the object

    df['volatility_t+1'] = imputer.transform(df[['volatility_t+1']]) # Call the "transform" method on the object

    ## Imputer2 work on df['log_returns']

    imputer2 = SimpleImputer(strategy="constant", fill_value = df.iloc[1,84])

    imputer2.fit(df[['log_returns']]) # Call the "fit" method on the object

    df['log_returns'] = imputer2.transform(df[['log_returns']]) # Call the "transform" method on the object

    ## Imputer3 work on y['realized_volatility']

    imputer3 = SimpleImputer(strategy="constant", fill_value = 0) # Instantiate a SimpleImputer object with your strategy of choice

    imputer3.fit(y[['realized_volatility']]) # Call the "fit" method on the object

    y['realized_volatility'] = imputer3.transform(y[['realized_volatility']]) # Call the "transform" method on the object

    return y['realized_volatility']

def model_prediction(y):

    boosted_model = tb.ThymeBoost(verbose=0)
    model = boosted_model.fit(y,
                                trend_estimator=['ransac', 'arima'],
                                arima_order='auto',
                                global_cost='mse')
    predicted_output = boosted_model.predict(model, 5760) # 5760 * 5 as 8hours prediction

    y_pred = predicted_output['predictions']

    return y_pred

def prediction_display():
    data = pd.read_csv('scraping/new_data.csv')
    y = preprocessing(data)
    #prediction = model_prediction(y)
    prediction = np.random.randn(5760)

    prediction_time = []
    last_time = pd.to_datetime(data["scrape_time"].iloc[-1])
    time_change = timedelta(seconds=5)
    for i in range(5760):
        last_time = last_time + time_change
        prediction_time.append(last_time)

    display = pd.DataFrame({"time": prediction_time, 'prediction':prediction})

    return display.reset_index(drop=True)
