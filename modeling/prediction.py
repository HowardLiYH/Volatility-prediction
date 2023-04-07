import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import joblib
from datetime import timedelta

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

def load_data(path):
    suggest_columns=['scrape_time','lastUpdateId','bp1','bs1','bp2','bs2',
                    'bp3','bs3','bp4','bs4','bp5','bs5','bp6','bs6','bp7',
                    'bs7','bp8','bs8','bp9','bs9','bp10','bs10','bp11',
                    'bs11','bp12','bs12','bp13','bs13','bp14','bs14',
                    'bp15','bs15','bp16','bs16','bp17','bs17','bp18',
                    'bs18','bp19','bs19','bp20','bs20','ap1','as1','ap2',
                    'as2','ap3','as3','ap4','as4','ap5','as5','ap6','as6',
                    'ap7','as7','ap8','as8','ap9','as9','ap10','as10','ap11',
                    'as11','ap12','as12','ap13','as13','ap14','as14','ap15',
                    'as15','ap16','as16','ap17','as17','ap18','as18','ap19',
                    'as19','ap20','as20']
    data = pd.read_csv(path, skiprows=lambda x:x>0 and x % 12 !=0)
    if list(data.columns) == suggest_columns:
        return data
    data.columns = suggest_columns
    return data

def data_preprocess(data):
    data['WAP'] = (data['bp1']*data['bs1']
               +data['bp2']*data['bs2']
               +data['ap1']*data['as1']
               +data['ap2']*data['as2'])/(data['bs1']+
                                         data['bs2']+
                                         data['as1']+
                                         data['as2'])
    data['spread'] = ((data['ap1']/data['bp1']) - 1)
    data.insert(0, 'log_price', log_price(data['WAP']))
    data['log_returns'] = data.log_price.diff()
    data['realized_volatility'] = realized_volatility(data)
    data['volatility_t+1'] = data['realized_volatility'].shift(-1)
    imputer = SimpleImputer(strategy="constant", fill_value = 0) # Instantiate a SimpleImputer object with your strategy of choice
    imputer.fit(data[['volatility_t+1']]) # Call the "fit" method on the object
    data['volatility_t+1'] = imputer.transform(data[['volatility_t+1']])
    stime = pd.to_datetime(data['scrape_time'])
    data['scrape_time'] = (stime - stime.shift(1)).dt.total_seconds()
    data['bid depth'] = data[['bs1', 'bs2', 'bs3','bs4', 'bs5', 'bs6','bs7', 'bs8', 'bs9','bs10',
                         'bs11', 'bs12', 'bs13','bs14', 'bs15', 'bs16','bs17', 'bs18', 'bs19','bs20']].sum(axis=1)
    data['ask depth'] = data[['as1', 'as2', 'as3','as4', 'as5', 'as6','as7', 'as8', 'as9','as10',
                         'as11', 'as12', 'as13','as14', 'as15', 'as16','as17', 'as18', 'as19','as20']].sum(axis=1)
    data['FDOFI'] = (data['bid depth']-data['ask depth'])/(data['bid depth']+data['ask depth'])
    return data[2:-1]

def p480(model, dataf, look):
    # flist = ['WAP', 'spread', 'log_price', 'scrape_time']
    flist = ['WAP', 'spread', 'FDOFI', 'log_price', 'scrape_time']
    features = dataf[flist]
    scaled_data = np.hstack((MinMaxScaler().fit_transform(features), np.expand_dims(dataf['volatility_t+1'].values, 1)))
    newest_data = scaled_data[-look['back']:,:].reshape([1,look['back'],len(flist)+1])
    prediction = model.predict(newest_data)
    # print(prediction.shape)
    predictions = [*prediction[0]]
    while len(predictions) < 480:
        newest_data = np.hstack((newest_data, prediction))[:,-look['back']:,:]
        # print(newest_data.shape)
        prediction = model.predict(newest_data)
        # print(prediction.shape)
        for p in prediction[0]:
            predictions.append(p)
    return np.array(predictions)

def prediction_display():
    model1 = joblib.load('modeling/modelLSTM.joblib')
    # load scaler
    data = load_data("scraping/new_data.csv")
    scrape_time = data["scrape_time"].iloc[-1]
    data_p = data_preprocess(data)
    prediction = p480(model=model1, dataf=data_p, look={'back': 64, 'forward': 32})[:,-1] # pass scaler

    prediction_time = []
    last_time = pd.to_datetime(scrape_time)
    time_change = timedelta(seconds=60)
    for i in range(480):
        last_time = last_time + time_change
        prediction_time.append(last_time)

    display = pd.DataFrame({"time(GMT)": prediction_time, 'prediction':prediction*1000})

    # display['prediction'] = display['prediction'].expanding().mean()

    return display
