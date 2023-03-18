import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

def split_data(stock, lookback):
    data_raw = stock.to_numpy() 
    data = []
    
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback): 
        data.append(data_raw[index: index + lookback])
    
    data = np.array(data) # n 17
    test_set_size = int(np.round(0.2*data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)
    
    x_train = data[:train_set_size,:-1] # n , 19 , 17
    y_train = data[:train_set_size,-1,-1] # n , 1 
    
    x_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1,-1]
    
    return [x_train, y_train, x_test, y_test]

def string_to_time(date):
    
    date = datetime.strptime(date, '%Y-%m-%d')
    date = datetime.timestamp(date)
    return date 

def normalize(x, min, max):
    return (x - min)/(max - min)

def reverse_normalize(y, min, max):
    return y*(max - min) + min

def min_max_dic(dataset):
    dic = {}
    for col in dataset.columns:
        dic[col] = [dataset[col].min(), dataset[col].max()]
        dataset[col] = dataset[col].apply(lambda x: normalize(x, dic[col][0], dic[col][1]))
    return dic


    # print(y_test)
    train = np.concatenate((X_train, y_train), axis=1)
    # print(train[0])
    scaler = MinMaxScaler().fit(train)
    scaled_train = scaler.transform(train)
    print(scaled_train.shape)
    # y_scaled_train = scaler.transform(y_train)
    test = np.concatenate((X_test, y_test), axis=1)
    scaled_test = scaler.transform(test)
    # y_scaled_test = scaler.transform(y_test)

    # print(scaled_train[0])
    # print("Saving artifacts...")
    # Path("artifacts").mkdir(exist_ok=True)
    dump(scaler, open('scaler.pkl', 'wb'))

    # self.nscaler_features = scaled_train.shape[1]

    return (
        scaled_train[:, :-1],
        scaled_train[:, -1].reshape(-1, 1),
        scaled_test[:, :-1],
        scaled_test[:, -1].reshape(-1, 1),
    )

def prepare_regression_data(X_data, y_data, look_back=30):
    dataX, dataY = [], []
    for i in range(len(X_data) - look_back - 1):
        a = X_data[i : (i + look_back)]
        dataX.append(a)
        # print(i + look_back+1)
        # print(y_data[i + look_back+1, 0])
        dataY.append(y_data[i + look_back + 1, 0])

    return np.array(dataX), np.array(dataY)
