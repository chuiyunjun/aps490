
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import random
import numpy as np
import torch

datetime = 'DateTime'

input1 = 'Modern Niagara > CPPIB > 17th Flr > VAV-1704 > Primary Air.Air Flow Setpoint'
#second input is our output
input2 = 'Modern Niagara > CPPIB > 17th Flr > VAV-1704 > Fin Tube Radiation.Valve Position'
input3 = 'Modern Niagara > CPPIB > 17th Flr > VAV-1704 > Zone Air.Temperature'
input4 = 'Modern Niagara > CPPIB > 17th Flr > VAV-1704 > Primary Air.Air Flow'
input5 = 'Temp (°C)'

INPUT_LIST_V = [datetime, input1, input2, input3, input4, input5]
INPUT_LIST_A = [datetime, input1, input4, input3, input2, input5]

class Data:
    def __init__(self, option = 'V', data_root='./prediction/datasets/train/') -> None:

        if option == 'V':
            self.INPUT_LIST = INPUT_LIST_V
        else:
            self.INPUT_LIST = INPUT_LIST_A
        self._min, self._max = None, None
        self._mng = self.read_mng_data(data_root)
        self._weather = self.read_weather_data()
        self._data = self.weather_join_mng()
        self._normalize_data = self.normalize_data()
        
    def get_min(self):
        return self._min
    
    def get_max(self):
        return self._max
        
    def read_mng_data(self, data_root):
        mng_list = []
        for file in os.listdir(data_root):
            df = pd.read_csv(data_root + file)
            mng_list.append(df)
        mng = pd.concat(mng_list, axis=0)
        mng.sort_values(datetime)
        mng = mng[self.INPUT_LIST[:-1]]
        return mng
    
    def read_weather_data(self):
        weather_list = []
        weather_data_root = './prediction/datasets/weather/'
        for file in os.listdir(weather_data_root):
            df = pd.read_csv(weather_data_root + file)
            weather_list.append(df)
        weather = pd.concat(weather_list, axis=0)
        
        weather = weather[['Date/Time (LST)', 'Temp (°C)']]
        weather.rename(columns = {'Date/Time (LST)':'DateTime'}, inplace = True)
        return weather
    
    def weather_join_mng(self):

        self._weather['DateTime 2'] = pd.to_datetime(self._weather.DateTime)
        self._mng['DateTime'] = pd.to_datetime(self._mng.DateTime)
        self._mng['DateTime 2'] = self._mng['DateTime'].dt.floor('h')
        data = pd.merge(self._mng, self._weather, how='left', on = 'DateTime 2')
        data.to_csv('data.csv')
        data = data.dropna(how='any')
        data.rename(columns = {'DateTime_x': datetime}, inplace = True)
        data = data[[datetime, input1, input2, input3, input4, input5]]

        return data
    
    def normalize_data(self):
        inputs = self._data[self.INPUT_LIST[1:]]
        sc = MinMaxScaler()
        normalized_inputs = sc.fit_transform(inputs)
        
        self._min = sc.data_min_[1]
        self._max = sc.data_max_[1]
        
        #
        datetime = self._data[self.INPUT_LIST[:1]]
        
        return normalized_inputs

    def get_og_data(self):
        return self._data
    
    def get_normalized_data(self):
        return self._normalize_data
    
    def recover_y(self, normalized_y):
        return self._min + (self._max - self._min) * normalized_y


def sliding_windows(data, seq_length, pred_length, shuffle=True):
    print(data)
    x = []
    y = []
    samples = []
    for i in range(len(data)-seq_length-pred_length):
        sample = data[i:(i+seq_length+pred_length)]
        if np.array(sample[seq_length:,1]).prod() != 0:
            samples.append(sample)
    if shuffle:
        random.shuffle(samples)
    x = np.array(samples)[:,:seq_length,:]
    y = np.array(samples)[:,seq_length:,:]
    y = y[:,:,1]
    
    return torch.Tensor(x), torch.Tensor(y)
    
    
        
 