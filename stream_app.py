import streamlit as st
import torch.nn as nn
from torch.autograd import Variable
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from prediction import data, main
from prediction.model import *
import seaborn as sns
from pycaret.anomaly import *
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import os
import random
from pynput.keyboard import Key, Controller




def load_input_data():
    
    d = data.Data()
    input = d.get_og_data()

    return input


def format_names(name):
  temp = name.split(' > ')[-1]
  temp = temp.replace('.', ' ')
  return temp

def press_enter():
    keyboard = Controller()
    keyboard.press(Key.enter)
    keyboard.release(Key.enter)


def plot(results, name):
    # plot value on y-axis and date on x-axis
    fig = px.line(results, x=results.index, y=name, title=name, template='plotly_dark')
    # create list of outlier_dates
    outlier_dates = results[results['Anomaly'] == 1].index
    # obtain y value of anomalies to plot
    y_values = [results.loc[i][name] for i in outlier_dates]
    df = pd.DataFrame()
    fig.add_trace(go.Scatter(x=outlier_dates, y=y_values, mode='markers',
                             name='Anomaly',
                             marker=dict(color='red', size=10)))
    st.plotly_chart(fig)
    st.write(y_values)


def iforest_ad(past, new,name, f):
    data = pd.concat([past, new])
    data = pd.DataFrame(data)
    # creature features from date
    data['day'] = [i.day for i in data.index]
    data['day_name'] = [i.day_name() for i in data.index]
    data['day_of_year'] = [i.dayofyear for i in data.index]
    data['week_of_year'] = [i.weekofyear for i in data.index]
    data['hour'] = [i.hour for i in data.index]
    data['is_weekday'] = [i.isoweekday() for i in data.index]
    s = setup(data, session_id=None, silent=True)

    #LOF
    iforest = create_model('iforest',fraction=f)
    iforest_results = assign_model(iforest)
    st.subheader('Isolation Forests')
    plot(iforest_results[-1200:], name)
    st.subheader('HBOS (Global) AD')
    #HBOS
    hist = create_model('histogram', fraction=f)
    hist_results = assign_model(hist)
    hist_results.head()
    plot(hist_results[-1200:], name)

def main(name = 'Fin Tube Radiation Valve Position' ):
    ##format
    d = data.Data()
    df = d.get_og_data()
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    df.columns = [format_names(col) for col in df.columns]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    norm_df = pd.DataFrame(data=scaled, index=df.index, columns=df.columns)
    st.title("Monitoring Air Flow Anomalies")
    st.subheader('Input Data')
    st.dataframe(df)
    PATH = r'/Users/shxryz/aps490/output/model.pth'
    model = torch.load(PATH, map_location=torch.device('cpu'))
    model.eval()
    last48 = np.array(norm_df[-24 - 48:-24]).reshape(1, 48, 5)
    x = torch.from_numpy(last48).float()
    with torch.no_grad():
        predict = model(x)
    pred = np.array(predict).reshape((24,))
    pred = d.recover_y(pred)
    original = df[-24:][name]
    past = df[:-24][name]
    new = pd.Series(pred, name=name, index=df[-24:].index)
    # create plots
    st.subheader('Original Data Vs Predictions')
    sns.set_style("darkgrid")
    sns.set_palette("tab10")
    fig1 = plt.figure()
    past[-240:].plot(figsize=(12, 6))
    original.plot(label='original', color='blue')
    plt.title('Original' + name)
    st.pyplot(fig1)
    fig2 = plt.figure()
    past[-240:].plot(figsize=(12, 6))
    new.plot(label='new')
    plt.title('LSTM Predicted' + name)
    st.pyplot(fig2)
    #
    f = st.number_input("Contamination Fraction", min_value=0.00, max_value=0.99, value=0.01, step=0.005)
    iforest_ad(past,new, name, f)




if __name__ == "__main__":
    main()