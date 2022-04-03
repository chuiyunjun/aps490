import streamlit as st
import torch.nn as nn
from torch.autograd import Variable
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from prediction import data
from prediction.model import *
import seaborn as sns
from pycaret.anomaly import *
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import os
import random


@st.cache(suppress_st_warning=True)
def load_input_data():
    d = data.Data()
    input = d.get_og_data()
    return input


def format_names(name):
  temp = name.split(' > ')[-1]
  temp = temp.replace('.', ' ')
  return temp


def plot(results, name):
    # plot value on y-axis and date on x-axis
    fig = px.line(results, x=results.index, y=name, title=name, template='plotly_dark', width=1500, height=600)
    # create list of outlier_dates
    outlier_dates = results[results['Anomaly'] == 1].index
    # obtain y value of anomalies to plot
    y_values = [results.loc[i][name] for i in outlier_dates]
    df = pd.DataFrame()
    fig.add_trace(go.Scatter(x=outlier_dates, y=y_values, mode='markers',
                             name='Anomaly',
                             marker=dict(color='red', size=10)))
    st.plotly_chart(fig)
    st.write(results[results['Anomaly'] == 1])


@st.cache(suppress_st_warning=True)
def ad(past, new,name, f):
    data = pd.concat([past, new])
    data = pd.DataFrame(data)
    # creature features from date
    data['day'] = [i.day for i in data.index]
    data['hour'] = [i.hour for i in data.index]
    data['is_weekday'] = [i.isoweekday() for i in data.index]
    s = setup(data, session_id=None, silent=True, normalize=True, normalize_method='minmax')
    #IFOREST
    iforest = create_model('iforest',fraction=f)
    iforest_results = assign_model(iforest)
    st.subheader('Isolation Forests')
    plot(iforest_results, name)
    st.subheader('HBOS (Global) AD')
    #HBOS
    hist = create_model('histogram', fraction=f/2)
    hist_results = assign_model(hist)
    hist_results.head()
    plot(hist_results, name)


def generate_block(option):
    ##format
    d = data.Data(data_root='./prediction/datasets/all/',option=option)
    df = d.get_og_data()
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    df.sort_index(inplace=True)
    df.columns = [format_names(col) for col in df.columns]
    name = df.columns[1]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    norm_df = pd.DataFrame(data=scaled, index=df.index, columns=df.columns)
    if option == 'V':
        st.title('Input Data')
        st.dataframe(df)
        st.title("Monitoring Valve Position Anomalies")
    else:
        st.title("Monitoring Air Flow Anomalies")
    p = select_option(option)
    model = torch.load(p, map_location=torch.device('cpu'))
    model.eval()
    last48 = np.array(norm_df[-24 - 48:-24]).reshape(1, 48, 8)
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
    fig1 = plt.figure(figsize=(10,3))
    past[-240:].plot()
    original.plot(label='original', color='blue')
    plt.title('Original ' + name)
    st.pyplot(fig1)
    fig2 = plt.figure(figsize=(10,3))
    past[-240:].plot()
    new.plot(label='new')
    plt.title('GRU Predicted ' + name)
    st.pyplot(fig2)
    #
    f = st.number_input("Contamination Fraction", min_value=0.00, max_value=0.99, value=0.01, step=0.0025, key=name)
    ad(past, new, name, f)


def select_option(option ='V'):
    type = st.selectbox("Select Model", options=('GRU', 'LSTM'), index=0, key=option)
    if type == 'GRU':
        if option =='V':
            p = './output/ValveModel.pth'
        else:
            p = './output/AirFlowModel.pth'
    else:
        # TODO: UPDATE FOR LSTM
        if option =='V':
            p = './output/ValveModel.pth'
        else:
            p = './output/AirFlowModel.pth'
    return p


st.set_page_config(layout='wide')
def main():
    generate_block(option='V')
    generate_block(option='A')


if __name__ == "__main__":
    main()
