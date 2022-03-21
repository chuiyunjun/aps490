import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from LSTM_model import *
import seaborn as sns
from pycaret.anomaly import *
import plotly.graph_objects as go
import plotly.express as px
import os
import random

def load_input_data():
  file_name = '/Users/shxryz/aps490/datasets/VAV-1704-Airflow-analysis-v2.csv'
  df = pd.read_csv(file_name)
  date = 'DateTime_x'
  input1 = 'Temp (°C)'
  input2 = 'Modern Niagara > CPPIB > 17th Flr > VAV-1704 > Primary Air.Air Flow'
  input3 = 'Modern Niagara > CPPIB > 17th Flr > VAV-1704 > Zone Air.Temperature'
  input4 = 'Modern Niagara > CPPIB > 17th Flr > VAV-1704 > Fin Tube Radiation.Valve Position'
  input5 = 'Modern Niagara > CPPIB > 17th Flr > VAV-1704 > Primary Air.Air Flow Setpoint'

  input_list = [input1, input2, input3, input4,input5]
  # dataset file name
  df_input = df[input_list]
  df_input = df_input.dropna()
  df_date_input = df[[date, input1, input2, input3, input4,input5]]
  df_date_input = df_date_input.dropna()
  return df_date_input

def format_names(name):
  temp = name.split(' > ')[-1]
  temp = temp.replace('.', ' ')
  return temp


def plot(results):
    # plot value on y-axis and date on x-axis
    fig = px.line(results, x=results.index, y='Primary Air Air Flow', title='Primary Air Flow', template='plotly_dark')
    # create list of outlier_dates
    outlier_dates = results[results['Anomaly'] == 1].index
    # obtain y value of anomalies to plot
    y_values = [results.loc[i]['Primary Air Air Flow'] for i in outlier_dates]
    df = pd.DataFrame()
    fig.add_trace(go.Scatter(x=outlier_dates, y=y_values, mode='markers',
                             name='Anomaly',
                             marker=dict(color='red', size=10)))
    st.plotly_chart(fig)
    st.write(y_values)


def iforest_ad(past, new):
    data = pd.concat([past, new])
    data = pd.DataFrame(data)

    # creature features from date
    data['day'] = [i.day for i in data.index]
    data['day_name'] = [i.day_name() for i in data.index]
    data['day_of_year'] = [i.dayofyear for i in data.index]
    data['week_of_year'] = [i.weekofyear for i in data.index]
    data['hour'] = [i.hour for i in data.index]
    data['is_weekday'] = [i.isoweekday() for i in data.index]
    s = setup(data, session_id=None)

    #isolation forest
    iforest = create_model('iforest')
    iforest_results = assign_model(iforest)
    st.subheader('Isolation Forest (Local) AD')
    plot(iforest_results[-1200:])
    st.subheader('HBOS (Global) AD')
    hist = create_model('histogram', fraction=0.01)
    hist_results = assign_model(hist)
    hist_results.head()
    plot(hist_results[-240:])


def main():
    ##format
    df = load_input_data()
    df['DateTime_x'] = pd.to_datetime(df['DateTime_x'])
    df.set_index('DateTime_x', inplace=True)
    df.columns = [format_names(col) for col in df.columns]
    norm_df=(df-df.min())/(df.max()-df.min())
    st.title("Monitoring Air Flow Anomalies")
    st.subheader('Input Data')
    st.dataframe(df)

    PATH = r'/Users/shxryz/aps490/Models/Airflow_Huber0.015_LSTM.pth'
    model = torch.load(PATH, map_location=torch.device('cpu'))
    model.eval()
    last48 = np.array(norm_df[-24 - 48:-24]).reshape(1, 48, 5)
    x = torch.from_numpy(last48).float()
    with torch.no_grad():
        predict = model(x)
    pred = np.array(predict).reshape((24,))
    original = norm_df[-24:]['Primary Air Air Flow']
    past = norm_df[:-24]['Primary Air Air Flow']
    new = pd.Series(pred.T, name=original.name, index=original.index)
    # create plots
    st.subheader('Original Data Vs Predictions')
    sns.set_style("darkgrid")
    sns.set_palette("tab10")
    fig1 = plt.figure()
    past[-240:].plot(figsize=(12, 6))
    original.plot(label='original', color='blue')
    plt.title('Original Primary Air Flow')
    st.pyplot(fig1)
    fig2 = plt.figure()
    past[-240:].plot(figsize=(12, 6))
    new.plot(label='new')
    plt.title('LSTM Predicted Primary Air Flow')
    st.pyplot(fig2)

    iforest_ad(past,new)




if __name__ == "__main__":
    main()