import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# Import Statsmodels
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic
from prediction.data import Data


# def adf_test(series,title=''):
#     """
#     Pass in a time series and an optional title, returns an ADF report
#     """
#     print(f'Augmented Dickey-Fuller Test: {title}')
#     result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
#     labels = ['ADF test statistic','p-value','# lags used','# observations']
#     out = pd.Series(result[0:4],index=labels)
#     for key,val in result[4].items():
#         out[f'critical value ({key})']=val
#     print(out.to_string())          # .to_string() removes the line "dtype: float64"
#     if result[1] <= 0.05:
#         print("Strong evidence against the null hypothesis")
#         print("Reject the null hypothesis")
#         print("Data has no unit root and is stationary")
#     else:
#         print("Weak evidence against the null hypothesis")
#         print("Fail to reject the null hypothesis")
#         print("Data has a unit root and is non-stationary")

# def format_names(name):
#   temp = name.split(' > ')[-1]
#   temp = temp.replace('.', ' ')
#   return temp

# df_input.columns = [format_names(col) for col in df_input.columns]

# for col in df_input.columns:
#   adf_test(df_input[col])
  
# def cointegration_test(df, alpha=0.05): 
#     """Perform Johanson's Cointegration Test and Report Summary"""
#     out = coint_johansen(df,-1,5)
#     d = {'0.90':0, '0.95':1, '0.99':2}
#     traces = out.lr1
#     cvts = out.cvt[:, d[str(1-alpha)]]
#     def adjust(val, length= 6): return str(val).ljust(length)

#     # Summary
#     print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
#     for col, trace, cvt in zip(df.columns, traces, cvts):
#         print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)

# cointegration_test(df_input)

# sc = MinMaxScaler()
# # data = sc.fit_transform(df_input)
# data=df_input
# data = pd.DataFrame(data, columns=df_input.columns)

####################
datetime = 'DateTime'
input1 = 'Modern Niagara > CPPIB > 17th Flr > VAV-1704 > Primary Air.Air Flow Setpoint'
#second input is our output
input2 = 'Modern Niagara > CPPIB > 17th Flr > VAV-1704 > Fin Tube Radiation.Valve Position'
input3 = 'Modern Niagara > CPPIB > 17th Flr > VAV-1704 > Zone Air.Temperature'
input4 = 'Modern Niagara > CPPIB > 17th Flr > VAV-1704 > Primary Air.Air Flow'
input5 = 'Temp (°C)'

INPUT_LIST_V = [datetime, input1, input2, input3, input4, input5]
INPUT_LIST_A = [datetime, input1, input4, input3, input2, input5]

train = Data(option='A', data_root='./prediction/datasets/train/').get_raw_data()
test = Data(option='A', data_root='./prediction/datasets/valid/').get_raw_data()

def format_names(name):
  temp = name.split(' > ')[-1]
  temp = temp.replace('.', ' ')
  return temp

train.columns = [format_names(col) for col in train.columns]
test.columns = [format_names(col) for col in test.columns]
print(train.columns)
forecasting_model = VAR(train)
results_aic = []
for p in range(1,10):
  results = forecasting_model.fit(p)
  results_aic.append(results.aic)

import seaborn as sns
sns.set()
plt.plot(list(np.arange(1,10,1)), results_aic)
plt.xlabel("Order")
plt.ylabel("AIC")
plt.show()

results = forecasting_model.fit(6)
results.summary()

sample_index = 100
lagged_values = test[:sample_index].values[-6:]
next = results.forecast(y= lagged_values, steps=24)
next = pd.DataFrame(next, columns=train.columns)
actual_air_flow = test[sample_index:sample_index+24]['Primary Air Air Flow']
predicted = next['Primary Air Air Flow']
predicted.index = actual_air_flow.index
plt.plot(actual_air_flow, label='Actual Valve Position')
plt.plot(predicted, label='Predicted Valve Position')
plt.legend()
plt.title('Baseline Forecast of Valve Position')
final = pd.DataFrame()
final['Actual Valve Position'] = actual_air_flow 
final['Predicted Valve Position'] = predicted 
path = r'./new_baseline_results_model6.csv'
final.to_csv(path)


mae_list = []
mape_list = []
for i in np.arange(start=6,step=24,stop=len(test)-24):
  lagged_values = test[:i].values[-6:]
  next = (results.forecast(y= lagged_values, steps=24))
  next = pd.DataFrame(next, columns=train.columns)
  actual = test[i:i+24]['Primary Air Air Flow']
  forecasted = np.array(next['Primary Air Air Flow'])
  actual = np.array(actual)
  mae = np.mean(np.abs(forecasted - actual))
  mae_list.append(mae)

print('TEST MAE OF BASELINE: ', np.mean(mae_list))
# print('TEST MAPE OF BASELINE (Percent): ', np.mean(mape_list)*100)