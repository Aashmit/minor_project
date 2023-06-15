#importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st

#title screen, user input
st.title('Stock Trend Prediction')
user_input=st.text_input('Enter Stock Ticker','AAPL')
df = yf.download(user_input, start='2013-01-01',  end='2023-01-01', progress=False,)

#Data description
st.subheader("Data from 2013 - 2022")
st.write(df.describe())

st.subheader("Closing Price vs Time Chart")
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader("Closing Price vs Time Chart with 200MA and 100 MA")
ma200=df.Close.rolling(200).mean()#moving average for 200 days
ma100=df.Close.rolling(100).mean()#moving average for 100 days
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close,'b',label='Closing Price')
plt.plot(ma200,'g',label='Moving Average for 200 days')
plt.plot(ma100,'r',label='Moving Average for 100 days')
plt.legend()
plt.plot(df.Close)
st.pyplot(fig)

#Training and Testing Split
data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
print(data_training.shape)
print(data_testing.shape)

#Preprocessing
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

data_training_array=scaler.fit_transform(data_training)


#Load Model
model=load_model('keras_model.h5')
gru_model=load_model('GRU.h5')

past_100_days=data_training.tail(100)
final_df=past_100_days.append(data_testing,ignore_index=True)

input_data=scaler.fit_transform(final_df)

#Testing Dataset for past 100 days 
x_test=[]
y_test=[]
for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
x_test,y_test=np.array(x_test),np.array(y_test)

y_predicted=model.predict(x_test)
y_predicted2=gru_model.predict(x_test)

scaler=scaler.scale_
scale_factor=1/scaler[0]
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor
y_predicted2=y_predicted2*scale_factor

#Graph plot to compare the models vs the original price
st.subheader("Predictions vs Original")
fig=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price using LSTM')
plt.plot(y_predicted2,'y',label='Predicted Price using GRU')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig)

#error evaluation
import math
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error
def return_error(test,predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    mae=mean_absolute_error(test,predicted)
    mape=mean_absolute_percentage_error(test,predicted)
    return rmse,mae,mape

rmse_LSTM=return_error(y_test,y_predicted)
rmse_GRU=return_error(y_test,y_predicted2)
error_df=pd.DataFrame(columns=('Model','RMSE'))
error_data=[('LSTM',rmse_LSTM[0],rmse_LSTM[1],rmse_LSTM[2]),('GRU',rmse_GRU[0],rmse_GRU[1],rmse_GRU[2])]
error_df=pd.DataFrame(error_data,columns=('Model','RMSE','MAE','MAPE'))
st.subheader("Error Evaluation")
st.write(error_df)
