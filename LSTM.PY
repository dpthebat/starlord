import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import streamlit as st
import tensorflow as tf 
st.title('stock forecasting')
df = web.DataReader('^DJI','stooq')


#describing data of 1 year
st.subheader('data from 2019-2020')
st.write(df.describe())

#graph of the given data
st.subheader('closing price vs time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Open)
st.pyplot(fig)

#100 day moving averages
st.subheader('100 day moving averages')
m100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(m100,'g')
plt.plot(df.Close,'r')
st.pyplot(fig)

# LSTM MODEL
data_traning = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
# min max scaling of data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_traning_array = scaler.fit_transform(data_traning)
print(data_traning_array)
# spliting data into x_train and y_train\0jbn nm\;;
x_train = []
y_train = []

for i in range(100, data_traning_array.shape[0]):
    x_train.append(data_traning_array[i-100:i])
    y_train.append(data_traning_array[i,0])
    
x_train = np.array(x_train)
y_train = np.array(y_train)

# train the LSTM MODEL

from  keras.layers import Dense,Dropout,LSTM
from keras.models import Sequential

model = Sequential()
model.add(LSTM(units=50,activation='relu',return_sequences=True,input_shape = (x_train.shape[1],1)))
model.add(Dropout(0.2))


model.add(LSTM(units=60,activation='relu',return_sequences=True))
model.add(Dropout(0.3))  


model.add(LSTM(units=80,activation='relu',return_sequences=True))
model.add(Dropout(0.4))  
 
model.add(Dense(units=1))
model.summary()
model.compile(optimizer='adam',loss = 'mean_squared_error')
model.fit(x_train,y_train,epochs=50)
model.save('dprock.keras')




