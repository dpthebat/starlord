import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import streamlit as st
import tensorflow as tf
import keras
import numpy 

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

# train the LSTM MODEL

model = keras.models.load_model('dprock.keras')


# testinng part 

past_100_days = data_traning.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True,verify_itegrity=False ,sort = False )
final = final_df
input_data = scaler.fit_transform(final)



x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test,y_test = np.array(x_test),np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted*scale_factor
y_test = y_test*scale_factor

#final graph

st.subheader('prediction vs original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b',label = 'original price')
plt.plot(y_predicted,'r',label = 'predicted price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)


