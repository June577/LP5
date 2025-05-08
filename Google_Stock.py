import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


data = pd.read_csv('/content/GOOGLE Stock Data set.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
closing_prices = data['Close'].values.reshape(-1, 1)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(closing_prices)

def create_dataset(data, time_step=60):
  X,y=[], []
  for i in range(time_step, len(data)):
    X.append(data[i-time_step:i, 0])
    y.append(data[i,0])
  return np.array(X), np.array(y)

time_step = 60
X, y = create_dataset(scaled_data, time_step)
X = X.reshape((X.shape[0], X.shape[1],1))

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

model.fit(X,y, epochs = 5, batch_size=32)

predicted = model.predict(X)
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(y.reshape(-1,1))

plt.figure(figsize=(10,6))
plt.plot(actual_prices, color='blue', label = 'Actual')
plt.plot(predicted_prices, color='red', label = 'Predicted')
plt.title('Google')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()