# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
# optional
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
#x_train.shape[0] give no of rows and x_train.shape[1] will give no of column
#x_train, 1 in the end which make rows in to columns 

#building the rnn

#initialsing the keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#initialize the RNN
regressor = Sequential()

#adding first LSTM layer and some dropout regularisation
regressor.add(LSTM(units = 50, return_sequences= True, input_shape = (X_train.shape[1], 1)))
# return_sequences means we want another neural layer
#X_train.shape[1], 1) means time step and indicator
regressor.add(Dropout(0.2))

#add second rstm layer and dropout
regressor.add(LSTM(units = 50, return_sequences= True))
regressor.add(Dropout(0.2))

#add third rstm layer and dropout
regressor.add(LSTM(units = 50, return_sequences= True))
regressor.add(Dropout(0.2))

#add fourth rstm layer and dropout
regressor.add(LSTM(units = 50, return_sequences= False))
regressor.add(Dropout(0.2))

#output layer
regressor.add(Dense(units=1))

#compiling
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#fitting the rnn to the training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)


#part 3 - making the prediction and visualization

#getting the real stock prices
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

#getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)  
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60 : ].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs) 
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stocked_price = regressor.predict(X_test)
predicted_stocked_price = sc.inverse_transform(predicted_stocked_price)

#visualising the results
plt.plot(real_stock_price, color='yellow', label='real_google_stock_price')
plt.plot(predicted_stocked_price, color = 'red', label = 'precited_price')
plt.title('google_stock_price_prediction')
plt.xlabel('time')
plt.ylabel('Google_stock_price')
plt.legend()
plt.show()

import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price,predicted_stocked_price))