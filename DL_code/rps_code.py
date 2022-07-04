# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 17:08:51 2022

@author: stanl
"""

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

from sklearn.metrics import mean_squared_error

# Define number of points to interpolate and number of spectra in that set
NUM_POINTS = 1000
NUM_SPEC = 10
RND_SEED = 97


data_set_processed = pd.read_csv('data_set_processed.csv', index_col=False)
data_set_processed = data_set_processed.drop('Unnamed: 0', axis=1)

print('First 15 spectra')
data_set_processed.head(n=15)

# Printing the last 15 rows.
print('Last 15 spectra')
data_set_processed.tail(n=15)

"""## Getting the features, X (intensities) and target, y (temperatrues)"""

X = data_set_processed.iloc[:, 2:].to_numpy()
y = np.array(data_set_processed.Temperature)
y = np.expand_dims(y, 1)

# Printing the shape of X
print('Shape of X: ', X.shape)

# Printing the shape of y
print('Shape of y: ', y.shape)

"""## Scaling the target with "MinMaxScaler"
The target value range is a few order of magnitude larger than the feature values range. So to make the process more robust the target is scaled to be between 0-1.
"""

scaler = MinMaxScaler()
y_s = scaler.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_s, shuffle=True, test_size=0.2,
                                                    random_state=RND_SEED, stratify=y_s)

print('Training NN...')

model = Sequential()
model.add(Dense(1000, input_dim=NUM_POINTS, activation='relu', name='FC1_plus_ReLU'))
model.add(Dense(800, activation='relu', name='FC2_plus_ReLU'))
model.add(Dense(1, name='Temperatrue_Output'))
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(x = X_train, y = y_train, validation_data = (X_test, y_test), epochs=80, batch_size=24)

pred_train = model.predict(X_train) 
pred_test = model.predict(X_test)

# scaling back the temperatures to the original values
pred_traininv = scaler.inverse_transform(pred_train)
train_inv = scaler.inverse_transform(y_train)
pred_testinv = scaler.inverse_transform(pred_test)
test_inv = scaler.inverse_transform(y_test)

# Calculating mse and rmse for train
mse_train = mean_squared_error(train_inv, pred_traininv)
print('RMSE - train = ', np.sqrt(mse_train))

# Calculating mse and rmse for test
mse_test = mean_squared_error(test_inv, pred_testinv)
print('RMSE = test = ', np.sqrt(mse_test))