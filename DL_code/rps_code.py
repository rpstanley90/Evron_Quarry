# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 17:08:51 2022

@author: stanl
"""

import pandas as pd
import numpy as np
#import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt


import argparse

from sklearn.metrics import mean_squared_error

import helpers
import conductor

from models import standard_predict

def ave_temp(pred_traininv, train_inv, pred_testinv, test_inv):
    
    print('Ave train 25 = ', np.average(pred_traininv[train_inv[:,0] == 25][:,0]))
    print('Ave test 25 = ', np.average(pred_testinv[test_inv[:,0] == 25][:,0]))
    
    print('Ave train 275 = ', np.average(pred_traininv[train_inv[:,0] == 275][:,0]))
    print('Ave test 275 = ', np.average(pred_testinv[test_inv[:,0] == 275][:,0]))
    
    print('Ave train 400 = ', np.average(pred_traininv[train_inv[:,0] == 400][:,0]))
    print('Ave test 400 = ', np.average(pred_testinv[test_inv[:,0] == 400][:,0]))
    
    print('Ave train 500 = ', np.average(pred_traininv[train_inv[:,0] == 500][:,0]))
    print('Ave test 500 = ', np.average(pred_testinv[test_inv[:,0] == 500][:,0]))
    
    print('Ave train 600 = ', np.average(pred_traininv[train_inv[:,0] == 600][:,0]))
    print('Ave test 600 = ', np.average(pred_testinv[test_inv[:,0] == 600][:,0]))
    
    print('Ave train 700 = ', np.average(pred_traininv[train_inv[:,0] == 700][:,0]))
    print('Ave test 700 = ', np.average(pred_testinv[test_inv[:,0] == 700][:,0]))
    
    print('Ave train 800 = ', np.average(pred_traininv[train_inv[:,0] == 800][:,0]))
    print('Ave test 800 = ', np.average(pred_testinv[test_inv[:,0] == 800][:,0]))

# Parse input argument
parser = argparse.ArgumentParser()
parser.add_argument('--config_file', dest='config_file',
                    type=str, help='path to configuration file')
args = parser.parse_args()
config_file = args.config_file

# Define number of points to interpolate and number of spectra in that set
NUM_POINTS = 1000
NUM_SPEC = 10
RND_SEED = 97

# Loading configuration settings
config_obj = helpers.Config(config_file)
config = config_obj.dictionary

conductor = conductor.Conductor(config)

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
                                                    random_state=RND_SEED)



print('Training NN...')


conductor.fit(X_train, y_train)

model=conductor.model

#model = Sequential()
#model.add(Dense(1000, input_dim=NUM_POINTS, activation='relu', name='FC1_plus_ReLU'))
#model.add(Dense(800, activation='relu', name='FC2_plus_ReLU'))
#model.add(Dense(1, name='Temperatrue_Output'))
#model.compile(optimizer='adam', loss='mean_squared_error')
#history = model.fit(x = X_train, y = y_train, validation_data = (X_test, y_test), epochs=80, batch_size=24)

#pred_train = model.predict(X_train) 
#pred_test = model.predict(X_test)

pred_train = standard_predict(model, X_train)[0].reshape(-1,1)
pred_test = standard_predict(model, X_test)[0].reshape(-1,1)

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
print('RMSE - test = ', np.sqrt(mse_test))

ave_temp(pred_traininv, train_inv, pred_testinv, test_inv)

# Make a plot of true vs predicted values for train
plt.figure()
plt.scatter(train_inv, pred_traininv)
plt.axis('equal')
plt.xlabel('True Value')
plt.ylabel('Predicted Value')
plt.title('Train')
plt.show()

plt.figure()
plt.scatter(test_inv, pred_testinv)
plt.axis('equal')
plt.xlabel('True Value')
plt.ylabel('Predicted Value')
plt.title('Test')
plt.show()