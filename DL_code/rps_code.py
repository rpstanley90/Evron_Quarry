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
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten

from keras.optimizers import Adam


import argparse

from sklearn.metrics import mean_squared_error

import helpers
import conductor

from models import standard_predict

def make_plot(y,yhat,title='None'):
    fig,ax = plt.subplots()
    plt.scatter(y, yhat)
    plt.axis('equal')
    plt.xlabel('True Value')
    plt.ylabel('Predicted Value')
    plt.title(f'{title}')
    plt.grid()
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    
    # now plot both limits against eachother
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    plt.show()
    
    

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

data_set_processed_OHE = pd.get_dummies(data_set_processed.Site)
data_set_processed = pd.concat([data_set_processed, data_set_processed_OHE], axis=1).drop('Site', axis=1)

cols = data_set_processed.columns.tolist()
# Indexing & Slicing Techniques
#cols = cols[-10:] + cols[:-10]
#data_set_processed = data_set_processed[cols]

print('First 15 spectra')
data_set_processed.head(n=15)

# Printing the last 15 rows.
print('Last 15 spectra')
data_set_processed.tail(n=15)

"""## Getting the features, X (intensities) and target, y (temperatrues)"""

X = data_set_processed.iloc[:, 1:].to_numpy()
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

X_train, X_rem, y_train, y_rem = train_test_split(X, y_s, shuffle=True, test_size=0.2,
                                                    random_state=RND_SEED)

X_test, X_val, y_test, y_val = train_test_split(X_rem, y_rem, shuffle=True, test_size=0.5,
                                                    random_state=RND_SEED)


print('Training NN...')


#conductor.fit(X_train, y_train)

X_train = np.expand_dims(X_train,2)
X_train = X_train.reshape(X_train.shape[0],X_train.shape[2],X_train.shape[1])
X_test = np.expand_dims(X_test,2)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[2],X_test.shape[1])
X_val = np.expand_dims(X_val,2)
X_val = X_val.reshape(X_val.shape[0],X_val.shape[2],X_val.shape[1])


conductor.fit(X_train,y_train,X_val,y_val)
model=conductor.model

# model = Sequential()
# model.add(Conv1D(filters=16, kernel_size=41, strides=1, padding='valid',
#                  input_shape=(X_train.shape[1], X_train.shape[2],),
#                  activation='relu', name='Conv1_plus_ReLU'))
# model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid', name='MaxPool1'))
# model.add(Conv1D(filters=16, kernel_size=41, strides=1, padding='valid', activation='relu', name='Conv2_plus_ReLU'))
# model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid', name='MaxPool2'))
# model.add(Conv1D(filters=32, kernel_size=41, strides=1, padding='valid', activation='relu', name='Conv3_plus_ReLU'))
# model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid', name='MaxPool3'))
# model.add(Flatten(name='Flatten'))
# model.add(Dense(256, activation='relu', name='FC1_plus_ReLU'))
# model.add(Dense(256, activation='relu', name='FC2_plus_ReLU'))
# model.add(Dense(1, name='Temperature_Output'))

# optimizer = Adam(lr=0.001)
# model.compile(optimizer=optimizer, loss='mean_squared_error')
# model.summary()

#pred_train = model.predict(X_train) 
#pred_test = model.predict(X_test)

pred_train = standard_predict(model, X_train)[0].reshape(-1,1)
pred_test = standard_predict(model, X_test)[0].reshape(-1,1)
pred_val = standard_predict(model, X_val)[0].reshape(-1,1)

# scaling back the temperatures to the original values
pred_traininv = scaler.inverse_transform(pred_train)
train_inv = scaler.inverse_transform(y_train)
pred_testinv = scaler.inverse_transform(pred_test)
test_inv = scaler.inverse_transform(y_test)
pred_valinv = scaler.inverse_transform(pred_val)
val_inv = scaler.inverse_transform(y_val)

ave_temp(pred_traininv, train_inv, pred_testinv, test_inv)


make_plot(train_inv,pred_traininv,title='Train')
make_plot(test_inv,pred_testinv,title='Test')
make_plot(val_inv,pred_valinv,title='Validation')

# Calculating mse and rmse for train
mse_train = mean_squared_error(train_inv, pred_traininv)
print('RMSE - train = ', np.sqrt(mse_train))

# Calculating mse and rmse for test
mse_test = mean_squared_error(test_inv, pred_testinv)
print('RMSE - test = ', np.sqrt(mse_test))

# Calculating mse and rmse for val
mse_val = mean_squared_error(val_inv, pred_valinv)
print('RMSE - val = ', np.sqrt(mse_val))