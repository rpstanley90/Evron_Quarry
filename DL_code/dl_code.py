# -*- coding: utf-8 -*-
"""DL_code.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GWsWZ1Z5dbkn4jLheE4Dbqp84m9RCGTb

## Important  Code Parts used in: 

## Ref. 1: Estimating temperatures of heated Lower Palaeolithic flint artefacts, A. Agam, I. Azuri, I. Pinkas, A. Gopher, and F. Natalio, Nat Hum Behav (2020)

## Ref. 2: Z. Stepka, I. Azuri, L. Kolska Horwitz, M. Chazan, and F. Natalio, PNAS (2022)

## The code was writen by Ido Azuri
"""

# Improting Python packages
import numpy as np # NumPy package for working with arrays, mathematical operations and linear algebra
import pandas as pd # Pandas package for working with Data Frames
import sklearn # Sklearn package for machine learning
import rampy as rp # Rampy package for spectra preprocessing. Should be installed.
import scipy # SciPy package for using the interp1d interpolation function 
import matplotlib # Matplotlib packagae for plotting and visualization
import matplotlib.pyplot as plt
import keras # Keras package for Deep Learning. Should be installed.
from scipy.interpolate import interp1d

"""### The following package versions were used in that project

NumPy version is:  1.15.4
<br>
Pandas version is:  0.24.2
<br>
Sklearn version is:  0.21.1
<br>
Scipy version is:  1.2.1
<br>
Matplotlib version is:  3.0.3
<br>
Keras version is:  2.2.2
"""

# Printing your current versions
print('NumPy version is: ', np.__version__)
print('Pandas version is: ', pd.__version__)
print('Sklearn version is: ', sklearn.__version__)
print('Scipy version is: ', scipy.__version__)
print('Matplotlib version is: ', matplotlib.__version__)
print('Keras version is: ', keras.__version__)

"""## Preprocessing the spectra

### For preprocessing the spectra the following steps applied:
### 1. Interpolating the spectra
For each set of spectra recording, the wave numbers are a little bit different. For the input to the machine learning/ deep learning models the intensities should be at fixed wave numbers for all spectra from all sets recordings. For this the spectra intensities are interpolated for a fixed given wave numbers. 
### 2. Smoothing
This step is for noise reduction. The number of points that are choosen in step 1 should be not small enough, so most of the information is kept and not too high, so the smoothing will be effective as much as possible.
### 3. Baseline correction
To correct distorted measuremets.
### 4. Intesity normalization
To put the spectra on the same scale.

## Note, here the steps are applied only on one set of measurements ("spec_1p2_control.txt") for demonstration. But the same pipline was applied on all the other spectra and supplied in the running folder as csv file as "data_set_processed.csv"

## 1. Interpolating the spectra
"""

# Reading an example set of spectra records. 10 Spectra.
temperature_control = []        
spec = pd.read_csv('spec_1p2_control.txt', header=None, sep='\t', comment='#', encoding='latin')
temperature_control.append(spec)

# Define number of points to interpolate and number of spectra in that set
Num_Points = 1000
Num_Spec = 10

# Minimum and Maximum wave numbers for the interpolation 
min_point_inter = 102
max_point_inter = 1798

print('Interpolate, with '+str(Num_Points)+' Points...')

Int_control = np.zeros((Num_Spec,Num_Points))

freq_min = []
freq_max = []

# Fucntion for the interpolation and getting more information
def Int_DF(DF):
    DF_array = np.zeros((Num_Spec,Num_Points))
    spec = DF[0]
    spec.drop(0, axis=1, inplace=True)
    freq = spec.iloc[0]
    freq_min.append(min(freq))
    freq_max.append(max(freq))
    # Getting the wave numbers
    x = np.array(freq)
    # Getting the intensities
    y = np.array(spec.iloc[1:])
    # Define the interpolation function, f
    f = interp1d(x, y, kind='linear')
    
    # Define the fixed wave numbers to interpolate
    xnew = np.linspace(min_point_inter, max_point_inter, num=Num_Points, endpoint=True)
    # Getting the interplated intensities
    fnew = f(xnew) 
    DF_array[:,:] = fnew
    return DF_array

interpolate_control = Int_DF(temperature_control)

print('Shape is', interpolate_control.shape, 'for 10 spectra of 1000 intensity wavenumbers')
interpolate_control

"""## Steps 2 - 4"""

# Choosing the smoother
smooth_method = "GCVSmoothedNSpline"
# Choosing the baseline method
baseline_method = "als"
# Choosing the normalization method
norm_method = "intensity"

print('Smoothing...\nBase correction...\nNormalization...')

freq_inter = np.linspace(min_point_inter, max_point_inter, num=Num_Points, endpoint=True)

interpolate_control_smooth = np.zeros((Num_Spec,Num_Points))

interpolate_control_base = np.zeros((Num_Spec,Num_Points))

for k in range(Num_Spec):
    # smoothing
    interpolate_control_smooth[k,:] = rp.smooth(freq_inter, interpolate_control[k,:], method=smooth_method)
    corr_spec, base = rp.baseline(freq_inter, interpolate_control_smooth[k,:], bir = np.array([[min_point_inter, max_point_inter]]), method=baseline_method)
    interpolate_control_base[k,:] = rp.normalise(corr_spec[:,0], method=norm_method)

"""## Printing the spectra"""

arr_plot = interpolate_control
arr_plot_smooth = interpolate_control_smooth
arr_plot_base = interpolate_control_base

for idx_plt in range(arr_plot.shape[0]):
    
    print('Spectrum Number = ', idx_plt + 1)
    
    fig = plt.figure(figsize=[14,5])
    
    plt.subplot(1,2,1)
    plt.plot(freq_inter, arr_plot[idx_plt,:])
    plt.plot(freq_inter, arr_plot_smooth[idx_plt,:])
    plt.xlabel('Wave Number')
    plt.ylabel('Intensity')
    plt.legend(['only interpolated','interpolated + smoothed'])
    
    plt.subplot(1,2,2)
    plt.plot(freq_inter, arr_plot_base[idx_plt,:])
    plt.xlabel('Wave Number')
    plt.ylabel('Normalized Intensity')
    plt.legend(['+ base correction + intensity normalozation'])
    
    plt.show()

"""## Machine Learning and Deep Learning example - Regression case

Here, the processed data set is loaded (data_set_processed.csv).
In this work we used models from Sklearn Python package and artificial neural network models with Keras. Here we demonstrate the artificial neural network cases (FC-ANN and 1D-CNN).

## Fully Connected Artificial Neural Network (FC-ANN)
"""
#%%
data_set_processed = pd.read_csv('data_set_processed.csv', index_col=False)
data_set_processed = data_set_processed.drop('Unnamed: 0', axis=1)

# Printing the first 15 rows. 
# The first column (Temperature) is the corresponding temperature of the flint.
# The second column (Site) is the site of the flint.
# The third column to the end are the intensities for the given wave mumbers in the columns.
# Each row is a spectrum intensities that measured on different position on the flint, or different flint (next cycle of temperatures in the data frame, index location is 614 in data frame)
print('First 15 spectra')
data_set_processed.head(n=15)

# Printing the last 15 rows. 
print('Last 15 spectra')
data_set_processed.tail(n=15)

"""## Getting the features, X (intensities) and target, y (temperatrues)"""

X = data_set_processed.iloc[:,2:].to_numpy()
y = np.array(data_set_processed.Temperature)
y = np.expand_dims(y,1)

# Printing the shape of X
print('Shape of X: ', X.shape)

# Printing the shape of y
print('Shape of y: ', y.shape)

"""## Scaling the target with "MinMaxScaler"
The target value range is a few order of magnitude larger than the feature values range. So to make the process more robust the target is scaled to be between 0-1.
"""

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
y_s = scaler.fit_transform(y)


# Function that will be used later that calculates the average predicted temperatures with respect to target values.
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

"""## Splitting the data set into training and testing sets

### Note, in the original work more validations methods and analyses and models applied. Here, we demonstrate a simple train test split. Such a split is over optimistic, and for better validation strategies and explanations see SI in Ref. 1. 
"""

from sklearn.model_selection import train_test_split

rnd_seed = 97
X_train, X_test, y_train, y_test = train_test_split(X, y_s, shuffle=True, test_size=0.2,
                                                    random_state=rnd_seed, stratify=y_s)

"""## Artificial Neural Network for Regression (FC-ANN). Ref. 1 and 2

### Here, we demonstarte how to build an artificial neural network in Keras, specifically, FC-ANN. Here, we take the final architecture as we used in the manuscript. Note: Different run will give slightly different results due to ramdomness of the ANN weights initialization and other randomness involved in the machine learning pipeline.
"""

from keras.models import Sequential
from keras.layers import Dense

from sklearn.metrics import mean_squared_error

print('Training NN...')

model = Sequential()
model.add(Dense(1000, input_dim=Num_Points, activation='relu', name='FC1_plus_ReLU'))
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

# Make a plot of true vs predicted values for train
plt.scatter(train_inv, pred_traininv)
plt.axis('equal')
plt.xlabel('True Value')
plt.ylabel('Predicted Value')
plt.title('Train')
plt.show()

# Make a plot of true vs predicted values for test
plt.scatter(test_inv, pred_testinv)
plt.axis('equal')
plt.xlabel('True Value')
plt.ylabel('Predicted Value')
plt.title('Test')
plt.show()

# Using the function ave_temp from above to calculate average predicted temperatures with respect to original temperatures
ave_temp(pred_traininv, train_inv, pred_testinv, test_inv)

# Printing loss function of traininig and validation set. See "validation_split" in model.fit function.
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('Loss')
plt.legend(['Train Loss', 'Validation Loss'])
plt.show()

# Printing model summary 
model.summary()

# Calculating Pearson correlation coefficient between true temperature values vs predicted values for training and testing.
corr_coeff_train = np.corrcoef(train_inv[:,0], pred_traininv[:,0])[0,1]
print('Pearson correlation coefficient Train', corr_coeff_train)
corr_coeff_test = np.corrcoef(test_inv[:,0], pred_testinv[:,0])[0,1]
print('Pearson correlation coefficient Test', corr_coeff_test)

#%%
"""## Artificial Neural Network for Regression (1D-CNN). Ref. 2

### Here, we demonstarte how to build an artificial neural network in Keras, specifically, 1D-CNN. Here, we take the final architecture as we used in the manuscript. Note: Different run will give slightly different results due to ramdomness of the ANN weights initialization and other randomness involved in the machine learning pipeline.
"""

data_set_processed = pd.read_csv('data_set_processed.csv', index_col=False)
data_set_processed = data_set_processed.drop('Unnamed: 0', axis=1)

"""## Getting the features, X (intensities) and target, y (temperatrues)"""

X = data_set_processed.iloc[:,2:].to_numpy()
y = np.array(data_set_processed.Temperature)
y = np.expand_dims(y,1)

"""## Taking regions from the spectra that correspond to the peaks regions mentioned in Ref. 2 (highlighted in the color horizontal lines)"""

Num_Points = 1000

rand_spec = np.random.randint(X.shape[0])

print('Chosen Spectra is: ', rand_spec)

fig = plt.figure(dpi=300, frameon=False)
fig.set_size_inches(7,2)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.plot(X[rand_spec,:], color='black')
plt.ylabel('X', fontsize=20)
plt.yticks(ticks=[], labels=[])
plt.xticks(ticks=[], labels=[])
plt.plot(range(0,125), 0.35 * np.ones(125) ,color='purple')
ax.annotate(text='First Region',xy=(50, 0.45), xytext=(0, 0.4))
plt.plot(range(175,275), 1.0 * np.ones(100) ,color='purple')
ax.annotate(text='Second Region',xy=(175, 1.05), xytext=(175, 1.05))
plt.plot(range(625,1000), 0.4 * np.ones(375) ,color='purple')
ax.annotate(text='Third Region',xy=(625, 0.45), xytext=(625, 0.45))
plt.show()

X_1 = X[:,0:125]
X_2 = X[:,175:275]
X_3 = X[:,625:]
X_new = np.concatenate((X_1, X_2, X_3), axis=1)

print('Shape of X: ', X_new.shape)

# Printing random spectra
rand_spec = np.random.randint(0,X_new.shape[0])
print('Chosen Spectra is: ', rand_spec)

plt.plot(X_new[rand_spec,:])
plt.show()

"""## Scaling the target with "MinMaxScaler"
The target value range is a few order of magnitude larger than the feature values range. So to make the process more robust the target is scaled to be between 0-1.
"""

scaler = MinMaxScaler()
y_s = scaler.fit_transform(y)

"""## Splitting the data set into training and testing sets

### Note, in the original work more validations methods and analyses and models applied. Here, we demonstrate a simple train test split.. Such a split is over optimistic, and for better validation strategies and explanations see SI in Ref. 1. 
"""

rnd_seed = 97
X_train, X_test, y_train, y_test = train_test_split(X_new, y_s, shuffle=True, test_size=0.2,
                                                    random_state=rnd_seed, stratify=y_s)

"""## Add a channel to the training data so it fits to the 1D-CNN model requirenments"""

X_train = np.expand_dims(X_train,2)
X_test = np.expand_dims(X_test,2)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import Adam
from keras.layers import ReLU
from sklearn.metrics import mean_squared_error, mean_absolute_error

# The model architecture chosen in Ref. 2 for the 1D-CNN model
model = Sequential()
model.add(Conv1D(filters=16, kernel_size=41, strides=1, padding='valid',
                 input_shape=(X_train.shape[1], X_train.shape[2],),
                 activation='relu', name='Conv1_plus_ReLU'))
model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid', name='MaxPool1'))
model.add(Conv1D(filters=16, kernel_size=41, strides=1, padding='valid', activation='relu', name='Conv2_plus_ReLU'))
model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid', name='MaxPool2'))
model.add(Conv1D(filters=32, kernel_size=41, strides=1, padding='valid', activation='relu', name='Conv3_plus_ReLU'))
model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid', name='MaxPool3'))
model.add(Flatten(name='Flatten'))
model.add(Dense(256, activation='relu', name='FC1_plus_ReLU'))
model.add(Dense(256, activation='relu', name='FC2_plus_ReLU'))
model.add(Dense(1, name='Temperature_Output'))

optimizer = Adam(lr=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')
model.summary()
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
rmse_k_train = np.sqrt(mse_train)
print('RMSE - train = ', rmse_k_train)
# Calculating mse and rmse for test
mse_test = mean_squared_error(test_inv, pred_testinv)
rmse_k_test = np.sqrt(mse_test)
print('RMSE = test = ', rmse_k_test)

# Calculating mae
mae_train = mean_absolute_error(train_inv, pred_traininv)
print('MAE - train = ', mae_train)
# Calculating mae 
mae_test = mean_absolute_error(test_inv, pred_testinv)
print('MAE = test = ', mae_test)   

# Make a plot of true vs predicted values for train
plt.scatter(train_inv, pred_traininv)
plt.axis('equal')
plt.xlabel('True Value')
plt.ylabel('Predicted Value')
plt.title('Train')
plt.show()

# Make a plot of true vs predicted values for test
plt.scatter(test_inv, pred_testinv)
plt.axis('equal')
plt.xlabel('True Value')
plt.ylabel('Predicted Value')
plt.title('Test')
plt.show()

# Using the function ave_temp from above to calculate average predicted temperatures with respect to original temperatures
ave_temp(pred_traininv, train_inv, pred_testinv, test_inv)

# Printing loss function of traininig and validation set. See "validation_split" in model.fit function.
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('Loss')
plt.legend(['Train Loss', 'Validation Loss'])
plt.show()

# Printing model summary 
model.summary()

# Calculating Pearson correlation coefficient between true temperature values vs predicted values for training and testing.
corr_coeff_train = np.corrcoef(train_inv[:,0], pred_traininv[:,0])[0,1]
print('Pearson correlation coefficient Train', corr_coeff_train)
corr_coeff_test = np.corrcoef(test_inv[:,0], pred_testinv[:,0])[0,1]
print('Pearson correlation coefficient Test', corr_coeff_test)