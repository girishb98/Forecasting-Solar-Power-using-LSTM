#data preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing training set
dataset = pd.read_csv('03-File1.csv')
dataset_power_zero = dataset[dataset['Date']=='0']
dataset = dataset.drop(dataset_power_zero.index, axis=0)
values = dataset.iloc[:,2:6].values
values = values.astype('float32')

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
scaled=sc.fit_transform(values)
sc_predict=MinMaxScaler(feature_range=(0,1))
sc_predict.fit_transform(values[:,3:4])


X_train1=[]
y_train1=[]
for i in range(160,11919):
    X_train1.append(scaled[i-160:i,:])
    y_train1.append(scaled[i,-1])
X_train1,y_train1 = np.array(X_train1),np.array(y_train1)

#reshaping
X_train1 = np.reshape(X_train1, (X_train1.shape[0],X_train1.shape[1],4))

#splitting dataset into training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train1, y_train1, test_size = 0.2, random_state = 0)

#building RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#initializing RNN
regressor = Sequential()

#adding first LSTM layer and Dropout
regressor.add(LSTM(units =50 ,return_sequences = True, input_shape=(X_train1.shape[1],4)))
regressor.add(Dropout(rate= 0.2))

#adding second LSTM layer and Dropout
regressor.add(LSTM(units =50 ,return_sequences = True))
regressor.add(Dropout(rate= 0.2))

#adding third LSTM layer and Dropout
regressor.add(LSTM(units =50 ,return_sequences = True))
regressor.add(Dropout(rate= 0.2))

#adding forth LSTM layer and Dropout
regressor.add(LSTM(units =50))
regressor.add(Dropout(rate= 0.2))

#adding output layer
regressor.add(Dense(units=1,activation = 'linear'))

#compiling RNN
regressor.compile(optimizer = 'rmsprop' , loss = 'mean_squared_error')

#fit RNN to trainig set
regressor.fit(X_train1 , y_train1 , epochs = 40 , batch_size = 64)

y_pred = regressor.predict(X_test)
y_pred = sc_predict.inverse_transform(y_pred)
y_test = y_test.reshape(-1,1)
y_test = sc_predict.inverse_transform(y_test)

#visualizing
plt.plot(y_test , color = 'red' , label = "Actual power")
plt.plot(y_pred , color = 'blue' , label = "predicted power")
plt.title("Solar power prediction")
plt.ylabel("Power")
plt.xlabel("time")
plt.legend()
plt.show()