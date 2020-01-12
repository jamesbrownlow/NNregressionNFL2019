# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 11:03:30 2020

@author: Computer
"""
import mystic as ms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.backend import clear_session
import os;

path='C:/Users/Computer/Documents/NFL'
#path = 'F:/NFL'
#path = 'C:/Users/Dr. J.DrJ-HP/Documents/Python Scripts/NFL'
os.chdir(path)
os.getcwd()
#Variables in the data set are 

# idea is to predict the value of a potential car sale 
# (i.e. how much a particular person will spend on buying a car) 
# for a customer based on the fol attributes:
# age, gender,	miles,	debt,	income
# the response variable is sales

#dataset=np.loadtxt("NFL2019.csv", delimiter=",", skiprows=1)
dataNFL2019 = pd.read_csv('NFL2019.csv')
dataNFL2019.columns


inputArray = np.array([dataNFL2019.Pass, dataNFL2019.att, 
                      dataNFL2019.ydsPerG, dataNFL2019.to,
                      dataNFL2019.yds, dataNFL2019.int,
                      dataNFL2019.sck,dataNFL2019.ydsSack,
                      
                      dataNFL2019.takeInt, dataNFL2019.Def,  # added
                      dataNFL2019.RushYdsPerG,dataNFL2019.fgm,
                      
                      dataNFL2019.firstD, dataNFL2019.firstDpass,
                      dataNFL2019.rushAtt, dataNFL2019.rushYds,
                      dataNFL2019.ydsPerG, dataNFL2019.FumbRec]).T

inputDF = pd.DataFrame(inputArray)



outputArray = np.array([dataNFL2019.Win, dataNFL2019.Loss]).T

outputDF = pd.DataFrame(outputArray)


x = inputDF
y = outputDF


# Data has to be scaled for NN

scaler_x = MinMaxScaler()
scaler_x.fit(x)

scaler_y = MinMaxScaler()
scaler_y.fit(y)

xscale=scaler_x.transform(x)
print(xscale)

yscale=scaler_y.transform(y)
print(yscale)


train_size = 0.67  # what fraction of the data for training
epochs = 100  # iterations of estimate
randomSeed= int(7)

X_train, X_test, y_train, y_test = train_test_split(xscale, yscale,
                    train_size = train_size,  random_state=randomSeed)

# 2 hidden layers
hidden1 = len(X_train)
hidden2 = int(0.7*np.ceil(hidden1))

input_dim = len(X_train[0,:])
outputLayer = len(yscale[0,])


clear_session()
model = Sequential()  # set up the neural net model

# number of input nodes = dimension of inut
model.add(Dense(hidden1, input_dim=input_dim, kernel_initializer='normal', 
                activation='relu'))

# first hidden layer: arount 70% or input vector length  
model.add(Dense(hidden2, activation='relu'))

# output layer rectifier-- choose one

#model.add(Dense(outputLayer, activation='relu'))
#model.add(Dense(outputLayer, activation='linear'))
model.add(Dense(outputLayer, activation= 'softmax'))


model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])

# run the model: 
history = model.fit(X_train, y_train, epochs=epochs, batch_size=32,  
                    verbose=1, validation_split=0.2)

print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()


'''
NE	15	362	591	3744	6.3	249.6	23	8	88	59	27	188	189	0.32	12	3	0	12	4	420	1568	28	104.5	16	114	26	33	79.00%	51	1	11	8	5	1	36	42	174	11.6	797	552	245	45	314	25	3	15	11	2	86	0	1	25	537	21.5	0	34	0	28	226	8.1	0	22	24	23	25	11	8	5	174	11.6	797	552	245	45	314	25	3	15	11	2	86	0	1	320	102	189	84	218	0.39	6	15	0.4	88	774
TEN	15	283	427	3360	7.9	224	27	8	107	91	56	374	166	0.39	8	7	0	9	7	406	1978	27.1	131.9	18	69	8	18	44.00%	53	0	0	2	4	2	45	47	313	20.9	944	615	329	39	243	13	1	11	9	2	68	0	1	28	589	21	0	45	0	20	160	8	0	21	27	5	13	9	8	9	313	20.9	944	615	329	39	243	13	1	11	9	2	68	0	1	292	91	166	65	175	0.37	4	14	0.29	94	895

Pass, yds, int, sck, ydsSack, firstD, firstDpass, rushAtt, rushYds,
                      zeroThru19, twntyThru29]
'''

Xnew = inputDF
Xnew= scaler_x.transform(Xnew)
ynew= model.predict(Xnew)
#invert normalize
ynew = scaler_y.inverse_transform(ynew)

yPrint = a = [(round(2*y[0])/2, round(2*y[1])/2) for y in ynew]
    
Xnew = scaler_x.inverse_transform(Xnew)

for i in range(len(dataNFL2019.team)):
    print("\n{} => \tPredicted \tW = {}, \tL = {}".format(dataNFL2019.team[i],
                   yPrint[i][0],yPrint[i][1]))
    print('       \tActual    \tW = {}, \tL = {}'.format(dataNFL2019.Win[i],
                                               dataNFL2019.Loss[i]))

    