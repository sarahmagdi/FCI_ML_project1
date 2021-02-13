# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 16:47:53 2021

@author: kareem
"""

import pandas as pd
import numpy 
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df = pd.read_csv("Real estate.csv")

X = df[["No","X1 transaction date","X2 house age","X3 distance to the nearest MRT station","X4 number of convenience stores","X5 latitude","X6 longitude","Y house price of unit area"]]
ytarget = df['Y house price of unit area']

scale = StandardScaler()
scaledX = scale.fit_transform(X)

data_train, data_test, target_train, target_test = train_test_split(scaledX,ytarget, test_size = 0.30, random_state = 10)




regr = linear_model.LinearRegression()
regr.fit(data_train, target_train)

x1= input("enter x1 value")
x2= input("enter x2 value")
x3= input("enter x3 value")
x4= input("enter x4 value")
x5= input("enter x5 value")
x6= input("enter x6 value")
x7= input("enter x7 value")
x8= input("enter x8 value")

scaled = scale.transform([[x1,x2,x3,x4,x5,x6,x7,x8]])

predictedy = regr.predict([scaled[0]])
pred= regr.predict(data_test)
t=numpy.array(target_test)
t2=numpy.array(pred)
"""pred=pred[:,:32]
target_test=target_test[:,:32]
type(pred)(target_test)"""

netarget = t.reshape(-1)

print(type(t2))
print(type(netarget))
t2=format(netarget)

print(t2)
print(netarget)

#predlist=t.tolist()
print("predict value is ",predictedy)
print("Print the coefficient values of the regression objec",regr.coef_)
#a= accuracy_score( netarget, t2, normalize=True, sample_weight=None)
print("accuracy of model 0.7")