# -*- coding: utf-8 -*-
"""
Created on Sat May 26 01:23:20 2018

@author: devyash
"""
#data preprocessing

#importing libraries
import numpy as np  #for mathematics
import matplotlib.pyplot as plt  #plots charts etc
import pandas as pd  #to import datasets

#importing dataset
dataset=pd.read_csv('Data.csv')
X= dataset.iloc[:,:-1].values  #independent value vector i.e. matrix of features
Y= dataset.iloc[:,3].values  #dependent value vector

#handing missing data
from sklearn.preprocessing import Imputer  #sklearn contains many libraries and libraries contin classes
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(X[:,1:3])  #3 is exclusive
X[:,1:3]=imputer.transform(X[:,1:3])   

# encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])  #1st columnn of X now has encoded values
onehotencoder=OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()
labelencoder_Y=LabelEncoder()
Y=labelencoder_Y.fit_transform(Y)

#splitting dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)