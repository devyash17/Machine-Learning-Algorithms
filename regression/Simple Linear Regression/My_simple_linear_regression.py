# -*- coding: utf-8 -*-
"""
Created on Sat May 26 20:31:16 2018

@author: devyash
"""

# Simple Linear regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Simple Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()   # we created a macine first
regressor.fit(X_train,y_train)  # machine learnt on training set in this step

# Predicting the Test Set results
y_pred=regressor.predict(X_test)

# Visualising the Training Set results
plt.scatter(X_train,y_train,c='red')
plt.plot(X_train,regressor.predict(X_train),c='blue')  # y coordinate will be predicted salaries
plt.title('Salary vs Experience(Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test Set results
plt.scatter(X_test,y_test,c='red')
plt.plot(X_train,regressor.predict(X_train),c='blue')  # y coordinate will be predicted salaries
plt.title('Salary vs Experience(Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()