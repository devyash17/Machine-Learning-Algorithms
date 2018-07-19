# -*- coding: utf-8 -*-
"""
Created on Mon May 28 11:50:18 2018

@author: devyash
"""
# Polynomial Regression
# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,[1]].values  # this needs to be matrix
y = dataset.iloc[:, 2].values   # its a vector

# Splitting the dataset into the Training set and Test set
# not needed as dataset is small
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)
lin_reg2=LinearRegression()
lin_reg2.fit(X_poly,y)

# Visualising the linear regression results
plt.scatter(X,y,c='red')
plt.plot(X,lin_reg.predict(X),c='blue')
plt.title('Truth vs Bluff(Linear Regression)')
plt.xlabel('Position Label')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial regressin results
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape(len(X_grid),1)  # reshaping X_grid vector to the matrix
plt.scatter(X,y,c='red')
plt.plot(X_grid,lin_reg2.predict(poly_reg.fit_transform(X_grid)),c='blue')
plt.title('Truth vs Bluff(Linear Regression)')
plt.xlabel('Position Label')
plt.ylabel('Salary')
plt.show()

#Predicting a new result with Linear Regression
lin_reg.predict(6.5)  # salary for 6.5 level

#Predicting a new result with Polynomial Regression
lin_reg2.predict(poly_reg.fit_transform(6.5))  # salary for 6.5 level
