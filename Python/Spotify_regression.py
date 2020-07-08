# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 15:16:23 2020

@author: vikas
"""
#Multi-linear regression for Spotify Data to predict Popoularity

import pandas as pd
import numpy as np

data=pd.read_csv("C:\\Users\\vikas\\Desktop\\Vibhav\\Spotify_with_ReleaseDate.csv")

#Encoding

#Artist As a class
artist_name=pd.get_dummies(data['artist'], columns='artist', prefix='artist', drop_first=True)
data=pd.concat([data, artist_name], axis=1)
data.drop(['artist'],axis=1,inplace= True)

data.info()

#split to X and Y
'''X has all the independent variables
Y has the dependent variable'''

X = data.loc[:, data.columns!='popularity'].values  #independent columns
y =data['popularity'].values

y=y.reshape(-1,1)

X=data.drop(['Unnamed: 0','album', 'track_number','id','uri','release_date','name'], axis=1)

print('Number of samples:', X.shape[0])
print('Number of features:', X.shape[1])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)

y_test_sc = sc_y.fit_transform(y_test)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
print(y_pred)

from sklearn.metrics import r2_score

print(r2_score(y_test_sc, y_pred))

#inverse transform
y_pred_invtrans=sc_y.inverse_transform(y_pred)
print(r2_score(y_test, y_pred_invtrans))

#compare both 
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred_invtrans.flatten()})
#coef_ : array, shape (n_features, ) or (n_targets, n_features)

print(regressor.coef_)

#intercept_ : array  Independent term in the linear model.

print(regressor.intercept_)