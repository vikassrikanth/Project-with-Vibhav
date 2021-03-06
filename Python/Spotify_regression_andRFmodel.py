# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 15:16:23 2020

@author: vikas
"""
#Multi-linear regression for Spotify Data to predict Popoularity

import pandas as pd
import numpy as np

data=pd.read_csv("C:\\Users\\vikas\\Desktop\\Vibhav\\Spotify_with_ReleaseDate.csv")

print(data['artist'].unique())
#Encoding

#Artist As a class
artist_name=pd.get_dummies(data['artist'], columns='artist', prefix='artist', drop_first=True)
data=pd.concat([data, artist_name], axis=1)
data.drop(['artist'],axis=1,inplace= True)

data.info()

#split to X and Y
'''X has all the independent variables
Y has the dependent variable'''
final_data=data.drop(['Unnamed: 0','album', 'track_number','id','uri','release_date','name','time_signature','length'], axis=1)

X = final_data.loc[:, final_data.columns!='popularity'].values  #independent columns
y =final_data['popularity'].values

y=y.reshape(-1,1) #to make it (x,1)


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

import seaborn as sns
#distribution of residuals using 
#distplot - Seaborn distplot lets you show a histogram with a line on it.
#The y-axis in a density plot is the probability density function - probability density is the probability per unit on the x-axis.
#The residuals are always actual minus predicted. 
sns.distplot(y_test-y_pred_invtrans)

#Scatter plot for true and predicted value - should be linear

import matplotlib.pyplot as plt
%matplotlib inline
plt.scatter(y_test_sc,y_pred)

#Random Forest
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import RandomizedSearchCV
#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)



# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='explained_variance', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)

rf_random.fit(X_train,y_train)


rf_random.best_score_

predictions=rf_random.predict(X_test)


sns.distplot(y_test-predictions)
#compare both 
compare_rf = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': predictions.flatten()})

print(r2_score(y_test, predictions))

from sklearn.metrics import mean_squared_error
import math
math.sqrt(mean_squared_error(y_test_sc, predictions))
#import sklearn
#sorted(sklearn.metrics.SCORERS.keys())


from prettytable import PrettyTable

    
x = PrettyTable()
x.field_names = ["Model","Train R2", "Test R2", "Mean"]
x.add_row(["Kneighbors Classifier",0.5833,0.462 ]) 
x.add_row(["Softmax Regression - Mutinomial Regression",0.98266, 0.962])