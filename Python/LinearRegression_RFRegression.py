# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 15:16:23 2020

@author: vikas
"""
#Multi-linear regression for Spotify Data to predict Popoularity

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import math

#read the data
data=pd.read_csv("C:\\Users\\vikas\\Desktop\\Vibhav\\Spotify_with_ReleaseDate.csv")

print(data['artist'].unique())
#Encoding

#Artist As a class - create dummy variables
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

#checking the shape of X - [0] indicates rows and [1] indicates columns
print('Number of samples:', X.shape[0])
print('Number of features:', X.shape[1])

# Splitting the dataset into the Training set and Test set - test size 20% of the values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#scaling technique used is - standard scaler
#what it does?
#It transforms the data in such a manner that it has mean as 0 and standard deviation as 1. In short it is standardizing the data

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)

y_test_sc = sc_y.fit_transform(y_test)

#Linear Regression
#from sklearn.linear_model module, import the LinearRegression package
from sklearn.linear_model import LinearRegression
#create a local variable
regressor = LinearRegression()
regressor.fit(X_train, y_train)


#To see what coefficients our regression model has chosen, execute the following script:
coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])  
coeff_df


# Predicting the Test set results
y_pred = regressor.predict(X_test)
print(y_pred)

from sklearn.metrics import r2_score
#inverse transform
y_pred_invtrans=sc_y.inverse_transform(y_pred)
print(r2_score(y_test,y_pred_invtrans))


print(regressor.score(X_train, y_train))


#print(r2_score(y_test_sc, y_pred))



#compare both 
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred_invtrans.flatten()})
#coef_ : array, shape (n_features, ) or (n_targets, n_features)
df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

print(regressor.coef_)

#intercept_ : array  Independent term in the linear model.

print(regressor.intercept_)

from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
#Lets print the MAE, MSE and RMSE
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred_invtrans))  
print('Mean Squared Error:', mean_squared_error(y_test, y_pred_invtrans))  
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred_invtrans)))

'''
Interpretation of Mean Absolute Error (MAE) -

MAE=10, implies that, on average, the predictions's distance from the true value is 10 (e.g true value 
of populatity is 80 and prediction is 70 or 90... there would be a distance of 10).
'''

print(data['popularity'].describe())

'''
You can see that the value of root mean squared error is 12.8, which is greater than 10% of the mean value which is 49.66.
This means that our algorithm was not very accurate but can still make reasonably good predictions.


There are many factors that may have contributed to this inaccuracy, for example :
Need more data: 
    We need to have a huge amount of data to get the best possible prediction.
Bad assumptions:
    We made the assumption that this data has a linear relationship, but that might not be the case.
    Visualizing the data may help you determine that.
Poor features:
    The features we used may not have had a high enough correlation to the values we were trying to predict.
'''
import seaborn as sns
#distribution of residuals using 
#distplot - Seaborn distplot lets you show a histogram with a line on it.
#The y-axis in a density plot is the probability density function - probability density is the probability per unit on the x-axis.
#The residuals are always actual minus predicted. 
sns.distplot(y_test-y_pred_invtrans)

#Scatter plot for true and predicted value - should be linear
plt.scatter(y_test_sc,y_pred)


#There is a way to check regression results like R - use statsmodel package - example -
import statsmodels.api as sm
#from statsmodels.sandbox.regression.predstd import wls_prediction_std
model1=sm.OLS(y,X)

result=model1.fit()

print(result.summary())


###################################################################################################################
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

from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(y_test_sc, predictions))
#import sklearn
#sorted(sklearn.metrics.SCORERS.keys())


'''
Interpretation of Mean Absolute Error (MAE) -

MAE=10, implies that, on average, the predictions's distance from the true value is 10 (e.g true value 
of populatity is 80 and prediction is 70 or 90... there would be a distance of 10).
'''

from prettytable import PrettyTable

    
x = PrettyTable()
x.field_names = ["Model","R2"]
x.add_row(["Linear Regression",0.88]) 
x.add_row(["Random forest regressor",0.94])
print(x)
