#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 21:00:04 2020

@author: jagveer
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Read csv file
df = pd.read_csv("Expander_data.csv")
df.info()
df=df.dropna(axis=1,how='all')

features=df.iloc[:,[1,2,3,4,5,7]]
lable=df.iloc[:,6]
#df.iloc[:,5].value_counts()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, lable, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
features_train = sc.fit_transform(features_train)
features_test = sc.transform(features_test)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(features_train, labels_train)

#To see the value of the intercept and slop calculated by the linear regression algorithm for our dataset, execute the following code.
print(regressor.intercept_)  
print (regressor.coef_)


Pred = regressor.predict(features_test)

print (pd.DataFrame(Pred, labels_test))



# Getting Score for the Multi Linear Reg model
Score = regressor.score(features_train, labels_train)
Score = regressor.score(features_test, labels_test)

from sklearn import metrics  

#print('Mean Absolute Error:', metrics.mean_absolute_error(labels_test, Pred))  
#print('Mean Squared Error:', metrics.mean_squared_error(labels_test,Pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(labels_test, Pred)))

print("Mean value is: ", np.mean(lable))
print("Standard Deviation is: ", np.std(lable))

#10 % of the mean value of the label is greater than to the RMSE
#Bad assumptions: We made the assumption that this data has a linear relationship, but that might not be the case. Visualizing the data may help you determine that.
#Poor features: The features we used may not have had a high enough correlation to the values we were trying to predict.



from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()
regressor.fit(features_train, labels_train)

labels_pred = regressor.predict(features_test)








