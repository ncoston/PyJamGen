## TO DO:
### Other helpful diagnostics?
##Comment everything

#Packages I need to perform linear regression
import numpy as np
import pandas as pd
import scipy as sp 
import matplotlib.pyplot as plt
from RidgeTrain import *
import sys
sys.path.insert(0, 'C:\LearningPython\PyJamGen\Database')
from Split import *
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error

#Ridge_model, spotify_covariates_test,spotify_danceability_test = RidgeMain()

#This function tests the linear regression on the testing data
def RidgeTest(model,x_test,y_test):
    danceability_pred_Ridge = model.predict(x_test)
    print("After training and testing the Ridge Regression model, here are some metrics of the model")
    print("Coefficients: {}".format(model.coef_)) #coefficients in model
    print("Mean Squared Error: {}".format(mean_squared_error(y_test, danceability_pred_Ridge))) #close to zero is better
    print('Mean absolute error: {}'.format(mean_absolute_error(y_test, danceability_pred_Ridge))) #error under L1 norm, closer to zero is better
    print('R-squared variance explained: {}'.format(r2_score(y_test, danceability_pred_Ridge))) #Closer to one is better
    print('Explained variance score: {}'.format(explained_variance_score(y_test, danceability_pred_Ridge))) #closer to one is better
    return danceability_pred_Ridge

#RidgeTest(Ridge_model, spotify_covariates_test,spotify_danceability_test)