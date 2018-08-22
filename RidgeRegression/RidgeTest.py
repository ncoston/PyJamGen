## TO DO:
### See if I actually need all these imports 
### Other helpful diagnostics?

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
from sklearn.metrics import mean_squared_error, r2_score

Ridge_model, spotify_covariates_test,spotify_danceability_test = RidgeMain()

#This function tests the linear regression on the testing data
def RidgeTest(model,x_test,y_test):
    danceability_pred_Ridge = model.predict(x_test)
    print("Coefficients: {}".format(model.coef_))
    print("Mean Squared Error: {}".format(mean_squared_error(danceability_pred_Ridge, y_test)))
    print('R-squared variance explained: {}'.format(r2_score(danceability_pred_Ridge, y_test)))
    return danceability_pred_Ridge

RidgeTest(Ridge_model, spotify_covariates_test,spotify_danceability_test)