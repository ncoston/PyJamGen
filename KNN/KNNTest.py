## TO DO:
### Other helpful diagnostics?



#Packages I need to perform linear regression
#import numpy as np
#import pandas as pd
#import scipy as sp 
from KNNTrain import *
import sys
sys.path.insert(0, 'C:\LearningPython\PyJamGen\Database')
from Split import *
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

KNN_model, spotify_covariates_test,spotify_danceability_test = KNNMain()

#This function tests the linear regression on the testing data
def KNNTest(model,x_test,y_test):
    danceability_pred_KNN = model.predict(x_test)
    #print("Coefficients: {}".format(model.coef_))
    print("Mean Squared Error: {}".format(mean_squared_error(danceability_pred_KNN, y_test)))
    print('R-squared variance explained: {}'.format(r2_score(danceability_pred_KNN, y_test)))
    return danceability_pred_KNN

KNNTest(KNN_model, spotify_covariates_test,spotify_danceability_test)