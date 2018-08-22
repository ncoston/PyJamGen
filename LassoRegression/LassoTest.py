## TO DO:
### Other helpful diagnostics?



#Packages I need to perform linear regression
#import numpy as np
#import pandas as pd
#import scipy as sp 
#import matplotlib.pyplot as plt
from LassoTrain import *
import sys
sys.path.insert(0, 'C:\LearningPython\PyJamGen\Database')
from Split import *
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

Lasso_model, spotify_covariates_test,spotify_danceability_test = LassoMain()

#This function tests the linear regression on the testing data
def LassoTest(model,x_test,y_test):
    danceability_pred_Lasso = model.predict(x_test)
    print("Coefficients: {}".format(model.coef_))
    print("Mean Squared Error: {}".format(mean_squared_error(danceability_pred_Lasso, y_test)))
    print('R-squared variance explained: {}'.format(r2_score(danceability_pred_Lasso, y_test)))
    return danceability_pred_Lasso

LassoTest(Lasso_model, spotify_covariates_test,spotify_danceability_test)