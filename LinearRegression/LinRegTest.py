## TO DO: 
### Other helpful diagnostics?


#Packages I need to perform linear regression
#import numpy as np
#import pandas as pd
#import scipy as sp 
#import matplotlib.pyplot as plt
from LinRegTrain import *
import sys
sys.path.insert(0, 'C:\LearningPython\PyJamGen\Database')
from Split import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

LinRegModel, spotify_covariates_test,spotify_danceability_test = LinRegMain()

#This function tests the linear regression on the testing data
def LinRegTest(model,x_test,y_test):
    danceability_pred_LinReg = model.predict(x_test)
    print("Coefficients: {}".format(model.coef_))
    print("Mean Squared Error: {}".format(mean_squared_error(danceability_pred_LinReg, y_test)))
    print('R-squared variance explained: {}'.format(r2_score(danceability_pred_LinReg, y_test)))
    return danceability_pred_LinReg

LinRegTest(LinRegModel, spotify_covariates_test,spotify_danceability_test)