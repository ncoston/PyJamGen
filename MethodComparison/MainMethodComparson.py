#Imports needed
import sys
sys.path.insert(0, 'C:\LearningPython\PyJamGen\LinearRegression')
from LinRegTest import *

sys.path.insert(0, 'C:\LearningPython\PyJamGen\LassoRegression')
from LassoTest import *

sys.path.insert(0, 'C:\LearningPython\PyJamGen\RidgeRegression')
from RidgeTest import *

sys.path.insert(0, 'C:\LearningPython\PyJamGen\KNN')
from KNNTest import *

sys.path.insert(0, 'C:\LearningPython\PyJamGen\SVR')
from SVRTest import *

def MethodMain():
    print('Hello and welcome to the spotify danceability predictor.')
    print('Based on the other features of the spotify data, this project will train a model to predict danceability.')
    print('This project offers a variety of models on which to train. The model you may choose from are the following:')
    print('Linear Regression \nLasso Regression \nRidge Regression \nk-Nearest Neighbors Regression \nSupport Vector Regression.')
    print('Which of these methods would you like to use? Please type it precisely as it is written above')
    method = input('>')
    if method == 'Linear Regression':
        LinRegModel, spotify_covariates_test,spotify_danceability_test = LinRegMain()
        LinRegTest(LinRegModel, spotify_covariates_test,spotify_danceability_test)
    elif method == 'Lasso Regression':
        Lasso_model, spotify_covariates_test,spotify_danceability_test = LassoMain()
        LassoTest(Lasso_model, spotify_covariates_test,spotify_danceability_test)
    elif method == 'Ridge Regression':
        Ridge_model, spotify_covariates_test,spotify_danceability_test = RidgeMain()
        RidgeTest(Ridge_model, spotify_covariates_test,spotify_danceability_test)
    elif method == 'k-Nearest Neighbors Regression':
        KNN_model, spotify_covariates_test,spotify_danceability_test = KNNMain()
        KNNTest(KNN_model, spotify_covariates_test,spotify_danceability_test)
    elif method == 'Support Vector Regression':
        SVR_model, spotify_covariates_test,spotify_danceability_test = SVRMain()
        SVRTest(SVR_model, spotify_covariates_test,spotify_danceability_test)
    else:
        print('That is not a valid method. Please select a method from the following options and type your selection exactly as written.')
        print('Linear Regression \nLasso Regression \nRidge Regression \nk-Nearest Neighbors Regression \nSupport Vector Regression.')
        method = input('>')

MethodMain()