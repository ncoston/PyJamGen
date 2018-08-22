## TO DO:
### Cross validation
### NEED TO FIND BEST ALPHA
### Other ways to optomize this method?
### How to use CV in Lasso

#Packages I need to perform linear regression
#import numpy as np
#import pandas as pd
#import scipy as sp 
from sklearn.linear_model import Lasso
#A method to import a file from another directory 
import sys
sys.path.insert(0, 'C:\LearningPython\PyJamGen\Database')
from Split import *



#Handles user input
def LassoMain():
    spotify_covariates_train, spotify_covariates_test, spotify_danceability_train, spotify_danceability_test = main()

    # Check for null values
    if spotify_covariates_train.isnull().values.any() == True:
        print('There are null values in the dataframe. Please remove them before proceeding.')
        exit(1)
    elif spotify_covariates_train.isnull().values.any() == False:
        print('There are no null values in the dataframe.')
    else:
        print('There was an error in checking for null values in the dataframe.')

    #Preview of data used for training  
    print('Here is the head of the covariate datafrme the model will be trained on:')
    print(spotify_covariates_train.head())
    print('Here is the head of the response dataframe the model will be trained on:')
    print(spotify_danceability_train.head())

    #look for zeros
    print("# of rows in data frame: {}".format(len(spotify_covariates_train)))
    print("# of zeros in energy: {}".format(len(spotify_covariates_train.loc[spotify_covariates_train['energy']==0])))
    print("# of zeros in key: {}".format(len(spotify_covariates_train.loc[spotify_covariates_train['key']==0])))
    print("# of zeros in loudness: {}".format(len(spotify_covariates_train.loc[spotify_covariates_train['loudness']==0])))
    print("# of zeros in mode: {}".format(len(spotify_covariates_train.loc[spotify_covariates_train['mode']==0])))
    print("# of zeros in speechiness: {}".format(len(spotify_covariates_train.loc[spotify_covariates_train['speechiness']==0])))
    print("# of zeros in acousticness: {}".format(len(spotify_covariates_train.loc[spotify_covariates_train['acousticness']==0])))
    print("# of zeros in instrumentalness: {}".format(len(spotify_covariates_train.loc[spotify_covariates_train['instrumentalness']==0])))
    print("# of zeros in liveness: {}".format(len(spotify_covariates_train.loc[spotify_covariates_train['liveness']==0])))
    print("# of zeros in valence: {}".format(len(spotify_covariates_train.loc[spotify_covariates_train['valence']==0])))
    print("# of zeros in tempo: {}".format(len(spotify_covariates_train.loc[spotify_covariates_train['tempo']==0])))
    print("# of zeros in duration_ms: {}".format(len(spotify_covariates_train.loc[spotify_covariates_train['duration_ms']==0])))
    print("# of zeros in time_signature: {}".format(len(spotify_covariates_train.loc[spotify_covariates_train['time_signature']==0])))

    
    # Look for correlated columns
    print('The following tables contains the correlation coefficients for all pairs of covariates: ')
    print(spotify_covariates_train.corr())

    
    #Handle user input about which columns to use as covariates
    print("Do you want to use all of these covariates to train the lasso regression model?")
    response = input('yes/no: ')

    while response == 'no':
        print('Which covariate would you like to exclude?')
        cov_to_exc = input('>')
        spotify_covariates_train = spotify_covariates_train.drop([cov_to_exc],axis =1)
        spotify_covariates_test = spotify_covariates_test.drop([cov_to_exc],axis = 1)
        print('The covariate {} has been excluded.'.format(cov_to_exc)) 
        print('The remaining covariates are the following:')
        print(list(spotify_covariates_train.columns.values))
        print('Do you want to use all of the remaining covariates to train on?')
        response = input('yes/no: ')
    while response != 'yes' and response != 'no':
        print('That is not a valid response. Please enter either yes or no.')
        response = input('yes/no: ')
    if response == 'yes':
        print('Would you like to train the model on the data as is or would you like to use cross validation?')
        train_type = input("Enter 'as is' or 'cross validation: ")
        while train_type != 'as is' and train_type != 'cross validation':
            print("That is not a valid response. Please enter either 'as is' or cross validation'.")
            train_type = input('>')
        if train_type == 'as is':
            return LasasoTrainer(spotify_covariates_train,spotify_danceability_train), spotify_covariates_test, spotify_danceability_test
        elif train_type == 'cross validation':
            return LassoTrainerCV(spotify_covariates_train,spotify_danceability_train), spotify_covariates_test, spotify_danceability_test
        else:
            print('There was an unexpected error.')
    else:
        print('There has been an error in selecting the covariates. Please run the function again.')
        exit(1)


#training the model without cross validation
def LassoTrainer(x_train,y_train):
    #Instantiation 
    #NEED TO FIND BEST ALPHA
    Lasso_model = Lasso(alpha=0.1)
    #fitting model 
    ### COME BACK AND DO CROSS VALIDATION!
    Lasso_model.fit(x_train,y_train)
    return Lasso_model


#Training the model with cross validation
def LassoTrainerCV(x_train,y_train):
    #Instantiation 
    #NEED TO FIND BEST ALPHA
    Lasso_model_CV = LassoCV(alpha=0.1)
    #fitting model 
    ### COME BACK AND DO CROSS VALIDATION!
    Lasso_model_CV.fit(x_train,y_train)
    return Lasso_model_CV

#LassoMain()