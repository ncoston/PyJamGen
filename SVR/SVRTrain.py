## TO DO:
### Cross validation
### NEED TO FIND BEST ALPHA
### Other ways to optomize this method?


#Packages I need to perform linear regression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
#A method to import a file from another directory 
import sys
sys.path.insert(0, 'C:\LearningPython\PyJamGen\Database')
from Split import *



#Handles user input
def SVRMain():
    spotify_covariates_train, spotify_covariates_test, spotify_danceability_train, spotify_danceability_test = main()

    # Check for null values
    if spotify_covariates_train.isnull().values.any() == True:
        print('There are null values in the dataframe. Please remove them before proceeding.')
        exit(1)
    elif spotify_covariates_train.isnull().values.any() == False:
        print('There are no null values in the dataframe.')
    else:
        print('There was an error in checking for null values in the dataframe.')

    #Preview data that will be used for training
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
    
    #Handles user input about which covariates to train on
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
        print('What value would you like to use for epsilon (tolerance)? The default is 0.1')
        eps_val = float(input(': '))
        print('Is there a C value you prefer to use? If you select "no", cross validation will be used to find optimal C value.')
        c_val_response = input('yes/no: ')
        while c_val_response != 'yes' and c_val_response != 'no':
            print('That is not a valid response. Please enter either "yes" or "no".')
            c_val_response = input('yes/no: ')
        if c_val_response == 'yes':
            c_val = float(input('What value would you like to use for C (penalty parameter)?'))
            return SVRTrainer(spotify_covariates_train,spotify_danceability_train,c_val,eps_val), spotify_covariates_test, spotify_danceability_test
        elif c_val_response == 'no':
            c_val = CValueSelector(spotify_covariates_train,spotify_danceability_train,eps_val)
            print('The C value selected by cross validation on the training data is {}.'.format(c_val))
            return SVRTrainer(spotify_covariates_train,spotify_danceability_train,c_val,eps_val), spotify_covariates_test, spotify_danceability_test
        else: 
            print('There has been an error in selecting C.')
            exit(1)
    else:
        print('There has been an error in selecting the covariates. Please run the function again.')
        exit(1)


    
    
#training the model
def SVRTrainer(x_train,y_train,c_val, eps_val):
    #Instantiation 
    SVR_model = SVR(C=c_val, epsilon=eps_val)
    #fitting model 
    SVR_model.fit(x_train,y_train)
    return SVR_model



#Function that performs cross validation to select the best c-value.
def CValueSelector(x_train,y_train,eps_val):
    #split training data to test to find alpha
    x_cval_train, x_cval_test, y_cval_train, y_cval_test = train_test_split(x_train,y_train,test_size = .3, random_state = 42)
    c_start = 0.1
    c_end = 100
    c_inc = 0.1

    c_values = []
    MSE_scores = []

    c_val = c_start
    best_MSE_score = 100
    best_c = 0
    while (c_val<c_end):
        c_values.append(c_val)
        svr_model_loop=SVR(C=c_val,epsilon = eps_val)
        svr_model_loop.fit(x_cval_train,y_cval_train)
        svr_predict_loop_test = svr_model_loop.predict(x_cval_test)
        MSE_score_loop = mean_squared_error(y_cval_test,svr_predict_loop_test)
        MSE_scores.append(MSE_score_loop)
        if (MSE_score_loop < best_MSE_score):
            best_MSE_score = MSE_score_loop
            best_c = c_val
        c_val = c_val + c_inc
    return best_c



#SVRMain()