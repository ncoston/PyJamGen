#Consider other ways to improve linear regression

#Packages I need to perform linear regression
from sklearn.linear_model import LinearRegression
#A method to import a file from another directory
#Split contains my main() file, which selects data and splits into training and testing sets.
import sys
sys.path.insert(0, 'C:\LearningPython\PyJamGen\Database')
from Split import *



#Handles user input
def LinRegMain():
    #main() splits data into testing and training sets, from the Database directory
    spotify_covariates_train, spotify_covariates_test, spotify_danceability_train, spotify_danceability_test = main()

    # Check for null values
    if spotify_covariates_train.isnull().values.any() == True:
        print('There are null values in the dataframe. Please remove them before proceeding.')
        exit(1)
    elif spotify_covariates_train.isnull().values.any() == False:
        print('There are no null values in the dataframe.')
    else:
        print('There was an error in checking for null values in the dataframe.')

    #Preview of data that will be used for training
    print('Here is the head of the covariate datafrme the model will be trained on:')
    print(spotify_covariates_train.head())
    print('Here is the head of the response dataframe the model will be trained on:')
    print(spotify_danceability_train.head())


    #Look for zeros that could cause problems
    #If a column contains too many zeros, I should consider excluding it as a covariate
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
    # If two columns are highly correlated, they could result in a large bias.
    print('The following tables contains the correlation coefficients for all pairs of covariates: ')
    print(spotify_covariates_train.corr())

    print("Do you want to use all of these covariates to train the linear regression model?")
    response = input('yes/no: ')

#I excluded the following method because normalizing didn't improve results. 
#   #Option to normalize data
#    print('Do you want to normalize your data?')
#    norm = input('yes/no: ')
#
#    while norm != 'yes' and norm != 'no':
#        print('That is not a valid response. Please enter either yes or no.')
#        norm = input('yes/no: ') 
#    if norm == 'no':
#        normalized = False
#    elif norm == 'yes':
#        normalized = True
#    else:
#        print('There has been an error. Please run the function again.')
#        exit(1)

    #Handle user input about which columns to use as covariates
    while response == 'no':
        print('Which covariate would you like to exclude?')
        cov_to_exc = input('>')
        spotify_covariates_train = spotify_covariates_train.drop([cov_to_exc],axis =1)
        spotify_covariates_test = spotify_covariates_test.drop([cov_to_exc],axis = 1)
        print('The covariate {} has been excluded.'.format(cov_to_exc)) 
        print('The remaining covariates are the following:')
        print(list(spotify_covariates_train.columns.values))
        print('Do you want to use all of the remaining covariates to train the linear model?')
        response = input('yes/no: ')
    while response != 'yes' and response != 'no':
        print('That is not a valid response. Please enter either yes or no.')
        response = input('yes/no: ')
    if response == 'yes':
        return LinRegTrainer(spotify_covariates_train,spotify_danceability_train), spotify_covariates_test, spotify_danceability_test
    else:
        print('There has been an error in selecting the covariates. Please run the function again.')
        exit(1)

    
    
#training the model without cross validation
def LinRegTrainer(x_train,y_train):
    #Instantiation
    #I am leaving 'fit_intercept = False' because I have not centered covariates.
    #After investigation, normalizing/standardizing data did not improve performance. This normalize = Flase is left as default.  
    Lin_model = LinearRegression()
    #fitting model 
    Lin_model.fit(x_train,y_train)
    return Lin_model


#LinRegMain()