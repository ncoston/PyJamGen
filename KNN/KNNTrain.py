#Packages I need to perform linear regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
#A method to import a file from another directory 
import sys
sys.path.insert(0, 'C:\LearningPython\PyJamGen\Database')
from Split import *



#Handles user input
#no inputs required 
#returns trained model, and covariate and response data
def KNNMain():
    spotify_covariates_train, spotify_covariates_test, spotify_danceability_train, spotify_danceability_test = main()

    # Check for null values
    if spotify_covariates_train.isnull().values.any() == True:
        print('There are null values in the dataframe. Please remove them before proceeding.')
        exit(1)
    elif spotify_covariates_train.isnull().values.any() == False:
        print('There are no null values in the dataframe.')
    else:
        print('There was an error in checking for null values in the dataframe.')

    #Preview data to be used to train model
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

    #Handles user input about which covariates to train on.
    print("Do you want to use all of these covariates to train the lasso regression model?")
    response = input('yes/no: ')

    while response == 'no': #removes user specified covariate from data frame
        print('Which covariate would you like to exclude?')
        cov_to_exc = input('>')
        spotify_covariates_train = spotify_covariates_train.drop([cov_to_exc],axis =1)
        spotify_covariates_test = spotify_covariates_test.drop([cov_to_exc],axis = 1)
        print('The covariate {} has been excluded.'.format(cov_to_exc)) 
        print('The remaining covariates are the following:')
        print(list(spotify_covariates_train.columns.values))
        print('Do you want to use all of the remaining covariates to train on?')
        response = input('yes/no: ')
    while response != 'yes' and response != 'no': #prompts user if input is invalid
        print('That is not a valid response. Please enter either yes or no.')
        response = input('yes/no: ')
    if response == 'yes': #prompts user to select metric for weights on neighbors
        print('Would you like to do distance based weights or uniform weights?')
        weights = input("Please enter 'distance' or 'uniform': ")
        while weights != 'distance' and weights != 'uniform': #prompts user if input is invalid
            print('That is not a valid response.')
            weights = input("Please enter 'distance' or 'uniform': ")
        print('Do you want to specify the value of k (number of neighbors)?') 
        print('(If you do not specify the value of k, cross validation will be used to select the value of k.)')
        num_of_neighbors =  input('yes/no: ')
        while num_of_neighbors !='yes' and num_of_neighbors != 'no':#prompts user if input is invalid
            print('That is not a valid response. Please enter either yes or no.')
            num_of_neighbors = input('yes/no: ')
        if num_of_neighbors == 'yes': #collects user input for the value of k, number of neighbors to use
            k = int(input('Please specify what value of k you would like to use: '))
            return KNNTrainer(spotify_covariates_train,spotify_danceability_train,k,weights), spotify_covariates_test, spotify_danceability_test #Calls the KNNTrainer function
        elif num_of_neighbors == 'no': #If user does not specify how many neighbors to use, KValueSelector is called which selects the number of neighbors based on cross validation
            k = KValueSelector(spotify_covariates_train,spotify_danceability_train,weights)
            return KNNTrainer(spotify_covariates_train,spotify_danceability_train,k,weights), spotify_covariates_test, spotify_danceability_test #calls the KNNTrainer function
        else:
            print('There has been an error in selecting the covariates. Please run the function again.')
            exit(1)
    else:
        print('There has been an error in selecting the covariates. Please run the function again.')
        exit(1)


    
    
#training the model
#requires training covariate data, training response data, the number of neighbors to use, and the weight type
#Returns trained model
def KNNTrainer(x_train, y_train, k, weight_type):
    #Instantiation 
    KNN_model = KNeighborsRegressor(n_neighbors = k, weights = weight_type)
    #fitting model 
    KNN_model.fit(x_train,y_train)
    return KNN_model



#Function that performs cross validation to select the number of neighbors to train on.
#requires training covariate data, testing covariate data, and eight type
def KValueSelector(x_train,y_train,weight_type):
    #split training data to test to find alpha
    x_kval_train, x_kval_test, y_kval_train, y_kval_test = train_test_split(x_train,y_train,test_size = .3, random_state = 42)
    k_start = 1
    k_end = x_kval_train.shape[0]
    k_inc = 1

    k_values = []
    MSE_scores = []

    k_val = k_start
    best_MSE_score = 100
    best_k = 0
    #this while loop will increment the value of k staarting with k_start and ending with k_end with the stepsize of 1
    #it will train and predict using each k value and store the best k value
    while (k_val<k_end):
        k_values.append(k_val) #add k value to list
        knn_model_loop=KNeighborsRegressor(n_neighbors = k_val, weights = weight_type) #instantiate with k value
        knn_model_loop.fit(x_kval_train,y_kval_train) #train with k value
        knn_predict_loop_test = knn_model_loop.predict(x_kval_test) #predict with trained model
        MSE_score_loop = mean_squared_error(y_kval_test,knn_predict_loop_test) #test accuracy of predictions
        MSE_scores.append(MSE_score_loop) #add score to loop
        if (MSE_score_loop < best_MSE_score):
            best_MSE_score = MSE_score_loop #store best mean square error
            best_k = k_val #store k value corresponding to best mean square error
        k_val = k_val + k_inc #increment and continue
    return best_k





#KNNMain()