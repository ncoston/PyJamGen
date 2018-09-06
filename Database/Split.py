#Imports I need
from sklearn.model_selection import train_test_split
import pandas as pd

#function to read in and split csv file
def Split(csv_name,split_test_size):
    # read in CSV file
    spotify_songs = pd.read_csv(csv_name)
    #Split data into predictors and values I want to predict
    spotify_covariates = spotify_songs[['energy','key','loudness','mode','speechiness','acousticness','instrumentalness', 'liveness', 'valence', 'tempo','duration_ms', 'time_signature']]
    spotify_danceability = spotify_songs['danceability']

    #Split into testing and training data 
    spotify_covariates_train, spotify_covariates_test, spotify_danceability_train, spotify_danceability_test = train_test_split(spotify_covariates,spotify_danceability,test_size = split_test_size, random_state = 42)
    return spotify_covariates_train, spotify_covariates_test, spotify_danceability_train, spotify_danceability_test

#main function to handle user input
def main():
    print('Please specify the csv file you would like to use:')
    csv_name = input('csv file:')
    print('What percentage of data would you like to use for testing? Enter as a decimal between 0 and 1:')
    split_test_size = float(input('percent for testing:'))
    while split_test_size <0 or split_test_size>1:
        print('You must enter the percentage as a decimal. Please enter a value between 0 and 1')
        split_test_size = float(input('percent for testing:'))
    spotify_covariates_train, spotify_covariates_test, spotify_danceability_train, spotify_danceability_test = Split(csv_name,split_test_size)

    #verify the split worked as expected
    print('Covariates for testing:', spotify_covariates_test.shape)
    print('Covariates for training:', spotify_covariates_train.shape)
    print('Response for testing:',spotify_danceability_test.shape)
    print('Response for training:', spotify_danceability_train.shape)

    print('Do these data frames appear to be the correct size?')
    response = input('yes/no>')

    #prompt user to verify the data looks correct
    while(response != 'yes' and response != 'no'):
        print("That is not a valid response. Please enter either 'yes' or 'no'.")
        response = input('yes/no>')
    if response == 'yes':
        return spotify_covariates_train, spotify_covariates_test, spotify_danceability_train, spotify_danceability_test
    else:
        print('There is an issue with the size of the data sets. Please see Split.py')
        exit(1)

#main() 