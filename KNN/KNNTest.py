#Packages I need to test KNN
from KNNTrain import *
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error

#KNN_model, spotify_covariates_test,spotify_danceability_test = KNNMain()

#This function tests the linear regression on the testing data
def KNNTest(model,x_test,y_test):
    danceability_pred_KNN = model.predict(x_test)
    print("After training and testing the k-Nearest Neighbors Regression model, here are some metrics of the model")
    print("Mean Squared Error: {}".format(mean_squared_error(y_test, danceability_pred_KNN))) #close to zero is better
    print('Mean absolute error: {}'.format(mean_absolute_error(y_test, danceability_pred_KNN))) #error under L1 norm, closer to zero is better
    print('R-squared variance explained: {}'.format(r2_score(y_test, danceability_pred_KNN))) #Closer to one is better
    print('Explained variance score: {}'.format(explained_variance_score(y_test, danceability_pred_KNN))) #closer to one is better
    return danceability_pred_KNN

#KNNTest(KNN_model, spotify_covariates_test,spotify_danceability_test)