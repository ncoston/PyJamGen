#Packages I need to test support vector regression
from SVRTrain import *
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error

#SVR_model, spotify_covariates_test,spotify_danceability_test = SVRMain()

#This function tests the linear regression on the testing data
def SVRTest(model,x_test,y_test):
    danceability_pred_SVR = model.predict(x_test)
    print("After training and testing the Support Vector Regression model, here are some metrics of the model")
    print("Mean Squared Error: {}".format(mean_squared_error(y_test, danceability_pred_SVR))) #close to zero is better
    print('Mean absolute error: {}'.format(mean_absolute_error(y_test, danceability_pred_SVR))) #error under L1 norm, closer to zero is better
    print('R-squared variance explained: {}'.format(r2_score(y_test, danceability_pred_SVR))) #Closer to one is better
    print('Explained variance score: {}'.format(explained_variance_score(y_test, danceability_pred_SVR))) #closer to one is better
    return danceability_pred_SVR

#SVRTest(SVR_model, spotify_covariates_test,spotify_danceability_test)