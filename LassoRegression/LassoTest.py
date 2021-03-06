#Packages I need to test lasso regression
from LassoTrain import *
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error

#Lasso_model, spotify_covariates_test,spotify_danceability_test = LassoMain()

#This function tests the lasso regression on the testing data
#Inputs required are a trained lasso model, covariate data to predict with, and true output data to test predictions
def LassoTest(model,x_test,y_test):
    danceability_pred_Lasso = model.predict(x_test)
    print("After training and testing the Lasso Model, here are some metrics from the model.")
    print("Coefficients: {}".format(model.coef_)) #coefficients in model
    print("Mean Squared Error: {}".format(mean_squared_error(y_test, danceability_pred_Lasso))) #close to zero is better
    print('Mean absolute error: {}'.format(mean_absolute_error(y_test, danceability_pred_Lasso))) #error under L1 norm, closer to zero is better
    print('R-squared variance explained: {}'.format(r2_score(y_test, danceability_pred_Lasso))) #Closer to one is better
    print('Explained variance score: {}'.format(explained_variance_score(y_test, danceability_pred_Lasso))) #closer to one is better
    return danceability_pred_Lasso

#LassoTest(Lasso_model, spotify_covariates_test,spotify_danceability_test)