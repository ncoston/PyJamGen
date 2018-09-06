## TO DO: 
#Comment everything 


#Packages I need to test linear regression
#LinRegTrain file contains my training method
from LinRegTrain import * 
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error

#The next line is used for testing
#LinRegModel, spotify_covariates_test,spotify_danceability_test = LinRegMain()

#This function tests the linear regression on the testing data
def LinRegTest(model,x_test,y_test):
    danceability_pred_LinReg = model.predict(x_test)
    print("Coefficients: {}".format(model.coef_)) #coefficients in model
    print("Mean Squared Error: {}".format(mean_squared_error(y_test, danceability_pred_LinReg))) #close to zero is better
    print('Mean absolute error: {}'.format(mean_absolute_error(y_test, danceability_pred_LinReg))) #error under L1 norm, closer to zero is better
    print('R-squared variance explained: {}'.format(r2_score(y_test, danceability_pred_LinReg))) #Closer to one is better
    print('Explained variance score: {}'.format(explained_variance_score(y_test, danceability_pred_LinReg))) #closer to one is better

    return danceability_pred_LinReg

#LinRegTest(LinRegModel, spotify_covariates_test,spotify_danceability_test)