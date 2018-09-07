# PyJamGen - A Python regression program for Spoitfy data

### Project Description
This project reads in a csv file containing spotify data, then predicts the danceability of a new song based on the other covariates. The user may select from a variety of regresion methods to train and test a model. The project gives the user to implement Linear Regression, Lasso Regression, Ridge Regression, Support Vector Regression, and k-Nearest Neighbors Regression. The user is returned a variety of statistics to determine how well the model performed. 

### Project set up
The project contains the following directories and files that the user may wish to interact with:
* Database
    * Split.py
    * featuresdf.csv
    * TopTracks.sql
* KNN
    * KNNTrain.py
    * KNNTest.py
* LassoRegression
    * LassoTrain.py
    * LassoTest.py
* LinearRegression
    * LinRegTrain.py
    * LinRegTest.py
* RidgeRegression
    * RidgeTrain.py
    * RidgeTest.py
* SVR
    * SVRTrain.py
    * SVRTest.py
* MethodComparison
    * MainMethodComparison.py

The Database directory contains the Split.py file, which reads in the data and splits into appropriate training and testing sets. Database also contains the data itself in the featuresdf.csv file. The TopTracks.sql is not currently used, but I plan to expand the project in the future to interact with a SQL database. 

The directories KNN, LassoRegression, LinearRegression, RidgeRegression, and SVR all contain train and test files which train a specific model, then test the model by predicting using the trained model and testing these predictions against the true values in the testing data. 

Finally, the MethodComparison directory contains the MainMethodComparison.py file, which is the main file that runs all other files. This gives the user a choice of what method to use and calls the appropriate function.

### Instructions for use
After cloning the project, navigate to the MethodComparison directory and run the MainMethodComparison.py file. This fist will prompt the user to enter which method for regression they would like to use. Once the method is selected, the user will be prompted to enter the file which contains the data for use. The user will need to use "..\Database\featuresdf.csv" to access the data. The program will then prompt the user to enter the percentage of data used for testing. It will give a summary of the data and ask the user to select which covariates to train on. Depending on the method, the user will need to either enter parameters necessary for the model or select the option to do cross validation to select the most optimal parameter. The program will then return various statistics to determine how well the program performed. If the user would like to use a different model, rin MainMethodComparison.py again. 
