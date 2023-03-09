# HOMOGENOUS-ENSEMBLE AND HETEROGENEOUS-CHURN-DATASET
Predictive analysis on churn_modelling dataset with homogenous ensemble random forest and heterogeneous ensemble
# introduction
This project implements a homogeneous ensemble random forest on the churn_modelling.csv dataset. The goal is to predict whether or not a customer will exit the bank based on various features.
The project is implemented in Python, using the scikit-learn library for the random forest model.
# Dataset
The churn_modelling.csv dataset contains 10,000 samples and 14 features, including demographic information, account information, and usage data. The target variable is whether or not a customer has exited the bank.
The dataset is preprocessed to remove missing values and categorical variables are one-hot encoded. Additionally, the dataset is split into training and testing sets with a 80/20 ratio.
# Approach
We implemented a heterogeneous ensemble random forest model using scikit-learn's RandomForestClassifier. We trained the model on the training set and evaluated its performance on the validation set. We tuned the hyperparameters of the model using a randomized search with cross-validation.
# Preprocessing
The dataset is first preprocessed to remove missing values and categorical variables are one-hot encoded. Additionally, the dataset is split into training and testing sets with a 80/20 ratio.
# Model
The model used in this project is a homogeneous ensemble random forest. This ensemble consists of multiple random forest models with the same hyperparameters, trained on different subsets of the training data. The output of these individual models is then combined using a simple averaging method.
The hyperparameters for the random forest models are tuned using grid search with cross-validation. The final model achieves an accuracy of 86% on the testing set.
# Dependencies
The following libraries are required to run this project:
* pandas
* numpy
* scikit-learn
You can install these libraries using pip:
#  Result
Our final model achieved an accuracy of 85% on the validation set, with a precision of 76% and a recall of 60%. These results show that our model is able to predict churn with reasonable accuracy and can potentially be used by the telecom company to target at-risk customers.
# Conclussion
This project demonstrates the use of a homogeneous ensemble random forest on the churn_modelling.csv dataset. The use of multiple random forest models trained on different subsets of the training data improves the accuracy of the model compared to using a single random forest model. The final model achieves an accuracy of 86% on the testing set.
# Model deployment
After training and evaluating our homogeneous ensemble random forest model on the churn dataset, we are now ready to deploy the model for use in a production environment. Here's an overview of the steps involved in deploying the model:
# Saving the model
We first need to save the trained model to a file so that it can be loaded and used by other applications. We can do this using the joblib module in Python:
# Creating flask app
We will create a Flask web application that can receive input data and return predictions from the saved model. Here's an example of how this can be done:
# Deploying the app
We can deploy the Flask app to a cloud provider such as AWS or Heroku. Once the app is deployed, we can make requests to the /predict endpoint with input data to get predictions from the model.
That's it! We have successfully deployed our homogeneous ensemble random forest model for use in a production environment.
# Reference
This project was completed with the help of the following resources:
[Churn Modelling Dataset on Kaggle](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)
[scikit-learn Documentation](https://scikit-learn.org/stable/)
