# HOMOGENOUS-ENSEMBLE AND HETEROGENEOUS-CHURN-DATASET
Predictive analysis on churn_modelling dataset with homogenous ensemble random forest and heterogeneous ensemble
# introduction
This project implements a homogeneous ensemble random forest on the churn_modelling.csv dataset. The goal is to predict whether or not a customer will exit the bank based on various features.
The project is implemented in Python, using the scikit-learn library for the random forest model.
# Dataset
The churn_modelling.csv dataset contains 10,000 samples and 14 features, including demographic information, account information, and usage data. The target variable is whether or not a customer has exited the bank.
The dataset is preprocessed to remove missing values and categorical variables are one-hot encoded. Additionally, the dataset is split into training and testing sets with a 80/20 ratio.
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
