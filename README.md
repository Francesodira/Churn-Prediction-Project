# Bank Customer Churn Prediction



## Overview

This project aims to predict customer churn in the banking industry using machine learning algorithms. Customer churn prediction is vital for banks to retain valuable customers and improve their overall performance. The project follows the CRISP-DM process, consisting of several phases, including data understanding, data preparation, model training, model evaluation, and model deployment.

## Table of Contents

- [Business Understanding](#business-understanding)
- [Data Understanding](#data-understanding)
- [Data Preparation](#data-preparation)
- [Model Development and Training](#model-development-and-training)
- [Model Evaluation](#model-evaluation)
- [Model Optimization](#model-optimization)
- [Model Deployment](#model-deployment)
- [Usage](#usage)
- [Contributors](#contributors)
- [License](#license)

## Business Understanding

The primary goal of this project is to develop a robust machine learning model to predict customer churn in the banking industry. By identifying customers at risk of churning, banks can implement retention strategies to minimize the negative impacts of churn, increase customer satisfaction, and strengthen their financial health.

## Data Understanding

The dataset used in this project, named "Churn_Modeling.csv," is sourced from [superdatascience.com](https://www.superdatascience.com) and is available on [Kaggle](https://www.kaggle.com). It contains information about bank customers, including age, tenure, balance, number of products, credit score, geography, gender, and credit card ownership. The target variable indicates whether the customer has churned (1 for churned, 0 for not churned).

## Data Preparation

Data preparation involves several steps, including feature engineering, encoding categorical variables, feature selection, balancing the dataset, data splitting, and feature scaling. These steps are crucial to prepare the data for machine learning model training.

## Model Development and Training

In this project, we experiment with various supervised machine learning algorithms, including Decision Trees, Logistic Regression, Random Forest, and Artificial Neural Networks, to predict customer churn. These algorithms are chosen based on their suitability for the classification problem.

## Model Evaluation

The performance of the models is evaluated using various metrics such as accuracy, precision, recall, and F1-score. We assess the models' ability to correctly predict churn and identify customers at risk.

## Model Optimization

Model optimization involves fine-tuning hyperparameters and selecting the best-performing model. We use techniques like Grid Search to find the best parameters for each model.

## Model Deployment

The best-performing model is saved and deployed into a Flask web application to facilitate easy interaction and predictions.

## Usage

- Clone this repository.
- Install the required libraries using `pip install -r requirements.txt`.
- Run the Flask web application to make predictions.

## Contributors

- [Frances odira Omegara](https://github.com/yourusername) - Student



