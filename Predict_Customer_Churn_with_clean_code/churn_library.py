    # library doc string
"""
Importing the required libraries to be used to solve this problem
author:Hyacinth Ampadu
date:8th february 2022
"""

#import libraries
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    try:
        df = pd.read_csv(pth)
        print('data returned')
        return df
    except FileNotFoundError:
        print('File cannot be found')


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    try:

        cat_columns = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'
        ]

        quant_columns = [
            'Customer_Age',
            'Dependent_count',
            'Months_on_book',
            'Total_Relationship_Count',
            'Months_Inactive_12_mon',
            'Contacts_Count_12_mon',
            'Credit_Limit',
            'Total_Revolving_Bal',
            'Avg_Open_To_Buy',
            'Total_Amt_Chng_Q4_Q1',
            'Total_Trans_Amt',
            'Total_Trans_Ct',
            'Total_Ct_Chng_Q4_Q1',
            'Avg_Utilization_Ratio']

        df['Churn'] = df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        # churn
        plt.figure(figsize=(20, 10))
        plt.title('Distribution of Churn among customers')
        plt.ylabel('Number of Customers')
        plt.xlabel('Churn or no-churn')
        df['Churn'].hist()
        plt.savefig(
            "./images/eda/churn_distribution.png",
            bbox_inches='tight',
            dpi=1000)
        # age
        plt.figure(figsize=(20, 10))
        df['Customer_Age'].hist()
        plt.savefig(
            "./images/eda/customer_age_distribution.png",
            bbox_inches='tight',
            dpi=1000)
        plt.title('Distribution of Age of customers')
        plt.ylabel('Number of Customers')
        plt.xlabel('Ages')
        # transactions
        plt.figure(figsize=(20, 10))
        sns.displot(df['Total_Trans_Ct'])
        plt.title('Distribution of Transactions')
        plt.ylabel('Number of Customers')
        plt.xlabel('Transactions')
        plt.savefig(
            "images/eda/total_transaction_distribution.png",
            bbox_inches='tight',
            dpi=1000)
        # heatmap
        plt.figure(figsize=(20, 10))
        sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
        plt.title('Correlation among the features')
        plt.ylabel('Features')
        plt.xlabel('Features')
        plt.savefig("images/eda/heatmap.png", bbox_inches='tight', dpi=1000)
        y = df['Churn']
        X = pd.DataFrame()
        print('EDA performed')
        return X, cat_columns, y
    except BaseException:
        print('No dataframe found')


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could
             be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''

    try:
        group_lst = []
        for i in category_lst:
            groups = df.groupby(i).mean()[response]
        for val in df[i]:
            group_lst.append(groups.loc[val])

        print('encoder_helper_performed')
        return df
    except BaseException:
        print('dataframe cannot be found')


def perform_feature_engineering(df, response, X, category_lst):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could
               be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    df = pd.concat([df[category_lst].add_suffix('_Churn'), df], axis=1)
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']
    try:

        X[keep_cols] = df[keep_cols]
        X = pd.get_dummies(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, response, test_size=0.3, random_state=42)
        print('feature engineering performed')

        return X_train, X_test, y_train, y_test, X
    except BaseException:
        print('Data could not be found')


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    try:

        # grid search
        rfc = RandomForestClassifier(random_state=42)
        lrc = LogisticRegression(solver='liblinear')

        param_grid = { 
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth' : [4,5,100],
        'criterion' :['gini', 'entropy']
                }
        print('models training')

        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
        cv_rfc.fit(X_train, y_train)
        print('random forest finished')
        lrc.fit(X_train, y_train)
        print('logistic regression finished')
        y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
        y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)
        print('best estimators determined')
        y_train_preds_lr = lrc.predict(X_train)
        y_test_preds_lr = lrc.predict(X_test)

        print('training finished')

        lrc_plot = plot_roc_curve(lrc, X_test, y_test)
        plt.figure(figsize=(15, 8))
        ax = plt.gca()
        rfc_disp = plot_roc_curve(
            cv_rfc.best_estimator_,
            X_test,
            y_test,
            ax=ax,
            alpha=0.8)
        lrc_plot.plot(ax=ax, alpha=0.8)
        plt.savefig(
            "./images/results/roc_curve_results.png",
            bbox_inches='tight',
            dpi=1000)
        joblib.dump(cv_rfc.best_estimator_, "./models/rfc_model.pkl")
        joblib.dump(lrc, "./models/logistic_model.pkl")
        print('roc curves obtained and models saved')
        return lrc, cv_rfc, y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf
    except BaseException:
        print('Training and testing data not found')


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    try:
        print('random forest results')
        print('test results')
        print(classification_report(y_test, y_test_preds_rf))
        print('train results')
        print(classification_report(y_train, y_train_preds_rf))
        plt.savefig(
            "'./images/results/rf_results.png'",
            bbox_inches='tight',
            dpi=1000)

        print('logistic regression results')
        print('test results')
        print(classification_report(y_test, y_test_preds_lr))
        print('train results')
        print(classification_report(y_train, y_train_preds_lr))
        plt.savefig(
            "./images/results/logistic_results.png",
            bbox_inches='tight',
            dpi=1000)

    except BaseException:
        print('no report to be generated')


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    try:
        explainer = shap.TreeExplainer(model.best_estimator_)
        shap_values = explainer.shap_values(X_data)
        shap.summary_plot(shap_values, X_data, plot_type="bar", show=False)
        plt.savefig(
            "./images/results/shap_feature_importance.png",
            bbox_inches='tight',
            dpi=1000)
        print('Shap feature importance performed')
        importances = model.best_estimator_.feature_importances_
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]

        # Rearrange feature names so they match the sorted feature importances
        names = [X_data.columns[i] for i in indices]

        # Create plot
        plt.figure(figsize=(20, 5))
        # Create plot title
        plt.title("Feature Importance")
        plt.ylabel('Importance')
        # Add bars
        plt.bar(range(X_data.shape[1]), importances[indices])
        # Add feature names as x-axis labels
        plt.xticks(range(X_data.shape[1]), names, rotation=90)
        plt.savefig(
            "./images/results/RFfeature_importance.png",
            bbox_inches='tight',
            dpi=1000)
        print('RF feature importance performed')
        print('Code succesfully run')

    except BaseException:
        print('Feature importance could not be generated')
