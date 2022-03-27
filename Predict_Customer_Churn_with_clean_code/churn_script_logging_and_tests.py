"""
Logging and testing
author:Hyacinth Ampadu
date:8 Feb 2022
"""

import logging
from sklearn.model_selection import train_test_split
from churn_library import import_data, perform_eda, perform_feature_engineering, \
    encoder_helper, train_models

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
        test data import - this example is completed for you to assist with the other test functions
        '''
    try:
        dataframe = import_data("BankChurners.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err
    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
        test perform eda function
        inputs:output from import_data function
        outputs:
    '''
    try:
        dataframe = import_data("BankChurners.csv")
        input_data, categorical_columns, output_data = perform_eda(dataframe)
        logging.info('Testing EDA data:SUCCESS')

    except FileNotFoundError as err:
        logging.error("Testing perform_eda: Figures not saved")

    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing perform_eda: The file doesn't appear to have rows and columns")
        raise err


def test_encoder_helper(encoder_helper):
    '''
        test encoder helper
        '''
    try:
        dataframe = import_data("BankChurners.csv")
        input_data, categorical_columns, output_data = perform_eda(dataframe)
        dataframe = encoder_helper(
            dataframe, dataframe[categorical_columns], 'Churn')
        group_lst = []
        for i in categorical_columns:
            groups = dataframe.groupby(i).mean()['Churn']
        for val in dataframe[i]:
            group_lst.append(groups.loc[val])
        logging.info('SUCCESS:mean categorical of churn performed')
    except BaseException:
        logging.error('ERROR:mean categorical of churn failed')
    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "test_encoder_helper: data doesn't appear to have rows and columns")
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
        test perform_feature_engineering
        '''
    try:
        dataframe = import_data("BankChurners.csv")
        input_data, categorical_columns, output_data = perform_eda(dataframe)
        dataframe = encoder_helper(
            dataframe, dataframe[categorical_columns], 'Churn')
        input_training_data, input_testing_data, output_training_data, output_testing_data,\
            input_data = perform_feature_engineering(
                dataframe, output_data, input_data, categorical_columns)
        input_training_data, input_testing_data, output_training_data, output_testing_data\
            = train_test_split(
                input_data, output_data, test_size=0.3, random_state=42)
        logging.info(
            'SUCCESS:Data succesfully split into training and testing')
    except BaseException:
        logging.error(
            'ERROR:Problem occured during splitting into training and testing')
    try:
        if not input_data.shape[0] == output_data.shape[0]:
            raise AssertionError((input_data.shape[0], output_data.shape[0]))
    except AssertionError as err:
        logging.error("X and y have different number of rows!")
        raise err


def test_train_models(train_models):
    '''
        test train_models
        '''
    try:
        dataframe = import_data("BankChurners.csv")
        input_data, categorical_columns, output_data = perform_eda(dataframe)
        dataframe = encoder_helper(
            dataframe, dataframe[categorical_columns], 'Churn')
        input_training_data, input_testing_data, output_training_data, output_testing_data,\
            input_data = perform_feature_engineering(
                dataframe, output_data, input_data, categorical_columns)
        lrc, cv_rfc, output_training_data, output_testing_data, y_train_preds_lr, y_train_preds_rf,\
            y_test_preds_lr, y_test_preds_rf = train_models(
                input_training_data, input_testing_data, output_training_data, output_testing_data)
        cv_rfc.fit(input_training_data, output_training_data)
        lrc.fit(input_training_data, output_training_data)
        logging.info('SUCESS:Models trained Succesfully')
    except BaseException:
        logging.error('ERROR: Models failed to train')
    try:
        y_train_preds_rf = cv_rfc.best_estimator_.predict(input_training_data)
        y_test_preds_rf = cv_rfc.best_estimator_.predict(input_testing_data)
        y_train_preds_lr = lrc.predict(input_training_data)
        y_test_preds_lr = lrc.predict(input_testing_data)
        logging.info(
            'SUCCESS: best estimators and predictions obtained sucessfuly')
        logging.info('SUCCESS: Code successfully run')
    except BaseException:
        logging.error('ERROR: Estimators and predictors malfunctioned')
    try:
        if not input_training_data.shape[0] == output_training_data.shape[0]:
            raise AssertionError(
                (input_training_data.shape[0],
                 output_training_data.shape[0]))
        if not input_testing_data.shape[0] == output_testing_data.shape[0]:
            raise AssertionError(
                (input_testing_data.shape[0],
                 output_testing_data.shape[0]))
    except AssertionError as err:
        logging.error(
            "Lengths of the Independent and dependent training and testing data mismatch")
        raise err


if __name__ == '__main__':
    test_import(import_data)
    test_eda(perform_eda)
    test_encoder_helper(encoder_helper)
    test_perform_feature_engineering(perform_feature_engineering)
    test_train_models(train_models)
