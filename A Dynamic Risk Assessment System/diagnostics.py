
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import glob
import logging
import subprocess

logging.basicConfig(
    filename='./logs/logs.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')
##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
prod_path = os.path.join(config['prod_deployment_path']) 

with open(f'{prod_path}/trainedmodel.pkl', 'rb') as mod:
    model = pickle.load(mod)

def read_dataset():
    df=pd.concat([pd.read_csv(i) for i in glob.glob(f'{test_data_path}/*.csv')])
    df=glob.glob(f'{test_data_path}/*.csv')
    df=pd.concat([pd.read_csv(i) for i in (df)])
    df=df.drop(columns=['corporation'])
    y=df['exited']
    test_data = df.drop(columns='exited') 
    return test_data

##################Function to get model predictions
def model_predictions(test_data):
    try:
        with open(f'{prod_path}/trainedmodel.pkl', 'rb') as mod:
            model = pickle.load(mod)
        predictions=model.predict(test_data)
        logging.info('SUCCESS: Model predictions obtained')
    except:
        logging.info('ERROR: Model predictions not obtained') 
    try:
        if not len(predictions) == test_data.shape[0]:
            raise AssertionError(
                (len(predictions),
                 test_data.shape[0]))
        logging.info('Lengths of the Prediction equals the number of rows of test data')         
    except AssertionError as err:
        logging.error(
            "Lengths of the Prediction and number of rows of test data dont match")
        raise err
    #read the deployed model and a test dataset, calculate predictions
    return predictions

##################Function to get summary statistics
def dataframe_summary():
    df=[pd.read_csv(i) for i in glob.glob("ingesteddata/*.csv")]
    df=pd.concat(df)
    df=df.drop(columns='Unnamed: 0').select_dtypes(include=[np.int64])
    summary_statistics=[]
    for i in df.columns:
        mean = df[i].mean()
        standard_deviation = df[i].std()
        median = df[i].median()
        summary_statistics=({'column_name':i,'mean':mean, 'median':median, 'standard_deviation':standard_deviation})
        print(list(summary_statistics.values()))
    return list(summary_statistics.values())
    #calculate summary statistics here
     #return value should be a list containing all summary statistics
def get_na_values():
    df=[pd.read_csv(i) for i in glob.glob("ingesteddata/*.csv")]
    df=pd.concat(df)
    df=df.drop(columns='Unnamed: 0').select_dtypes(include=[np.int64])
    print(list(df.columns))
    print('Number of missing data in each column:',list(df.isnull().sum()))
    print('Percentage of data that is missing in each column:',list(df.isnull().sum()/len(df)))
    return list((df.isnull().sum()/len(df)))
##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    start_time_ingest=timeit.default_timer()
    os.system('python3 ingestion.py')
    end_time_ingest=timeit.default_timer()-start_time_ingest
    print(end_time_ingest)
    start_time_train=timeit.default_timer()
    os.system('python3 training.py')
    end_time_train=timeit.default_timer()-start_time_train
    print(end_time_train)
    return end_time_ingest, end_time_train


##################Function to check dependencies
def outdated_packages_list():
    outdated = subprocess.check_output(['pip', 'list','--outdated'])
    with open('requirements.txt', 'wb') as f:
       f.write(outdated)
    return outdated     


if __name__ == '__main__':
    test_data=read_dataset()
    model_predictions(test_data)
    dataframe_summary()
    get_na_values()
    execution_time()
    outdated_packages_list()





    
