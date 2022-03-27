from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import shutil
import logging

logging.basicConfig(
    filename='./logs/logs.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path'])
output_model = os.path.join(config['output_model_path']) 
test_data = os.path.join(config['test_data_path'])  


####################function for deployment
def store_model_into_pickle(path):
    try:
        #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
        ingested_file="/Users/hyacinthampadu/Documents/Jos Folder/Data Science/Udacity mL devops engineer/project 4/starter-file/ingesteddata/ingestedfiles.txt"
        pickle_file="/Users/hyacinthampadu/Documents/Jos Folder/Data Science/Udacity mL devops engineer/project 4/starter-file/practicemodels/trainedmodel.pkl"
        scores_file="/Users/hyacinthampadu/Documents/Jos Folder/Data Science/Udacity mL devops engineer/project 4/starter-file/testdata/latestscore.txt"
        for i in [pickle_file,scores_file,ingested_file]:
            shutil.copy(i, f'{path}')
        logging.info("SUCCESS:All files moved to production folder")
    except:
        logging.info("ERROR:Files could not be moved to production folder")        


if __name__ == '__main__':
    store_model_into_pickle(prod_deployment_path)          
        
        
        

