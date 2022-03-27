from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import glob
from sklearn.metrics import f1_score
import logging


logging.basicConfig(
    filename='./logs/logs.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
output_model = os.path.join(config['output_model_path']) 

with open(f'{output_model}/trainedmodel.pkl', 'rb') as mod:
    model = pickle.load(mod)

def load_test_data():
    try:
        df=glob.glob(f'{test_data_path}/*.csv')
        df=pd.concat([pd.read_csv(i) for i in (df)])
        df=df.drop(columns=['corporation'])
        y=df['exited']
        test = df.drop(columns='exited')  
        logging.info('SUCCESS: data obtained and ready to be predicted upon')
        return test, y 
    except:
         logging.info('ERROR: data not obtained')    

#################Function for model scoring
def score_models(model, test, y):
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    try:
        predictions=model.predict(test)
        score=f1_score(y, predictions)
        
        with open("testdata/latestscore.txt","w") as file:
            file.write(score.astype('str'))
        logging.info("SUCCESS: f1 score obtained and results written to file")  
        return score  
    except:
        logging.info("ERROR: f1 score could not be obtained")
        

if __name__ == '__main__':
    test, y=load_test_data()
    score_models(model, test, y)    
