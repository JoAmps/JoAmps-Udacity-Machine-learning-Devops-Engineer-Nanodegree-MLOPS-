import logging
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

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 

def read_data():
    try:
        df=glob.glob(f'{dataset_csv_path}/*.csv')
        df=pd.concat([pd.read_csv(i) for i in (df)])
        df=df.drop(columns=['Unnamed: 0','corporation'])
        X=df.drop(columns='exited')
        y=df['exited']
        logging.info('SUCCESS:Data read and X and y generated')
        return X,y
    except:
        logging.info('ERROR:Data not read and X and y not generated')


#################Function for training the model
def train_model(X,y):
    
    #use this logistic regression for training
    try:
        model=LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                        intercept_scaling=1, l1_ratio=None, max_iter=100,
                        multi_class='warn', n_jobs=None, penalty='l2',
                        random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                        warm_start=False)
        model=LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
        intercept_scaling=1, l1_ratio=None, max_iter=100,
        multi_class='auto', n_jobs=None, penalty='l2',
        random_state=0, solver='liblinear', tol=0.0001, verbose=0,
        warm_start=False)
        
        #fit the logistic regression to your data
        model.fit(X,y)
                        
        #write the trained model to your workspace in a file called trainedmodel.pkl
        file = open(f'{model_path}/trainedmodel.pkl', 'wb')
        logging.info('SUCESS:Model trained succesfully and stored as a pickle file')
        return pickle.dump(model,file)
    except:
        logging.info('ERROR:Model failed to train and stored')



if __name__ == '__main__':
    X,y=read_data()
    train_model(X,y)

