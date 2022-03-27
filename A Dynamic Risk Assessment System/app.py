from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
##import create_prediction_model
from diagnostics import model_predictions, read_dataset, dataframe_summary,get_na_values,execution_time,outdated_packages_list
from scoring import load_test_data,score_models
#import predict_exited_from_saved_model
import json
import os
import glob
import logging


logging.basicConfig(
    filename='./logs/logs.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
output_model = os.path.join(config['output_model_path']) 

with open(f'{output_model}/trainedmodel.pkl', 'rb') as mod:
    prediction_model= pickle.load(mod)
#def load_data(file):
  #  df = read_dataset()
  #  return df



#######################Prediction Endpoint
@app.route("/prediction")
def predict():
    df=read_dataset()
    predictions=model_predictions(df)

    #logging.info('SUCCESS: predict function run')
    return str(predictions)
 

#######################Scoring Endpoint
@app.route("/scoring")
def score():  
    test, y = load_test_data()
    score=score_models(prediction_model, test, y) 
  #  logging.info('SUCCESS: score function run')
    return str(score)
    

    #check the score of the deployed model
   # return #add return value (a single F1 score number)

#######################Summary Statistics Endpoint
@app.route("/summarystats")
def stats():
    summary_statistics=dataframe_summary()
  #  logging.info('SUCCESS: stats function run')
    return str([summary_statistics])

    #check means, medians, and modes for each column
    #return #return a list of all calculated summary statistics

#######################Diagnostics Endpoint
@app.route("/diagnostics")
def diag():
    df=get_na_values()
    end_time_ingest, end_time_train=execution_time()
    end_time_ingest_new=str(end_time_ingest)
    end_time_train_new=str(end_time_train)
    outd=outdated_packages_list()
    logging.info('SUCCESS: diagonistic function run')
    
    return str(df) + str(' ') + end_time_ingest_new + str(' ') + end_time_train_new + str(outd)
    #end_time_ingest_new,outd

    #check timing and percent NA values
  #  return #add return value for all diagnostics

if __name__ == "__main__": 
    #app.run(host='0.0.0.0', port=8000)
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
    #predict()
    #score()
    #s#tats()
    #diag()

    #predict()
    
