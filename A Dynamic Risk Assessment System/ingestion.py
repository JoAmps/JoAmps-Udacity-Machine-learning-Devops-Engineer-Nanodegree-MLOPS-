import logging
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import glob

logging.basicConfig(
    filename='./logs/logs.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']



#############Function for data ingestion
def merge_multiple_dataframe():
    try:
        #check for datasets, compile them together, and write to an output file
        datasets=glob.glob(f'{input_folder_path}/*.csv')
        compiled_datasets=pd.concat([pd.read_csv(i) for i in (datasets)])
        compiled_datasets=compiled_datasets.drop_duplicates().reset_index().drop(columns='index')
        logging.info('SUCCESS:Datasets found, compiled, and duplicates dropped')  
        return compiled_datasets.to_csv(f'{output_folder_path}/'+'finaldata.csv')
    except:
        logging.info('ERROR:Datasets could not be merged')    
    

def save_record():
    try:
        current_datasets = glob.glob(f'{input_folder_path}/*.csv')
        with open('ingesteddata/ingestedfiles.txt', 'w') as f:
            for dataset in current_datasets:
                f.write(dataset[13:])
                f.write('\n')
        logging.info('SUCCESS:Ingested files written to file')        
    except:
        logging.info('ERROR: Ingested files not written to file')

if __name__ == '__main__':
    merge_multiple_dataframe()
    save_record()
