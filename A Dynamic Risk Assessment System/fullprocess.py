import logging
import ingestion         
from training import train_model
from scoring import score_models
from deployment import store_model_into_pickle
from reporting import score_model
#from apicalls import store_response
from sklearn.metrics import confusion_matrix
#from app import predict, score 
import subprocess

##import diagnostics
#import reporting
import os
import json
import glob
import pandas as pd
import pickle
import numpy as np
import ast
import logging
##################Check and read new data
#first, read ingestedfiles.txt
with open('config.json','r') as f:
    config = json.load(f)
paths = os.path.join(config['output_model_path'])

prod_path = os.path.join(config['prod_deployment_path'])
input_data_path = os.path.join(config['input_folder_path'])

logging.basicConfig(
    filename='./logs/fp.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

with open('config.json','r') as f:
    config = json.load(f) 
    

prod_path = os.path.join(config['prod_deployment_path'])
input_data_path = os.path.join(config['input_folder_path'])

with open(f'{prod_path}/trainedmodel.pkl', 'rb') as mod:
    model = pickle.load(mod)

def run():      
    ingested_files = open(f'{prod_path}/ingestedfiles.txt', 'r')
    input_data=glob.glob(f'{input_data_path}/*.csv')
    for x in ingested_files:
        for y in input_data:
            result=x==y
            if result ==True: 
                print('no new data available')
                print('end process')
                exit
            else:
                print('new_data_available')
                os.system('python3 ingestion.py')
                latest_data_files = glob.glob(f'{input_data_path}/*.csv')
                most_recent_data = max(latest_data_files)
                df=pd.read_csv(most_recent_data)
                df.head()
                df=df.drop(columns=['corporation'])
                y=df['exited']
                test = df.drop(columns='exited')
                #print(test)
                #model,confusion_matrix,path
                score=score_models(model, test, y)
                print(score)
                with open(f'{prod_path}/latestscore.txt', 'r') as fp:
                    contents = ast.literal_eval(fp.read())
                if score<np.min(contents)!=True:
                    print('model_drift not occurred')
                    print('end process')
                    exit
                else:
                    print('model_drift occurred')
                    train_model(test,y)
                    store_model_into_pickle(prod_path)
                    score_model(model,confusion_matrix,"models")
                    exit
                        #subprocess.run("python3 app.py & python3 apicalls.py", shell=True)

if __name__ == '__main__':
    run()                        
                
                
              