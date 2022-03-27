import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import glob
from sklearn.metrics import confusion_matrix


###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
output_model = os.path.join(config['output_model_path']) 

with open(f'{output_model}/trainedmodel.pkl', 'rb') as mod:
    model = pickle.load(mod)


##############Function for reporting
def score_model(model,confusion_matrix,path):
    df=glob.glob(f'{test_data_path}/*.csv')
    df=pd.concat([pd.read_csv(i) for i in (df)])
    df=df.drop(columns=['corporation'])
    y=df['exited']
    test = df.drop(columns='exited') 
    predictions=model.predict(test)
    sns.heatmap(confusion_matrix(predictions,y),annot=True, annot_kws={"fontsize":20}, fmt='d', cbar=False,cmap='icefire')
    plt.savefig(
            f"{path}/confusion_matrix.png",
            bbox_inches='tight',
            dpi=1000)
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace





if __name__ == '__main__':
    score_model(model,confusion_matrix,"testdata")
