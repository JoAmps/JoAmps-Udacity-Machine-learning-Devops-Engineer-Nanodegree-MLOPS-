import requests
import glob
import json
import os
#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1/"

with open('config.json','r') as f:
    config = json.load(f)

paths = os.path.join(config['test_data_path'])     

#Call each API endpoint and store the responses
response1 = requests.get('http://127.0.0.1:8000/prediction').content
response2 = requests.get('http://127.0.0.1:8000/scoring').content
response3 = requests.get('http://127.0.0.1:8000/summarystats').content
response4 = requests.get('http://127.0.0.1:8000/diagnostics').content


#combine all API responses
responses = (response1, response2, response3, response4)

#write the responses to your workspace

def store_response(paths):
    with open(f'{paths}/apireturns.txt', 'w') as f:
        for response in str(responses):
            f.write(response)
    print("responses saved to disk")
        
if __name__ == '__main__':
    store_response(paths)        




