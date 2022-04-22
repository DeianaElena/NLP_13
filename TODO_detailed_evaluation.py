# The original code only outputs accuracy & loss.
# Process the file model_output.tsv and calculate: precision, recall & F1 for each class

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
import re

def process_output(test_input):
    predictions = []
    instances = []
    
    
    for instance in test_input:
        instance = instance.replace("\n","")
        instance = instance.replace("----------","")
        tokens = instance.split("\t")
        tokens = list(filter(None, tokens))
        
        
        if tokens != []:
            for t in tokens[2]:
                predictions.append(t)
                predictions = [val for sublist in predictions for val in sublist]
                                   
            for t in tokens[1]:
                instances.append(t)
                instances = [val for sublist in instances for val in sublist]
            
            
    cm = classification_report(instances, predictions, digits = 2)
            
    return cm


if __name__ == '__main__':
    path = "/experiments/base_model/"

    # Note: this loads all instances into memory. If you work with bigger files in the future, use an iterator instead.

    with open(path + "/model_output.tsv", encoding='utf8') as sent_file:
        model_output = sent_file.readlines()

    cm = process_output(model_output)
    print(cm)



    data = {
        'epoch': [1,5,10,15,25, 40,50],
        'F1 weighted average': [0.70, 0.82,0.84,0.84,0.84,0.84,0.84],
        }
    data2 = {
        'embedding dimension': [5,20,50,100,200,400,800],
        'F1 weighted average': [0.70, 0.82,0.84,0.84,0.85,0.86,0.84],
        } 

    df = pd.DataFrame(data)
    df1 = pd.DataFrame(data2)
    df.plot.line(x='epoch', y='F1 weighted average')
    df1.plot.line(x='embedding dimension', y='F1 weighted average')
