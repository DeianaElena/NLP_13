# %%
# Implement four baselines for the task.
# Majority baseline: always assigns the majority class of the training data
# Random baseline: randomly assigns one of the classes. Make sure to set a random seed and average the accuracy over 100 runs.
# Length baseline: determines the class based on a length threshold
# Frequency baseline: determines the class based on a frequency threshold

from model.data_loader import DataLoader
import spacy 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy  
import wordfreq
from scipy import stats
from wordfreq.tokens import tokenize, simple_tokenize
from wordfreq import word_frequency
from nltk import pos_tag

#Task 8 
df_header= ['ID', 'sentence', 'start', 'end', 'target_word', 'num_native', 'num_non-native',
             'native_difficult', 'non-native_difficult', 'binary', 'probabilistic_class']  #as described here https://sites.google.com/view/cwisharedtask2018/datasets

df_train = pd.read_csv('data/original/english/WikiNews_Train.tsv', sep='\t', names=df_header)

#The 10th and 11th columns show the gold-standard label for the binary and probabilistic classification tasks.
new_df = df_train[(df_train['binary'] != 0) & (df_train['target_word'].str.split(" ").apply(len) == 1)]   #data frame with only complex token and to include only tokens with one word
#new_df.to_csv('newdf5.csv')  #in case we want to export it
list_ctokens= list(new_df['target_word'])    #list of complex token

# #Calculate the length of the tokens as the number of characters and its frequency
len_ctoken=[]
freq_ctoken=[]
pos_list=[]
for token in list_ctokens:
    len_ctoken.append(len(token))  #list with length of each complex token is in a list
    freq_ctoken.append(word_frequency(token, 'en')) #list of token's frequency as a decimal between 0 and 1
#print(len_ctoken)   
#print(freq_ctoken)
print(pos_list)

# #Provide the Pearson correlation of length and frequency with the probabilistic complexity label:
y_complex= list(new_df['probabilistic_class'])  #list of proobabilistic complexity
#print('y complex',y_complex)

rho_lc= stats.pearsonr(len_ctoken, y_complex)   #Pearson correlation length and complexity: 
print(rho_lc)  #(0.2814998053444496, 1.2455732080643942e-46)
rho_fc= stats.pearsonr(freq_ctoken, y_complex)  #Pearson correlation frequency and complexity:
print(rho_fc)  #(-0.297710467861033, 3.365294215612193e-52)

#Provide 3 scatter plots with the probabilistic complexity on the y-axis. 
# 1)  X-axis = Length
plt.scatter(len_ctoken, y_complex)
plt.xlim(0,25); plt.ylim(0,1.15) 
plt.title("Correlation of Length and Probabilistic complexity")  
plt.xlabel('Length'); plt.ylabel('Complexity')
plt.show()

# 2) X-axis = Frequency
plt.scatter(freq_ctoken, y_complex)
plt.xlim(0,0.0006); plt.ylim(0,1.15)  
plt.title("Correlation of Frequency and Probabilistic complexity")
plt.xlabel('Frequency'); plt.ylabel('Complexity')
plt.show()

# 3) X-axis = POS tag 
pos_tags= pos_tag(list_ctokens)
only_POStag = [a_tuple[1] for a_tuple in pos_tags]
#print(only_POStag)

plt.scatter(freq_ctoken, only_POStag)
plt.xlim(0,0.0006); plt.ylim(0,20)
plt.title("Correlation of Frequency and POS tags")
plt.xlabel('Frequency'); plt.ylabel('POS tags')
plt.show()

################################################################


# Each baseline returns predictions for the test data. The length and frequency baselines determine a threshold using the development data.

# def majority_baseline(train_sentences, train_labels, testinput, testlabels):
#     predictions = []

#     # TODO: determine the majority class based on the training data
#     # ...
#     majority_class = "X"
#     predictions = []
#     for instance in testinput:
#         tokens = instance.split(" ")
#         instance_predictions = [majority_class for t in tokens]
#         predictions.append(instance, instance_predictions)

#     # TODO: calculate accuracy for the test input
#     # ...
#     return accuracy, predictions



if __name__ == '__main__':
    train_path = "./data/preprocessed/train"
    val_path = "./data/preprocessed/val"
    test_path = "./data/preprocessed/test"

    # Note: this loads all instances into memory. If you work with bigger files in the future, use an iterator instead.

    with open(train_path + "/sentences.txt", encoding="utf8") as sent_file:
        train_sentences = sent_file.readlines()

    with open(train_path + "/labels.txt", encoding="utf8") as label_file:
        train_labels = label_file.readlines()

    with open(val_path + "/sentences.txt", encoding="utf8") as val_file:
        val_sentences = val_file.readlines()

    with open(train_path + "/labels.txt", encoding="utf8") as val_label_file:
        val_labels = val_label_file.readlines()
    with open(test_path + "/sentences.txt") as testfile:
        testinput = testfile.readlines()

    with open(test_path + "/labels.txt", encoding="utf8") as test_labelfile:
        testlabels = test_labelfile.readlines()
   # majority_accuracy, majority_predictions = majority_baseline(train_sentences, train_labels, testinput)

    # TODO: output the predictions in a suitable way so that you can evaluate them



