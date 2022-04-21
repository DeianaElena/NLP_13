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
import matplotlib.pyplot as plt
import spacy  
import wordfreq
from wordfreq.tokens import tokenize, simple_tokenize
from wordfreq import word_frequency
from wordfreq import zipf_frequency


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



############################
#Task 8 (Elena)
#Calculate the length of the tokens as the number of characters
text_tokenized=[]
length_tokens= dict()
len_token=[]
freq_token=dict()
f_tokens=[]
zip_freq_list=[]

for sent in train_sentences:
    sent_tokenized= wordfreq.tokenize(sent, 'en')  #split text in the given language into words/tokens 
    text_tokenized.append(sent_tokenized)
#print(text_tokenized)   #list with each sentence tokenized in a list

for s in text_tokenized:
    for token in s: 
        length_tokens.update({token:len(token)})  #length of the tokens as the number of characters
        len_token.append(len(token)) #length of tokens in a list
        freq_token.update({token : word_frequency(token, 'en')})  #dictionary with token and token's frequency as a decimal between 0 and 1.
        f_tokens.append(word_frequency(token, 'en')) #list of token's frequency as a decimal between 0 and 1
        zip_freq_list.append(zipf_frequency(token, 'en'))  #list of token's frequency on a human-friendly logarithmic scale.
# print('Length of each token as the number of characters \n',length_tokens)
# print('Word frequency (0-1): \n', freq_token)
# print('Zip frequency (logarithmic scale): \n', zip_freq_list)

#Provide the Pearson correlation of length and frequency with the probabilistic complexity label:

xl= np.array(len_token) #array length
xf= np.array(f_tokens)  #array frequency

#xp= np.array()   #array pos tag?

# #probabilistic complexity
# prob_complex= pass
# y_complex= np.array(prob_complex)  #array complexity


# #Pearson correlation length and complexity: 
# rho_lc= np.corrcoef(np.array(len_token), y_complex)

# #Pearson correlation frequency and complexity:

# rho_fc= np.corrcoef(np.array(f_tokens), y_complex)

# #Provide 3 scatter plots with the probabilistic complexity on the y-axis. 


# # 1)  

# #xl =                   #X-axis = Length
# # plt.scatter(xl, y)
# # plt.xlim(0,1)   #is this a good limit?
# # plt.ylim(0,1)   #is this a good limit?
# # plt.show()

# # 2) 

# #xf =                   #X-axis = Frequency
# # plt.scatter(xf, y)
# # plt.xlim(0,1)   #is this a good limit?
# # plt.ylim(0,1)   #is this a good limit?
# # plt.show()

# # 3) 


#xp = np.array(pos_tag)                 #X-axis = POS tag 
# # plt.scatter(xp, y_complex)
# # plt.xlim(0,1)   #is this a good limit?
# # plt.ylim(0,1)   #is this a good limit?
# # plt.show()
