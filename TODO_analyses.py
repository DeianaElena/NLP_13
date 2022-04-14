# %%
# File: data/preprocessed/train/sentences.txt 
# Model: en_core_web_sm.

# Import packages
import os
import numpy as np
import torch
import tabulate
from tqdm import tqdm
import nltk
import spacy  
from collections import Counter

# Part A
# Task 1: Tokenization   (Elena)

nlp = spacy.load("en_core_web_sm")
txt_path = './data/preprocessed/train/sentences.txt'

with open(txt_path,'r', encoding="utf8") as in_file:    #encoding for windows
    my_text = in_file.read()

# Number of tokens (with nltk):
    tokens = nltk.word_tokenize(my_text)
    #print(tokens)
    num_tokens = len(tokens) 
    print('Total of tokens (without /n):', num_tokens)    #15214

# Number of tokens (with spacy):
    doc= nlp(my_text)
    tokens1=[token.text for token in doc]
    print('Number of tokens (with /n):', len(tokens1))   #16130 it's more because it takes into account \n?   

# Number of types:
    unique_tokens1= set(tokens) 
    print('Number of types (with punctuation counted only ones and only unique words):',len(unique_tokens1))  #3765  #punctuation is not included so it's number of unique words
    
# Number of words:
    words=[token.text for token in doc if token.is_stop != True and token.is_punct != True]  #to break sentences into words without punctuation
    only_words = " ".join(words)
    doc2=nlp.tokenizer(only_words)   #text with only words without punctuation
    print('Number of words (without punctuation):', len(doc2))   #7995

# Average number of words per sentence:
sentences_tokenized=list(doc.sents)
print('Number of sentences:', len(sentences_tokenized))   #718
list_n_words=[]
for sent in sentences_tokenized:
    list_n_words.append(len(sent))
average_word_in_sentences= np.mean(list_n_words)
print('Average number of words per sentence:', average_word_in_sentences)   #22.465

      
# Average word length: 
list_l_words=[]
for word in tokens:
    list_l_words.append(len(word))
print('Average word length:', np.mean(list_l_words))

##########################################################################

# Task 2:

##########################################################################

# Task 3:

##########################################################################

# Task 4:

##########################################################################

# Task 5: Named Entity Recognition   (Elena)

#Number of named entities:
list_entities=[]
for ent in doc.ents:
            entities=ent.text
            list_entities.append(entities)
#print(list_entities)
print("Number of named entities (with \\\):",len(list_entities)) #not sure if // needs to be removed or not

#Number of different entity labels:
labels = set([w.label_ for w in doc.ents])
print('Number of different entity labels:', len(labels)) 


#Analyze the named entities in the first five sentences (in progress)
first_entities=[]
first_5= sentences_tokenized[0:5]
#print(first_5)
for sent in first_5:
    first_entities.append(sent.ents)
print("Entities first 5 sentences (with \\\):", first_entities)

