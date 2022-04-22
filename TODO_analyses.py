# %%
# File: data/preprocessed/train/sentences.txt 
# Model: en_core_web_sm.

# Import packages
import os
import numpy as np
import torch
import tabulate
from tqdm import tqdm
import pandas as pd
import spacy
import nltk
from collections import Counter
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet 
from nltk.tokenize import word_tokenize

# Part A
# Task 1: Tokenization  

nlp = spacy.load("en_core_web_sm")
txt_path = './data/preprocessed/train/sentences.txt'

with open(txt_path,'r', encoding="utf8") as in_file:    #encoding for windows
    my_text = in_file.read()
    #to remove new lines and additional initial/ending spaces
    my_text = my_text.rstrip('\n') 
    my_text= my_text.replace('\n', '')  

# Number of tokens (with spacy):
    doc= nlp(my_text)
    tokens=[token.text for token in doc]
    print('Number of tokens (without \n):', len(tokens))     

# Number of types:
    unique_tokens1= set(tokens) 
    print('Number of types (with punctuation counted only ones and only unique words):',len(unique_tokens1))  #3765  #punctuation is not included so it's number of unique words
    
# Number of words:

    words=[token.text for token in doc if token.is_stop != True and token.is_punct != True]  #to break sentences into words without punctuation
    #print(words)
    only_words = " ".join(words)
    print(only_words)
    doc2=nlp.tokenizer(only_words)   #text with only words without punctuation
    print('Number of words (without punctuation):', len(doc2))   

# Average number of words per sentence:
sentences_tokenized=list(doc.sents)
#print('Number of sentences:', len(sentences_tokenized))   
list_n_words=[]
for sent in sentences_tokenized:
    list_n_words.append(len(sent))
average_word_in_sentences= np.mean(list_n_words)
print('Average number of words per sentence:', average_word_in_sentences)   
    
# Average word length: 
list_l_words=[]
for word in tokens:
    list_l_words.append(len(word))
print('Average word length:', np.mean(list_l_words))

##########################################################################

# Task 2
#Tokens
tokens = []
for token in nltk.word_tokenize(my_text):
    tokens.append(token)
print(len(tokens))
# Finegrained POS-tag    
FG = []
for token in pos_tag(nltk.word_tokenize(my_text)):
    FG.append(token)

# Universal POS-tag    
Universal = []
for token in pos_tag(nltk.word_tokenize(my_text), tagset='universal') :
    Universal.append(token)
    
#Create df 
df = pd.DataFrame(FG, columns =['Token', 'FG_POS'])
df1 = pd.DataFrame(Universal, columns =['Token', 'Universal_POS'])
WC = df.merge(df1, left_index=True, right_index=True)

WC = WC.drop(['Token_y'], axis=1)
WC = WC.rename({'Token_x':'Token'}, axis=1)

WC_table = WC.groupby(['FG_POS', 'Universal_POS']).size().reset_index(name='Occurrences')
WC_table['Relative Tag Frequency (%)'] = round(WC_table['Occurrences']/WC_table['Occurrences'].sum() * 100,2)
#sort
WC_table = WC_table.sort_values( by="Occurrences", ascending=False)
#get top 10
WC_table = WC_table.nlargest(n=10, columns=['Occurrences'])

#Extract most occuring POS
WC_most_occ = WC_table[['FG_POS', 'Universal_POS']]
new_df = pd.merge(WC_most_occ, WC, on =['FG_POS', 'Universal_POS'])
new_df = new_df.groupby(['FG_POS', 'Universal_POS', 'Token']).size()

most_frequent = new_df.groupby(['FG_POS', 'Universal_POS']).nlargest(3)
infrequent = new_df.groupby(['FG_POS', 'Universal_POS']).nsmallest(1)
##########################################################################

# Task 3 N-grams
from nltk.util import bigrams, trigrams, ngrams

token = nltk.word_tokenize(my_text)
POS = pos_tag(nltk.word_tokenize(my_text), tagset='universal')

token_bigrams = list(nltk.bigrams(token))
token_trigrams = list(nltk.trigrams(token))
              
POS_bigrams = list(nltk.bigrams(POS))
POS_trigrams = list(nltk.trigrams(POS))

token_bigrams_df = pd.DataFrame(token_bigrams, columns =['BiGram', 'BiGram_2'])
token_trigrams_df = pd.DataFrame(token_trigrams, columns =['TriGram', 'TriGram_2', 'TriGram_3'])

POS_bigrams_df = pd.DataFrame(POS_bigrams, columns =['BiGram', 'BiGram_2'])
POS_trigrams_df = pd.DataFrame(POS_trigrams, columns =['TriGram', 'TriGram_2', 'TriGram_3'])

POS_bigrams_df[['BiGram1', 'BiGram2']] = pd.DataFrame(POS_bigrams_df['BiGram'].tolist(), index=POS_bigrams_df.index)
POS_bigrams_df[['BiGram_21', 'BiGram_22']] = pd.DataFrame(POS_bigrams_df['BiGram_2'].tolist(), index=POS_bigrams_df.index)
POS_bigrams_df = POS_bigrams_df.drop(['BiGram','BiGram_2','BiGram1','BiGram_21',], axis=1)

POS_trigrams_df[['TriGram1', 'TriGram2']] = pd.DataFrame(POS_trigrams_df['TriGram'].tolist(), index=POS_trigrams_df.index)
POS_trigrams_df[['TriGram_21', 'TriGram_22']] = pd.DataFrame(POS_trigrams_df['TriGram_2'].tolist(), index=POS_trigrams_df.index)
POS_trigrams_df[['TriGram_31', 'TriGram_32']] = pd.DataFrame(POS_trigrams_df['TriGram_3'].tolist(), index=POS_trigrams_df.index)
POS_trigrams_df = POS_trigrams_df.drop(['TriGram','TriGram_2','TriGram_3', 'TriGram1', 'TriGram_21','TriGram_31'], axis=1)

token_bigrams_df['BiGram'] = token_bigrams_df[['BiGram', 'BiGram_2']].agg(', '.join, axis=1)
token_trigrams_df['TriGram'] = token_trigrams_df[['TriGram', 'TriGram_2', 'TriGram_3']].apply(
    lambda x: ','.join(x.dropna().astype(str)),
    axis=1)

POS_bigrams_df['BiGram'] = POS_bigrams_df[['BiGram2', 'BiGram_22']].agg(', '.join, axis=1)
POS_trigrams_df['TriGram'] = POS_trigrams_df[['TriGram2', 'TriGram_22', 'TriGram_32']].apply(
    lambda x: ','.join(x.dropna().astype(str)),
    axis=1)

token_bigrams_df = token_bigrams_df.drop(['BiGram_2'], axis=1)
token_trigrams_df = token_trigrams_df.drop(['TriGram_2', "TriGram_3"], axis=1)
POS_bigrams_df = POS_bigrams_df.drop(['BiGram2','BiGram_22'], axis=1)
POS_trigrams_df = POS_trigrams_df.drop(['TriGram2','TriGram_22', "TriGram_32"], axis=1)

token_bigrams_df = token_bigrams_df['BiGram'].reset_index()
token_trigrams_df = token_trigrams_df['TriGram'].reset_index()
POS_bigrams_df = POS_bigrams_df['BiGram'].reset_index()
POS_trigrams_df = POS_trigrams_df['TriGram'].reset_index()

token_bigrams_df = token_bigrams_df.drop('index', 1)
token_trigrams_df = token_trigrams_df.drop('index', 1)

POS_bigrams_df = POS_bigrams_df.drop('index', 1)
POS_trigrams_df = POS_trigrams_df.drop('index', 1)

#top 3's
token_bigrams_df['BiGram'].value_counts()
token_trigrams_df['TriGram'].value_counts()
POS_bigrams_df['BiGram'].value_counts()
POS_trigrams_df['TriGram'].value_counts()

##########################################################################

# Task 4 Lemmatization
lemma = []
for token, POStag in pos_tag(word_tokenize(my_text)):
    if POStag.startswith('N'):
        lemma.append(WordNetLemmatizer().lemmatize(token, wordnet.NOUN))
    elif POStag.startswith('V'):
        lemma.append(WordNetLemmatizer().lemmatize(token, wordnet.VERB))
    elif POStag.startswith("J"):
        lemma.append(WordNetLemmatizer().lemmatize(token, wordnet.ADJ))
    elif POStag.startswith('R'):
        lemma.append(WordNetLemmatizer().lemmatize(token, wordnet.ADV))
    else:
        lemma.append(token)
        
df = pd.DataFrame(tokens, columns =['Token'])
df1 = pd.DataFrame(lemma, columns =['Lemma'])
lemma_df = df.merge(df1, left_index=True, right_index=True)

##########################################################################

# Task 5: Named Entity Recognition 
#Number of named entities:
list_entities=[]
for ent in doc.ents:
            entities=ent.text
            list_entities.append(entities)
#print(list_entities)
print("Number of named entities (with \\\):",len(list_entities)) #symbol // is included

#Number of different entity labels:
labels = set([w.label_ for w in doc.ents])
print('Number of different entity labels:', len(labels)) 

#Analyze the named entities in the first five sentences
first_entities=[]
first_5= sentences_tokenized[0:5]
#print(first_5)
for sent in first_5:
    first_entities.append(sent.ents)
print("Entities first 5 sentences (with \\\):", first_entities)
