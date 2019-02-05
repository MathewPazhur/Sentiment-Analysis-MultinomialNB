# -*- coding: utf-8 -*-
"""
Created on Thu Jan 03 19:00:05 2019

@author: Mathew
"""

# Importing the libraries

import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
import seaborn as sns
import string
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

#from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Importing the dataset

amazon = pd.read_csv('1000.csv')

#Removing null entries

amazon.isnull().sum()
amazon = amazon.fillna(' ')
amazon.shape

# Text Length 

amazon['text length'] = amazon['reviewText'].apply(len)

# Plotting distribution diagram

g = sns.FacetGrid(data=amazon, col='overall')
g.map(plt.hist, 'text length', bins=50)

# Creating a class with only 5 and 1 stars 

amazon_class = amazon[(amazon['overall'] <3 ) | (amazon['overall'] >3)]
amazon_class.shape

# Generating X and Y coordinates

X = amazon_class['reviewText']
y = amazon_class['overall']

# Data Preprocessing

def text_process(text):


    nopunc = [char for char in text if char not in string.punctuation]

    nopunc = ''.join(nopunc)
    
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

# Vectorizing reviews
    
bow_transformer = CountVectorizer(analyzer=text_process).fit(X)
len(bow_transformer.vocabulary_)

X = bow_transformer.transform(X)
print('Shape of Sparse Matrix: ', X.shape)
print('Amount of Non-Zero occurrences: ', X.nnz)

# Training and Testing

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
nb = MultinomialNB()
nb.fit(X_train, y_train)

preds = nb.predict(X_test)

# Print the Results

#print(confusion_matrix(y_test, preds))
#print('\n')
#print(classification_report(y_test, preds))
#print("Accuracy = ", accuracy_score(y_test, preds, normalize=True, sample_weight=None))

def eval_predictions(y_test, preds):
    print('accuracy:', metrics.accuracy_score(y_test, preds))
    print('precision:', metrics.precision_score(y_test, preds, average='weighted'))
    print('recall:', metrics.recall_score(y_test, preds, average='weighted'))
    print('F-measure:', metrics.f1_score(y_test, preds, average='weighted'))
eval_predictions(y_test, preds)

