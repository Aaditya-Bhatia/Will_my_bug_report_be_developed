# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 20:56:59 2019

@author: bhati
"""


import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
import re
import seaborn as sns
import matplotlib.pyplot as plt
import gensim
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
import multiprocessing
from sklearn import utils
cores = multiprocessing.cpu_count()



resDir = "C:/Users/bhati/Desktop/Stack Exchange Research/results/"
dirName = "C:/Users/bhati/Desktop/Stack Exchange Research/data/"
fea = pd.read_csv(dirName+'allFeatures_1.csv')


feaNames = ['CommentCount_q','FavoriteCount', 'Score_q', 
       'ViewCount', 'UpVotes_q', 'DownVotes_q', 'BountyAmount',
       'Reputation', 'CountBugs', 'prevBugs',
       'prevComments', 'prevAnswerCount', 'prevUpvotes', 'prevDownvotes',
       'prevFavourites', 'prevScore', 'prevViewcount', 'allPrevBugs',
       'allPrevComments', 'allPrevAnswerCount', 'allPrevUpvotes',
       'allPrevDownvotes', 'allPrevFavourites', 'allPrevScore',
       'allPrevViewcount', 'totalEditsOnBugs', 'totalEditsOnAllQues', 'finalLemma',
#       'finalText', 'initText','initLemma', 'finalLemma', 'diffWords',
       'diffWordLength', 'IsEditorDev', 'finalFlesch', 'finalcontainsImg',
       'finalcontainsEnv', 'finalEntropy', 'initEntropy', 'Label']

features = fea[feaNames]

train, test = train_test_split(features, test_size=0.3, random_state=42)

test_tagged = test.apply(
    lambda r: TaggedDocument(words=test.finalLemma, tags=[r.Label]), axis=1)
train_tagged = train.apply(
    lambda r: TaggedDocument(words=train.finalLemma, tags=[r.Label]), axis=1)

model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample = 0, workers = 8)
model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])

train_tagged.values

def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors
  
y_train, X_train = vec_for_learning(model_dbow, train_tagged)
y_test, X_test = vec_for_learning(model_dbow, test_tagged)

type(X_train[2])
# preprocessing b4 concatenating
del test['Label'], train['Label']
del test['finalLemma'], train['finalLemma']
del test['IsEditorDev'], test['level_0'], 
del train['IsEditorDev'], train['level_0']
test = test.reset_index()
train = train.reset_index()
del test['index'],  train['index']

test = (test - test.mean())/test.std()
train = (train- train.mean())/train.std()

X_test1 = pd.concat([pd.DataFrame(list(X_test)), test], axis = 1)
X_train1 = pd.concat([pd.DataFrame(list(X_train)), train], axis = 1)

pd.DataFrame(list(y_train)).to_csv(resDir+'y_train1.csv', index=False)
pd.DataFrame(list(y_test)).to_csv(resDir+'y_test1.csv', index=False)
X_test1.to_csv(resDir+'X_test1.csv', index=False, encoding = 'utf-8')
X_train1.to_csv(resDir+'X_train1.csv', index=False)

#saving the doc2 vec resullts as well
pd.DataFrame(list(X_test)).to_csv(resDir+'X_test.csv', index=False)
pd.DataFrame(list(X_train)).to_csv(resDir+'X_train.csv', index=False)
                    