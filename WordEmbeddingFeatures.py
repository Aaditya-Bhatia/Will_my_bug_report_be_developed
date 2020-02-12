# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 20:56:59 2019

@author: bhati
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
import multiprocessing
cores = multiprocessing.cpu_count()

dirName = "/Users/aadityabhatia/All Backup/all_sail_lab_TSE_stuff/StackExchangeResearch/data_new/"

class Embeddings():
    def __init__(self, dirName, feaNames):
        fea = pd.read_csv(dirName+'bugReportFeaturess.csv')
        features = fea[feaNames]    
        self.train, self.test = train_test_split(features, test_size=0.3, random_state=42)
        X_train, X_test, y_test, y_train= self.getWordEmbedding()
        self.clean_df()
        self.concatenate_save(X_train, X_test, y_test, y_train)
        
    def getWordEmbedding(self):
        test_tagged = self.test.apply(
            lambda r: TaggedDocument(words=self.test.finalLemma, tags=[r.Label]), axis=1)
        train_tagged = self.train.apply(
            lambda r: TaggedDocument(words=self.train.finalLemma, tags=[r.Label]), axis=1)
        model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample = 0, workers = 8)
        model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])

        y_train, X_train = self.vec_for_learning(model_dbow, train_tagged)
        y_test, X_test = self.vec_for_learning(model_dbow, test_tagged)
        return X_train, X_test, y_test, y_train
    
    
    def vec_for_learning(self, model, tagged_docs):
        sents = tagged_docs.values
        targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
        return targets, regressors
      

    def clean_df(self):
        test, train = self.test, self.train
        del test['Label'], train['Label']
        del test['finalLemma'], train['finalLemma']
        test = test.reset_index()
        train = train.reset_index()
        del test['index'],  train['index']
        
        self.test = (test - test.mean())/test.std()
        self.train = (train- train.mean())/train.std()
        
    def concatenate_save(self, X_train, X_test, y_test, y_train):
        X_test1 = pd.concat([pd.DataFrame(list(X_test)), self.test], axis = 1)
        X_train1 = pd.concat([pd.DataFrame(list(X_train)), self.train], axis = 1)
        
        pd.DataFrame(list(y_train)).to_csv(dirName+'y_train.csv', index=False)
        pd.DataFrame(list(y_test)).to_csv(dirName+'y_test.csv', index=False)
        
        X_test1.to_csv(dirName+'X_test_concat.csv', index=False, encoding = 'utf-8')
        X_train1.to_csv(dirName+'X_train_concat.csv', index=False, encoding = 'utf-8')

        #saving the doc2 vec embeddings resullts as well
        pd.DataFrame(list(X_test)).to_csv(dirName+'X_test_embeddings.csv', index=False)
        pd.DataFrame(list(X_train)).to_csv(dirName+'X_train_embeddings.csv', index=False)
        
Embeddings(dirName, feaNames=['CommentCount_q','FavoriteCount', 'Score_q', 
       'ViewCount', 'UpVotes_q', 'DownVotes_q', 'BountyAmount',
       'Reputation', 'CountBugs', 'prevBugs',
       'prevComments', 'prevAnswerCount', 'prevUpvotes', 'prevDownvotes',
       'prevFavourites', 'prevScore', 'prevViewcount', 'allPrevBugs',
       'allPrevComments', 'allPrevAnswerCount', 'allPrevUpvotes',
       'allPrevDownvotes', 'allPrevFavourites', 'allPrevScore',
       'allPrevViewcount', 'totalEditsOnBugs', 'totalEditsOnAllQues', 'finalLemma',
       'diffWordLength', 'finalFlesch', 'finalcontainsImg',
       'finalcontainsEnv', 'finalEntropy', 'initEntropy', 'Label'])