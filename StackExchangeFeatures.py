# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 19:15:58 2019

@author: bhati
"""

import pandas as pd
import textstat
from math import log, e
import numpy as np
from bs4 import BeautifulSoup

# NLP Libraries
import string
import nltk
#nltk.download()
from nltk.corpus import stopwords
import spacy
nlp = spacy.load("en", disable=['parser', 'tagger', 'ner'])
stops = stopwords.words("english")
punctuations = string.punctuation
for p in punctuations:
    stops.append(p)


class Features():
    def __init__(self, dirName):
        self.bugPosts = pd.read_csv(dirName+"bugPosts.csv")
        self.posts = pd.read_csv(dirName+"posts.csv")
        del self.bugPosts['AcceptedAnswerId'], self.bugPosts['ClosedDate'], self.bugPosts['ParentId_q']
        self.votes = pd.read_csv(dirName+'votes.csv')
        self.posts = self.posts[self.posts.PostTypeId == 1] # only questions
        self.bugEdits = pd.read_csv(dirName+"bugEdits.csv")
        self.edits = pd.read_csv(dirName+"postHistory.csv")
        self.process()
        
    def process(self):
        df = self.getTotalVotes()
        df = self.getAllPostFea(df)
        df = self.getEdits(df)
        df = self.getTextFeatures(df)
        df.to_csv(dirName + 'bugReportFeaturess.csv')
        
    def getTotalVotes(self):
        '''getting the total number of 
        (bugs, comments, answers, upvotes, downvotes, favorites, scores of bug reports) reported by the reporter'''
        totalBugs = self.bugPosts.groupby('OwnerUserId', as_index=False).aggregate({
                                         'Id_q':"count", 'CommentCount_q': "sum",
                                         'AnswerCount': 'sum', 'UpVotes_q':'sum', 
                                         'DownVotes_q':'sum', 'FavoriteCount':'sum', 
                                         'Score_q':'sum','ViewCount':'sum'})
        totalBugs.columns = ['OwnerUserId', 'prevBugs', 'prevComments', 'prevAnswerCount',
                             'prevUpvotes', 'prevDownvotes', 'prevFavourites',
                             'prevScore', 'prevViewcount'] 
                    
        df = pd.merge(self.bugPosts, totalBugs, how='left', on= 'OwnerUserId')
        return df
        

    def attachVotes(self, df):
        votes = self.votes[['PostId', 'VoteTypeId', 'BountyAmount']]
        votes['VoteTypeId'] = pd.to_numeric(votes.VoteTypeId)
        votes['PostId'] = pd.to_numeric(votes.PostId)
        
        # upvotes
        upvotes = votes[votes['VoteTypeId'] == 2]
        del upvotes['BountyAmount']
        uv = upvotes.groupby('PostId', as_index=False).count()
        uv.columns = ['Id', 'UpVotes']
        df = pd.merge(df, uv, how='left', on= 'Id')

        #downvotes
        downvotes = votes[votes['VoteTypeId'] == 3]
        del downvotes['BountyAmount']
        dv = downvotes.groupby('PostId', as_index=False).count()
        dv.columns = ['Id', 'DownVotes']
        df = pd.merge(df, dv, how='left', on='Id')
        return df

    # features of all posts reported by the user
    def getAllPostFea(self, df):
        self.posts = self.attachVotes(self.posts)
        totalques = self.posts.groupby('OwnerUserId', as_index=False).aggregate({'Id':"count", 'CommentCount': "sum",
                                    'AnswerCount': 'sum', 'UpVotes':'sum', 'DownVotes':'sum', 'FavoriteCount':'sum', 'Score':'sum', 
                                    'ViewCount':'sum'})
        totalques.columns = ['OwnerUserId', 'allPrevBugs', 'allPrevComments', 'allPrevAnswerCount', 'allPrevUpvotes', 'allPrevDownvotes', 'allPrevFavourites',
                             'allPrevScore', 'allPrevViewcount']
        df = pd.merge(df, totalques, how='inner', on= 'OwnerUserId')
        return df
        
    def getEdits(self, df):
        # getting all the
        be = self.bugEdits.groupby('EditorsId', as_index=False)['EditId'].count()
        be.columns = ['OwnerUserId', 'totalEditsOnBugs']
        df = pd.merge(df, be, how='left')
        
        ed = self.edits.groupby('UserId', as_index=False)['Id'].count()
        ed.columns = ['OwnerUserId', 'totalEditsOnAllQues']
        df = pd.merge(df, ed, how='left')
        return df

    def normalize(self, comment):
        comment = str(comment)
        soup = BeautifulSoup(comment)
        comment = (soup.get_text())
        comment = comment.lower()
        comment = nlp(comment) # This has been tokenized  + formatting added (\n\r)
        lemmatized = list()
        for word in comment: # comment is also called as token
            lemma = word.lemma_.strip() 
            if lemma:
                if (lemma not in stops) and not (word.like_email or word.like_url or word.like_num):
                    lemmatized.append(lemma)
        return lemmatized
    
    def containsImg(self, text):
        if ('<img src' in text) or ('://img' in text) or (".imageshack." in text) or ("imgur.com" in text):
            return 1
        else:
            return 0

    #applying keyword matching to get environment info        
    def containsEnv(self, text):
        text = text.lower()
        if (' ubuntu ' in text) or (" windows " in text) or (" linux " in text) or (" android " in text) or (" ios " in text):
            return 1
        else:
            return 0
            
    def entropy2(self, text, base=None):
      labels = text.split()
      n_labels = len(labels)
      if n_labels <= 1:
        return 0
      value,counts = np.unique(labels, return_counts=True)
      probs = counts / n_labels
      n_classes = np.count_nonzero(probs)
      if n_classes <= 1:
        return 0
      ent = 0.
      # Compute entropy
      base = e if base is None else base
      for i in probs:
        ent -= i * log(i, base)
      return ent
    
    def getDiff(self, df):
        final = df['finalLemma']
        init = df['initLemma']
        addition = set(final).difference(set(init))
        return addition
        
 
    def remFormat(self, comment):
        comment = str(comment)
        soup = BeautifulSoup(comment)
        return (soup.get_text())
    
    def getTextFeatures(self, df):
        initTitle = self.bugEdits[['EditText', 'Id_q']][(self.bugEdits.EditType == 1)]
        initBody = self.bugEdits[['EditText', 'Id_q']][(self.bugEdits.EditType == 2)]
        init = pd.merge(initTitle,initBody, how='outer',on='Id_q', suffixes= ['_t', '_b'])
        init['initText'] = init['EditText_t'] + '\n' + init['EditText_b']
        
        df['finalText'] = df['Title'] + '\n'+ df['Body_q']
        
        #removing the html tags
        df = pd.merge(df, init[['initText', 'Id_q']], on='Id_q', how = 'inner').drop_duplicates()
        
        df['initLemma'] = df['initText'].apply(self.normalize)
        df['finalLemma'] = df['finalText'].apply(self.normalize)
        tmp = df.apply(self.getDiff, axis = 1)
        
        
        df['diffWords'] = tmp.apply(lambda strList: ' '.join(strList))
    
        df['diffWordLength'] = tmp.apply(lambda strList: len(strList))
    
        df['initText'] = df.initText.apply(lambda comment: BeautifulSoup(str(comment)).get_text())
        df['finalText'] = df.finalText.apply(lambda comment: BeautifulSoup(str(comment)).get_text())
        
        df['finalFlesch'] = df.finalText.apply(lambda text: textstat.flesch_reading_ease(text))
        df['finalcontainsImg'] = df.finalText.apply(self.containsImg)
        df['finalcontainsEnv'] = df.finalText.apply(self.containsEnv)
        df['finalEntropy'] = df['finalText'].apply(self.entropy2)
        df['initEntropy'] = df['initText'].apply(self.entropy2)
    
        # Label
        df.loc[df['Tags_q'].str.contains('status-completed'), 'Label'] = 1
        df = df.fillna(0)
        return df


dirName = "/Users/aadityabhatia/All Backup/all_sail_lab_TSE_stuff/StackExchangeResearch/data_new/"
Features(dirName)