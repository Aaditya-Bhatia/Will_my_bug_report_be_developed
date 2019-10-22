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

# readomg the data
dirName = "C:/Users/bhati/Desktop/Stack Exchange Research/data/"

# get reporter info here
bugPosts = pd.read_csv(dirName+"bugPosts.csv")
posts = pd.read_csv(dirName+"posts.csv")

# removing the unwanted columns
del bugPosts['AcceptedAnswerId'], bugPosts['ClosedDate'], bugPosts['ParentId_q']

'''getting the total number of (bugs, comments, answers, upvotes, downvotes, favorites, scores of bug reports) reported by the reporter'''
totalBugs = bugPosts.groupby('OwnerUserId', as_index=False).aggregate({'Id_q':"count", 'CommentCount_q': "sum",
                            'AnswerCount': 'sum', 'UpVotes_q':'sum', 'DownVotes_q':'sum', 'FavoriteCount':'sum', 'Score_q':'sum',                             'ViewCount':'sum'})
totalBugs.columns = ['OwnerUserId', 'prevBugs', 'prevComments', 'prevAnswerCount', 'prevUpvotes', 'prevDownvotes', 'prevFavourites',
                     'prevScore', 'prevViewcount']
                     
# merge the previously reported features with main bugposts dataframe
df = pd.merge(bugPosts, totalBugs, how='left', on= 'OwnerUserId')

'''
Attaching the upvotes and downvotes
'''
votes = pd.read_csv(dirName+'votes.csv')
posts = posts[posts.PostTypeId == 1] # only questions

def attachVotes(votes, df):
    votes = votes[['PostId', 'VoteTypeId', 'BountyAmount']]
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

# we get the up and downvotes on all posts
posts = attachVotes(votes, posts)

# features of all posts reported by the user
totalques = posts.groupby('OwnerUserId', as_index=False).aggregate({'Id':"count", 'CommentCount': "sum",
                            'AnswerCount': 'sum', 'UpVotes':'sum', 'DownVotes':'sum', 'FavoriteCount':'sum', 'Score':'sum', 
                            'ViewCount':'sum'})
totalques.columns = ['OwnerUserId', 'allPrevBugs', 'allPrevComments', 'allPrevAnswerCount', 'allPrevUpvotes', 'allPrevDownvotes', 'allPrevFavourites',
                     'allPrevScore', 'allPrevViewcount']
df = pd.merge(df, totalques, how='inner', on= 'OwnerUserId')

''' Number of edits the reporter made before '''
bugEdits = pd.read_csv(dirName+"bugEdits.csv")
edits = pd.read_csv(dirName+"postHistory.csv")

# getting all the
be = bugEdits.groupby('EditorsId', as_index=False)['EditId'].count()
be.columns = ['OwnerUserId', 'totalEditsOnBugs']
df = pd.merge(df, be, how='left')

ed = edits.groupby('UserId', as_index=False)['Id'].count()
ed.columns = ['OwnerUserId', 'totalEditsOnAllQues']
df = pd.merge(df, ed, how='left')


'''
Module 2
NLP Text analysis
'''
import string
import pandas as pd
import nltk
#nltk.download()
from nltk.corpus import stopwords
import spacy

nlp = spacy.load("en", disable=['parser', 'tagger', 'ner'])
stops = stopwords.words("english")
punctuations = string.punctuation
for p in punctuations:
    stops.append(p)
    
def normalize(comment):
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

def containsImg(text):
    if ('<img src' in text) or ('://img' in text) or (".imageshack." in text) or ("imgur.com" in text):
        return 1
    else:
        return 0
def containsEnv(text):
    text = text.lower()
    #applying keyword matching to get environment info
    if (' ubuntu ' in text) or (" windows " in text) or (" linux " in text) or (" android " in text) or (" ios " in text):
        return 1
    else:
        return 0
def flesch(text):
    return textstat.flesch_reading_ease(text)

def entropy2(text, base=None):
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


def getDiff(df):
    final = df['finalLemma']
    init = df['initLemma']
    addition = set(final).difference(set(init))
    return addition
    
def getLen(lst):
    return len(lst)

def getJoint(lst):
    return ' '.join(lst)

def remFormat(comment):
    comment = str(comment)
    soup = BeautifulSoup(comment)
    return (soup.get_text())


initTitle = bugEdits[['EditText', 'Id_q']][(bugEdits.EditType == 1)]
initBody = bugEdits[['EditText', 'Id_q']][(bugEdits.EditType == 2)]
init = pd.merge(initTitle,initBody, how='outer',on='Id_q', suffixes= ['_t', '_b'])
init['initText'] = init['EditText_t'] + '\n' + init['EditText_b']

df['finalText'] = df['Title'] + '\n'+ df['Body']

#removing the html tags
df = pd.merge(df, init[['initText', 'Id_q']], on='Id_q', how = 'inner').drop_duplicates()

df['initLemma'] = df['initText'].apply(normalize)
df['finalLemma'] = df['finalText'].apply(normalize)
tmp = df.apply(getDiff, axis = 1)


df['diffWords'] = tmp.apply(getJoint)
df['diffWordLength'] = tmp.apply(getLen)

df['initText'] = df.initText.apply(remFormat)
df['finalText'] = df.finalText.apply(remFormat)

df['finalFlesch'] = df.finalText.apply(flesch)
df['finalcontainsImg'] = df.finalText.apply(containsImg)
df['finalcontainsEnv'] = df.finalText.apply(containsEnv)
df['finalEntropy'] = df['finalText'].apply(entropy2)
df['initEntropy'] = df['initText'].apply(entropy2)


# label
df.loc[df['Tags'].str.contains('status-completed'), 'Label'] = 1
df = df.fillna(0)
df.to_csv(dirName+'allFeatures_1.csv', index=False, encoding = 'utf-8')
