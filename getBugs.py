# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 13:11:09 2019

@author: bhati
"""
import pandas as pd
import numpy as np

dirName = "C:/Users/bhati/Desktop/Stack Exchange Research/data/"
users = pd.read_csv(dirName+'users.csv')
votes = pd.read_csv(dirName+'votes.csv')
posts = pd.read_csv(dirName+'posts.csv') 

# pandas date time
posts.CreationDate = pd.to_datetime(posts['CreationDate'])

# removing the unwanted pandas columns
del posts['LastActivityDate'], posts['LastEditDate'], posts['LastEditorDisplayName'], posts['LastEditorUserId'], posts['OwnerDisplayName'], posts['CommunityOwnedDate'], posts['PostTypeId']

''' attaching upvotes and downvotes'''
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
    #bounties
    bounty = votes[votes['VoteTypeId'] == 8]
    del bounty['VoteTypeId']
    bounty.columns = ['Id','BountyAmount']
    df = pd.merge(df, bounty, how='left', on='Id')
    return df

posts = attachVotes(votes, posts)

bugPosts = posts[posts.Tags.str.contains('bug')]

posts['feature-request'] = posts['Tags'].str.contains("feature-request")

def getDelay(answers, df):
    qa = pd.merge(df, ans, how='left', left_on='AcceptedAnswerId', right_on='Id', suffixes=['_q', '_aa'])
    qa['Delay'] = (qa['CreationDate_aa']-qa['CreationDate_q']).astype('timedelta64[m]')
#    qa = qa[pd.notnull(qa["Delay"])]
    return qa

# getting the answers for time to get the answer
answers = posts[posts['PostTypeId'] == 2]
answers = answers.dropna(axis='columns', how='all')

bugPosts = getDelay(answers, bugPosts)

#attach Se reputation
users = users[['Id', 'Reputation']]
users.columns = ['OwnerUserId', 'Reputation']
bugPosts = pd.merge(bugPosts, users, how='left', on='OwnerUserId')

#finding no.of reported bugs
bp = bugPosts[['OwnerUserId', 'Id_q']]
totalBugs = bp.groupby('OwnerUserId', as_index=False)['Id_q'].count()
totalBugs.columns = ['OwnerUserId', 'CountBugs']
bugPosts = pd.merge(bugPosts, totalBugs, how='left', on= 'OwnerUserId')


#taking logarithmic trasnformations
bugPosts['CommentCount_l'] = (bugPosts['CommentCount_q']+1).apply(np.log10)

bugPosts['DownVotes_l'] = (bugPosts['DownVotes_q']+1).apply(np.log10) 

bugPosts['FavoriteCount_l'] = (bugPosts['FavoriteCount']+1).apply(np.log10)

bugPosts['UpVotes_l'] = (bugPosts['UpVotes_q']+1).apply(np.log10)

bugPosts['DownVotes_l'] = (bugPosts['DownVotes_q']+1).apply(np.log10)

bugPosts['Reputation_l'] = (bugPosts['Reputation']+1).apply(np.log10)

bugPosts['AnswerCount_l'] = (bugPosts['AnswerCount']+1).apply(np.log10)

bugPosts['Delay_l'] = (bugPosts['Delay']+1).apply(np.log10)

bugPosts = bugPosts.replace([np.inf, -np.inf], np.nan)
bugPosts = bugPosts.fillna(0)


bugPosts = bugPosts.drop_duplicates()

bugPosts.to_csv(dirName+"bugPosts.csv", index=False, encoding='utf-8')
