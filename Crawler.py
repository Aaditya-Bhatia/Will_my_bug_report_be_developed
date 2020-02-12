# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 13:11:09 2019

@author: bhati

Info: This module extracts the required data from Meta Stack Exchange data dump
"""

import pandas as pd
import numpy as np

class Crawler:
    
    def __init__(self, dirName):
        self.users = pd.read_csv(dirName+'users.csv')
        self.votes = pd.read_csv(dirName+'votes.csv')
        self.posts = pd.read_csv(dirName+'posts.csv') 
        self.edits = pd.read_csv(dirName+'postHistory.csv')       
        
        # initializing the module from here
        bugPosts = self.process()
        self.getBugEdits(bugPosts)
    
    def getBugEdits(self, bugPosts):

        self.edits.columns = ['EditComment', 'EditCreationDate', 'EditId', 'EditType', 'Id_q','EditGUID', 'EditText', 'EditorsName','EditorsId']        
        self.edits = self.edits[self.edits['EditorsId'] > -1] #65.2K
        closingEdits = self.edits[(self.edits['EditType'] == 10)]
    
        closedBugs = pd.merge(bugPosts, closingEdits, on = 'Id_q', how='inner')

        # ONly considering title body and tag edits!!!
        self.edits = self.edits[(self.edits['EditType'] == 1) | (self.edits['EditType'] == 2) | (self.edits['EditType'] == 3)
                    | (self.edits['EditType'] == 4) | (self.edits['EditType'] == 5) | (self.edits['EditType'] == 6)]
    
        bugs = bugPosts[~bugPosts['Id_q'].isin(closedBugs['Id_q'])]
        
        bugEdits = pd.merge(bugs, self.edits, on = 'Id_q', how = 'inner')
        
        bugEdits.to_csv(dirName+"bugEdits.csv", index = False, encoding = 'utf-8')
    
    
    def process(self):
        # converting to pandas date time
        self.posts.CreationDate = pd.to_datetime(self.posts['CreationDate'])
                    
        posts = self.attachVotes(self.posts)
        posts = self.remCols(posts)
        bugPosts = posts[posts.Tags.str.contains('bug')]
        answers = posts[posts['PostTypeId'] == 2].dropna(axis='columns', how='all')      
        bugPosts = self.getDelay(answers, bugPosts)        
        bugPosts = self.getReputation(bugPosts)
        bugPosts = self.getReportedBugCount(bugPosts)
        bugPosts = self.getLogTransformation(bugPosts)    

        bugPosts = bugPosts.drop_duplicates()
        bugPosts.to_csv(dirName+"bugPosts.csv", index=False, encoding='utf-8')
        return bugPosts
    
    def remCols(self, df):
        del df['LastActivityDate'], df['LastEditDate'], df['LastEditorDisplayName'], df['LastEditorUserId'], df['OwnerDisplayName'], df['CommunityOwnedDate']
        return df
      
    def getReportedBugCount(self, df):
        bp = df[['OwnerUserId', 'Id_q']]
        totalBugs = bp.groupby('OwnerUserId', as_index=False)['Id_q'].count()
        totalBugs.columns = ['OwnerUserId', 'CountBugs']
        df = pd.merge(df, totalBugs, how='left', on= 'OwnerUserId')
        return df
    
    def getReputation(self, df):
        users = self.users[['Id', 'Reputation']]
        users.columns = ['OwnerUserId', 'Reputation']
        df = pd.merge(df, users, how='left', on='OwnerUserId')
        return df


    def getLogTransformation(self, df):
        df['CommentCount_l'] = (df['CommentCount_q']+1).apply(np.log10)        
        df['DownVotes_l'] = (df['DownVotes_q']+1).apply(np.log10)         
        df['FavoriteCount_l'] = (df['FavoriteCount']+1).apply(np.log10)        
        df['UpVotes_l'] = (df['UpVotes_q']+1).apply(np.log10)        
        df['DownVotes_l'] = (df['DownVotes_q']+1).apply(np.log10)        
        df['Reputation_l'] = (df['Reputation']+1).apply(np.log10)        
        df['AnswerCount_l'] = (df['AnswerCount']+1).apply(np.log10)        
        df['Delay_l'] = (df['Delay']+1).apply(np.log10)        
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        return df
    
    
    def getDelay(self, answers, df):
        del answers['OwnerUserId']
        qa = pd.merge(df, answers, how='left', left_on='AcceptedAnswerId', right_on='Id', suffixes=['_q', '_aa'])
        qa['Delay'] = (qa['CreationDate_aa']-qa['CreationDate_q']).astype('timedelta64[m]')
        return qa

    ''' attaching upvotes and downvotes'''
    def attachVotes(self, df):
        self.votes = self.votes[['PostId', 'VoteTypeId', 'BountyAmount']]
        self.votes['VoteTypeId'] = pd.to_numeric(self.votes.VoteTypeId)
        self.votes['PostId'] = pd.to_numeric(self.votes.PostId)
        
        # getting upvotes
        upvotes = self.votes[self.votes['VoteTypeId'] == 2]
        del upvotes['BountyAmount']
        uv = upvotes.groupby('PostId', as_index=False).count()
        uv.columns = ['Id', 'UpVotes']
        df = pd.merge(df, uv, how='left', on= 'Id')
    
        #getting downvotes
        downvotes = self.votes[self.votes['VoteTypeId'] == 3]
        del downvotes['BountyAmount']
        dv = downvotes.groupby('PostId', as_index=False).count()
        dv.columns = ['Id', 'DownVotes']
        df = pd.merge(df, dv, how='left', on='Id')
        
        #bounties
        bounty = self.votes[self.votes['VoteTypeId'] == 8]
        del bounty['VoteTypeId']
        bounty.columns = ['Id','BountyAmount']
        df = pd.merge(df, bounty, how='left', on='Id')
        
        df['Tags'] = df.Tags.fillna("")
        
        return df
        

dirName = "/Users/aadityabhatia/All Backup/all_sail_lab_TSE_stuff/StackExchangeResearch/data_new/"
Crawler(dirName=dirName)