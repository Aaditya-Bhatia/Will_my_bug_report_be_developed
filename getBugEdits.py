# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 20:17:11 2019

@author: bhati
"""

import pandas as pd

dirName = "C:/Users/bhati/OneDrive/Desktop/Stack Exchange Research/data/"

edits = pd.read_csv(dirName+'postHistory.csv') # 64K
bugPosts = pd.read_csv(dirName+'bugPosts.csv')

edits.columns = ['EditComment', 'EditCreationDate', 'EditId', 'EditType', 'Id_q','EditGUID', 'EditText', 'EditorsName','EditorsId']

#removing community edits
edits = edits[edits['EditorsId'] > -1] #65.2K
closingEdits = edits[(edits['EditType'] == 10)]

closedBugs = pd.merge(bugPosts, closingEdits, on = 'Id_q', how='inner')
len(closedBugs[closedBugs['Tags'].str.contains("status-")])

closedBugs = closedBugs[['Id_q']].drop_duplicates()
bugs = bugPosts[~bugPosts['Id_q'].isin(closedBugs['Id_q'])]

# ONly considering title body and tag edits!!!
edits = edits[(edits['EditType'] == 1) | (edits['EditType'] == 2) | (edits['EditType'] == 3)
                | (edits['EditType'] == 4) | (edits['EditType'] == 5) | (edits['EditType'] == 6)
                | (edits['EditType'] == 7) | (edits['EditType'] == 8) | (edits['EditType'] == 9)] #Rollbacks

bugEdits = pd.merge(bugs, edits, on = 'Id_q', how = 'inner')
bugEdits.to_csv(dirName+"bugEdits.csv", index = False, encoding = 'utf-8')
