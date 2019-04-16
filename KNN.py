# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 18:17:53 2018

@author: ananya
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from preprocessing import prepareData


reviewData = prepareData()

X_train, X_test, y_train, y_test = train_test_split(reviewData['X'], reviewData['Y'], test_size=0.1, random_state=7)

parameterC = [3, 5, 7, 10, 15]
crossScore = []

for c in parameterC:
    clf = KNeighborsClassifier(n_neighbors=c)
    score = cross_val_score(clf, X_train, y_train, cv=10)
    crossScore.append(score.mean())

for i, sent in enumerate(crossScore):
    if sent == max(crossScore):
        optimalC = parameterC[i]
        
optimalClf = KNeighborsClassifier(n_neighbors = optimalC,metric='hamming').fit(X_train, y_train)
trainScore = optimalClf.score(X_train,y_train)
testScore =  optimalClf.score(X_test,y_test)
print optimalC
print trainScore
print testScore

