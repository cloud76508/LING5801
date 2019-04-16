# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 22:03:41 2018

@author: ananya
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn.naive_bayes import BernoulliNB
from preprocessing import prepareData


reviewData = prepareData()

X_train, X_test, y_train, y_test = train_test_split(reviewData['X'], reviewData['Y'], test_size=0.1, random_state=6)

parameterC = [0.01, 0.1, 1, 10, 100]
crossScore = []

for c in parameterC:
    clf = BernoulliNB()
    score = cross_val_score(clf, X_train, y_train, cv=10)
    crossScore.append(score.mean())

for i, sent in enumerate(crossScore):
    if sent == max(crossScore):
        optimalC = parameterC[i]
     
optimalClf = BernoulliNB(alpha=1.0, binarize=0.2, class_prior=None, fit_prior=True).fit(X_train, y_train)
trainScore = optimalClf.score(X_train,y_train)
testScore =  optimalClf.score(X_test,y_test)
print optimalC
print trainScore
print testScore