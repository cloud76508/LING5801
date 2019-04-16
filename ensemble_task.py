# -*- coding: utf-8 -*-
"""
Created on Mon Apr 09 13:25:39 2018

@author: User
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn import svm
from sklearn import linear_model
from preprocessing import prepareData

#iris = datasets.load_iris()
#X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)

reviewData = prepareData()

X_train, X_test, y_train, y_test = train_test_split(reviewData['X'], reviewData['Y'], test_size=0.1, random_state=10)

parameterC = [0.01, 0.1, 1, 10, 100]
crossScore = []

for c in parameterC:
    clf = svm.SVC(kernel='linear', C=c)
    score = cross_val_score(clf, X_train, y_train, cv=10)
    crossScore.append(score.mean())

for i, sent in enumerate(crossScore):
    if sent == max(crossScore):
        optimalC = parameterC[i]
        
optimalClf = svm.SVC(kernel = 'linear', C = optimalC).fit(X_train, y_train)
trainScore = optimalClf.score(X_train,y_train)
testScore =  optimalClf.score(X_test,y_test)
print optimalC
print trainScore
print testScore

clfEnsemble = linear_model.SGDClassifier()
clfEnsemble1 = clfEnsemble.fit(X_train, y_train)
testScoreEnsemble = clfEnsemble1.score(X_test, y_test)
print testScoreEnsemble
