# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 20:15:31 2018

@author: ananya
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from preprocessing import prepareData


reviewData = prepareData()

X_train, X_test, y_train, y_test = train_test_split(reviewData['X'], reviewData['Y'], test_size=0.2, shuffle=True, random_state=1)

mlp = MLPClassifier(hidden_layer_sizes=(10,10,10),solver='sgd',learning_rate_init=0.01)       
mlp.fit(X_train, y_train)
trainScore = mlp.score(X_train,y_train)
testScore =  mlp.score(X_test,y_test)

print trainScore
print testScore
