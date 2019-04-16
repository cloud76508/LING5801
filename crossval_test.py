# -*- coding: utf-8 -*-
"""
Created on Mon Apr 09 13:34:47 2018

@author: User
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn import svm

clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, iris.data, iris.target, cv=5)