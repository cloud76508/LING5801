# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 17:03:14 2018

@author: Hsiang-Han Chen, chen4646@umn.edu
"""
from __future__ import division         
import pandas as pd
import nltk
from nltk import word_tokenize
import numpy as np

def prepareData():
    df = pd.read_csv('Yelp Reviews.csv')
    #saved_column = df.column_name #you can also 
    dataNeg = df[0:50]
    classNeg = dataNeg.Class
    reviewNeg = dataNeg.Review
    
    df = pd.read_csv('ling5106_train_dataset.csv')
    #saved_column = df.column_name #you can also 
    dataPos = df[0:50]
    classPos = dataPos.Class
    reviewPos = dataPos.Review
    
    df = pd.read_csv('Negative Food Words.csv')
    wordNeg = df.Mediocre
    
    df = pd.read_csv('Negative service words.csv')
    wordNegServ = df.Negative
    
    df = pd.read_csv('Positive Food Words.csv')
    wordPos = df.Positive
    
    keyWord = wordNeg.append(wordPos)
    keyWord = keyWord.append(wordNegServ)
    #keyWord = wordNeg.append(wordNegServ)
    
    tempWord = []
    for sen in reviewNeg:
        sen = sen.replace("\xc2\xa0", " ") #clear non-breaking space in UTF-8 
        tempWord.append(word_tokenize(sen))
    
    i= 0
    for sen in reviewNeg:
        tempFeature = np.array([])
        for word in keyWord:
            wordLower =  word.lower()
            if word in sen:
                tempFeature = np.append(tempFeature,[1], axis = 0)
            elif wordLower in sen:
                tempFeature = np.append(tempFeature,[1], axis = 0)
            else:
                tempFeature = np.append(tempFeature,[0], axis = 0)
        
        tempFeature = np.reshape(tempFeature, (1, tempFeature.size))
        
        if i == 0:
            featureNeg = tempFeature
        else:
            featureNeg = np.concatenate((featureNeg, tempFeature), axis = 0)
        i=i+1
    
    
    i= 0
    for sen in reviewPos:
        tempFeature = np.array([])
        for word in keyWord:
            wordLower =  word.lower()
            if word in sen:
                tempFeature = np.append(tempFeature,[1], axis = 0)
            elif wordLower in sen:
                tempFeature = np.append(tempFeature,[1], axis = 0)
            else:
                tempFeature = np.append(tempFeature,[0], axis = 0)
        
        tempFeature = np.reshape(tempFeature, (1, tempFeature.size))
        
        if i == 0:
            featurePos = tempFeature
        else:
            featurePos = np.concatenate((featurePos, tempFeature), axis = 0)
        i=i+1
    
    
    
    X_train = np.concatenate((featureNeg[:40],featurePos[:40]), axis = 0)
    X_test = np.concatenate((featureNeg[40:],featurePos[40:]), axis = 0)
    y_train = np.concatenate( (np.array(classNeg[:40]), np.array(classPos[:40])))
    y_test = np.concatenate( (np.array(classNeg[40:]), np.array(classPos[40:])))
    
    reviewData = {}
    reviewData['X'] = np.concatenate((featureNeg,featurePos), axis = 0)
    reviewData['Y'] = np.concatenate( (np.array(classNeg), np.array(classPos)))
        
    return reviewData