#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 03:59:04 2017

@author: fabricio
"""

import numpy as np
from sklearn.model_selection import train_test_split
import gzip
import sys
if (sys.version_info > (3, 0)):
    import pickle as pkl
else: #Python 2.7 imports
    import cPickle as pkl

outputFilePath = "join_data.pkl.gz"
seed = 1337
np.random.seed(seed) # # fix random seed for reproducibility

print("Load dataset")
f = gzip.open('freebase-relations.pkl.gz', 'rb')
data = pkl.load(f)
f.close()

print("Load Train data")
yTrain, sentenceTrain, positionTrain1, positionTrain2 = data['train_set']
print("Load Test data")
yTest, sentenceTest, positionTest1, positionTest2  = data['test_set']
del data

print("Concatenate data")
sentence = np.concatenate((sentenceTrain, sentenceTest), axis=0)

del sentenceTrain, sentenceTest
yLabel = np.concatenate((yTrain, yTest), axis=0)
del yTest, yTrain
positionMatrix1 = np.concatenate((positionTrain1, positionTest1), axis=0)
del positionTrain1, positionTest1
positionMatrix2 = np.concatenate((positionTrain2, positionTest2), axis=0)
del positionTrain2, positionTest2

# O pulo do gato !!!
Y = np.argmax(yLabel,axis=-1)
unique, counts = np.unique(Y, return_counts=True)
"""counts: 
[     1   9284      3      1     20 579428  10279    212    766   9540
    247      1      6  11769     71      7      1  12169     19      4
   9284   8289    837      4     19      2   2609      2     50     33
      6     10    133      4      8    321   4427   1156      8      2
      7    169    309      4     22  78762     10    697      7     59
      1     20    203     33      6    485      3    707]
"""
#define freq mínima por classe
freq_min = 20
del_index = np.where( counts < freq_min )
clean_labels = np.delete(unique,del_index)
mask = np.isin(Y,clean_labels)
del unique,counts,clean_labels,Y,del_index
sentence = sentence[mask]
yLabel = yLabel[mask]
positionMatrix1 = positionMatrix1[mask]
positionMatrix2 = positionMatrix2[mask]
del mask

print("Splitting train/test dataset...")
sentence_train, sentence_test, yLabel_train, yLabel_test, positionMatrix1_train, positionMatrix1_test, positionMatrix2_train, positionMatrix2_test = train_test_split(sentence, yLabel, positionMatrix1,
                                                                      positionMatrix2, train_size=0.7,
                                                                      stratify=yLabel)
del sentence, yLabel, positionMatrix1, positionMatrix2

data = {'train': [sentence_train, yLabel_train, positionMatrix1_train, positionMatrix2_train],
        'test': [sentence_test, yLabel_test, positionMatrix1_test, positionMatrix2_test]}

del sentence_train, sentence_test, yLabel_train, yLabel_test, positionMatrix1_train, positionMatrix1_test, positionMatrix2_train, positionMatrix2_test

f = gzip.open(outputFilePath, 'wb')
pkl.dump(data, f)
f.close()

del data, outputFilePath, freq_min

print("Data stored in pkl folder")
