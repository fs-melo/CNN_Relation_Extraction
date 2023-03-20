#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 03:59:04 2017

@author: fabricio
Using accuracy metric
"""

import numpy as np
#import matplotlib.pyplot as plt
import gzip
import sys
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers import Dropout, GaussianNoise
from keras.layers import Embedding, concatenate
from keras.layers import Convolution1D, GlobalMaxPooling1D #, MaxPooling1D
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
#import scikitplot as skplt
if (sys.version_info > (3, 0)):
    import pickle as pkl
else: #Python 2.7 imports
    import cPickle as pkl

seed = 1337
np.random.seed(seed) # # fix random seed for reproducibility
# Model Hyperparameters
filter_sizes = 3 #(3, 8)
num_filters = 100
max_num_words = 20000 #in vocabulare
position_dims = 50

# Training parameters
batch_size = 64
num_epochs = 1
dropout_rate = 0 #0.5
std_noise = 0 #0.3

print("Load dataset")
f = gzip.open('join_data.pkl.gz', 'rb')
data = pkl.load(f)
f.close()

sentence_train, yLabel_train, positionMatrix1_train, positionMatrix2_train = data['train']
sentence_test, yLabel_test, positionMatrix1_test, positionMatrix2_test = data['test']
del data
max_position = max(np.max(positionMatrix1_train), np.max(positionMatrix2_train))+1
n_out = yLabel_train.shape[1]
max_sequence_length = sentence_train.shape[1]

print("Loading embeddings")
embedding_matrix = np.load(open('embeddings.npz', 'rb'))

# The great trick !!!
Y = np.argmax(yLabel_train,axis=-1)
print("Training the model")

# create model
words_input = Input(shape=(max_sequence_length,), dtype='int32', name='words_input')
words = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], 
                  weights=[embedding_matrix], trainable=False)(words_input)

distance1_input = Input(shape=(max_sequence_length,), dtype='int32', name='distance1_input')
distance1 = Embedding(max_position, position_dims)(distance1_input)

distance2_input = Input(shape=(max_sequence_length,), dtype='int32', name='distance2_input')
distance2 = Embedding(max_position, position_dims)(distance2_input)

output = concatenate([words, distance1, distance2])

output = GaussianNoise(std_noise)(output)

output = Convolution1D(filters=num_filters,
                        kernel_size=filter_sizes,
                        padding='valid',
                        activation='relu',
                        strides=1)(output)
#    output = MaxPooling1D(5)(output)
#    output = Convolution1D(num_filters, filter_sizes, activation='relu')(output)
output = GlobalMaxPooling1D()(output)
output = Dropout(dropout_rate)(output)
output = Dense(n_out, activation='softmax')(output)
model = Model(inputs=[words_input, distance1_input, distance2_input], outputs=[output])
# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# define 10-fold cross validation test harness
k = 10
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

j=1
cvscores = []
for train, test in skf.split(sentence_train,Y):

	# Fit the model
    print("train k:",j,"/",k)      
    model.fit([sentence_train[train], positionMatrix1_train[train],positionMatrix2_train[train]],
              yLabel_train[train], epochs=num_epochs, batch_size=batch_size, verbose=1)
	# evaluate the model
    scores = model.evaluate([sentence_train[test], positionMatrix1_train[test],positionMatrix2_train[test]],
                            yLabel_train[test], verbose=0)
    print("val_acc: %.2f%%" % (scores[1]*100)) #model.metrics_names[1]
    cvscores.append(scores[1] * 100)
    j = j+1

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

del Y
predicted_yLabel=model.predict([sentence_test, positionMatrix1_test, positionMatrix2_test], 
                     batch_size=None, verbose=0, steps=None)

predicted_yLabel = np.argmax(predicted_yLabel,axis=-1)
yLabel_test = np.argmax(yLabel_test,axis=-1)

#plot the report

rep = classification_report(yLabel_test, predicted_yLabel)
rep = rep.splitlines()

cl = rep[2:-2]
cl = [ln.replace('class ','').split() for ln in cl]
last = rep[-1]
numbers = last[11:].split()

print('\hline')
print('&Precision & Recall & F1-score & Support \\\\')
print('\hline')
for ln in cl:
    print('class '+' & '.join(ln)+r'\\')
print('\hline')
print('avg/total & '+' & '.join(numbers)+r'\\')
print('\hline')

#print("Classification report: \n", (classification_report(yLabel_test, predicted_yLabel)))

#matriz de confusão ficou muito carregada
#skplt.metrics.plot_confusion_matrix(yLabel_test, predicted_yLabel, normalize=True,figsize=(20,20))
