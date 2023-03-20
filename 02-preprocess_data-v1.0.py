#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 11:42:18 2017

@author: fabricio
"""
import sys
import numpy as np
import gzip
from keras.preprocessing.text import Tokenizer
import json
if (sys.version_info > (3, 0)):
    import pickle as pkl
else: #Python 2.7 imports
    import cPickle as pkl

np.random.seed(1337) # for reproducibility

outputFilePath = 'freebase-relations.pkl.gz'

folder_files = "/Datasets/data_LI_2016/"
filenames = [folder_files+"clean_train.txt", folder_files+"clean_test.txt"]
pretrained_word_embedding = '/Datasets/word2vec.6B.100d.txt'
max_num_words = 250000 # max_num_words palavras mais frequentes
maxSentenceLen = 50
   
def create_embeddings(filenames=filenames, pretrained_word_embedding=pretrained_word_embedding,
                      embeddings_path='embeddings.npz', vocab_path='map.json', **params):
    print('Creating embeddings...')
    sentences=[]
    labels=[]
    for fname in filenames:
        with open(fname) as infile:
            for line in infile:
                splits = line.strip().split('\t') #combined.append(line) 
                sentence = splits[3]
                label = splits[0]
                labels.append(label)
                sentences.append(sentence)
    a=list(set(labels))
    mapLabels = dict((c, i) for i, c in enumerate(a))
    for word,i in mapLabels.items():
        l=[0]*len(mapLabels)
        l[i]=1
        mapLabels[word] = l
    with open('mapLabels.json', 'w') as f:
        f.write(json.dumps(mapLabels))
    tokenizer = Tokenizer(num_words = max_num_words, filters='\t\n')
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index
    with open(vocab_path, 'w') as f:
        f.write(json.dumps(word_index))
    if pretrained_word_embedding is not None:
        # load the whole embedding into memory
        embeddings_index = dict()
        f = open(pretrained_word_embedding)
        for line in f:
            	values = line.split()
            	word = values[0]
            	coefs = np.asarray(values[1:], dtype='float32')
            	embeddings_index[word] = coefs
        f.close()
        print('Loaded %s word vectors.' % len(embeddings_index))
        # create a weight matrix for words in training docs
        num_words = min(max_num_words,len(word_index))
        embeddings_dims = len(coefs)
        sd=1/np.sqrt(embeddings_dims)
        embedding_matrix = np.random.normal(0,scale=sd,size=[num_words+1,embeddings_dims]) #np.zeros((num_words, embeddings_dims))
        for word, i in word_index.items():
            if i >= num_words:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
            		embedding_matrix[i] = embedding_vector
        np.save(open(embeddings_path, 'wb'), embedding_matrix)
        print("Created vocab. with "+str(embedding_matrix.shape[0]), " words.")
        del embedding_matrix
    return tokenizer        

def load_vocab(vocab_path='map.json'):
    """
    Load word -> index and index -> word mappings
    :param vocab_path: where the word-index map is saved
    :return: word2idx, idx2word
    """
 
    print('Saving vocab...')
    with open(vocab_path, 'r') as f:
        data = json.loads(f.read())
    word2idx = data
    idx2word = dict([(v, k) for k, v in data.items()])
    return word2idx, idx2word

    
def load_mapLabels(mapLabels_path='mapLabels.json'):
    """
    Load word -> index and index -> word mappings
    :param vocab_path: where the word-index map is saved
    :return: word2idx, idx2word
    """
 
    with open(mapLabels_path, 'r') as f:
        data = json.loads(f.read())
    mapLabels = data
    return mapLabels



def createMatrices(file, word2Idx, mapLabels, maxSentenceLen=100):
    """Creates matrices for the events and sentence for the given file"""
    
    print('Creating matrices for the sentence of %s ...' % file)
    distanceMapping = {'PADDING': 0, 'LowerMin': 1, 'GreaterMax': 2}
    minDistance = -40
    maxDistance = 40
    for dis in range(minDistance,maxDistance+1):
        distanceMapping[dis] = len(distanceMapping)

    labels = []
    positionMatrix1 = []
    positionMatrix2 = []
    tokenMatrix = []
    
    for line in open(file):
        splits = line.strip().split('\t')
        
        label = mapLabels[splits[0]]
        pos1 = splits[1]
        pos2 = splits[2]
        sentence = splits[3].lower()
        tokens = sentence.split(" ")
        
        tokenIds = np.zeros(maxSentenceLen)
        positionValues1 = np.zeros(maxSentenceLen)
        positionValues2 = np.zeros(maxSentenceLen)
        
        for idx in range(0, min(maxSentenceLen, len(tokens))):
            tokenIds[idx] = word2Idx.word_index[tokens[idx]]
            
            distance1 = idx - int(pos1)
            distance2 = idx - int(pos2)
            
            if distance1 in distanceMapping:
                positionValues1[idx] = distanceMapping[distance1]
            elif distance1 <= minDistance:
                positionValues1[idx] = distanceMapping['LowerMin']
            else:
                positionValues1[idx] = distanceMapping['GreaterMax']
                
            if distance2 in distanceMapping:
                positionValues2[idx] = distanceMapping[distance2]
            elif distance2 <= minDistance:
                positionValues2[idx] = distanceMapping['LowerMin']
            else:
                positionValues2[idx] = distanceMapping['GreaterMax']
            
        tokenMatrix.append(tokenIds)
        positionMatrix1.append(positionValues1)
        positionMatrix2.append(positionValues2)
        labels.append(label)
            
    return np.array(labels, dtype='int32'), np.array(tokenMatrix, dtype='int32'), np.array(positionMatrix1, dtype='int32'), np.array(positionMatrix2, dtype='int32'),
    
######
    
word2Idx = create_embeddings(filenames=filenames, pretrained_word_embedding=pretrained_word_embedding)
# :: Create token matrix ::
mapLabels = load_mapLabels()
train_set = createMatrices(filenames[0], word2Idx, mapLabels, 50)
test_set = createMatrices(filenames[1], word2Idx, mapLabels, 50)

del word2Idx
del mapLabels

data = {'train_set': train_set, 'test_set': test_set}

f = gzip.open(outputFilePath, 'wb')
pkl.dump(data, f)
f.close()

del train_set, test_set

print("Data stored in pkl folder")
