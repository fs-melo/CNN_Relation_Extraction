#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 21:25:35 2017

@author: fabricio

Preprocess  [Lin et al., 2016] dataset

"""





folder_files = "/Documentos/Datasets/data_LI_2016/"
filenames = [folder_files+"test.txt"] #, folder_files+"test.txt"]
outputFilePath = folder_files + "clean_test.txt" #+ str(filenames)

#def process_freebase(filenames=filenames, pretrained_word_embedding=pretrained_word_embedding,
#                      embeddings_path='embeddings.npz', vocab_path='map.json', **params):
    #s1=[s.index(w) for w in s]
x=[]
print('Aguarde... Lendo arquivos...')
for fname in filenames:
    with open(fname) as infile:
        for line in infile:
            x.append(line) #x=np.append(x,line,axis=0) #
x = [s.strip().split('\t') for s in x]
for row in x:
    e1_name=row[2]
    e2_name=row[3]
    words = row[5].split(' ')
    for (i, subword) in enumerate(words):
        if (subword == e1_name): 
            pos1=str(i)
        if (subword == e2_name): 
            pos2=str(i)
    row[0] = row[4]
    row[1] = pos1
    row[2] = pos2
    row[5] = row[5].replace(' ###END###', '')
    del row[3]
    del row[3]
#file, format (relation, e1_position, e2_position, sentence).

print('Aguarde... Salvando arquivos...')
thefile = open(outputFilePath, 'w')
for row in x:
    linha=''
    for elem in row:
        linha = str(linha) 
        if (len(linha)>1):
            linha = linha +"\t"
        linha = linha +str(elem)
    thefile.write("%s\n" % linha)
print("Processo concluído!\nArquivo criado:\n%s" % thefile.name)
