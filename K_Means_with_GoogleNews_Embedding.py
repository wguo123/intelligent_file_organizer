# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 00:46:21 2019

@author: weig
"""

import numpy as np

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from os import listdir
from os.path import isfile, join
import re
import os
import time
from shutil import copyfile

from sklearn.cluster import KMeans
import gensim.corpora as corpora
from gensim.models import KeyedVectors

import nltk
import warnings
warnings.filterwarnings('ignore')


def retrieve_file_names(path):
    file_name_list = [f for f in listdir(path) if isfile(join(path, f))]
    print("{} files are retrieved.".format(len(file_name_list)))
    print("\nFirst 20 files are: ")
    for x in file_name_list[:20]:
        print("{}".format(x))
    return file_name_list


def clean_file_name(text):
    file_name_no_ext = os.path.splitext(text)[0]
    file_name_clean = re.sub('[^a-zA-Z]', ' ', file_name_no_ext) ## [^ a-zA-Z0-9]
    return file_name_clean


def data_process(file_name_cleaned):
    #create a Gensim dictionary and corpus from the texts
    texts = [file.split(" ") for file in file_name_cleaned]
    texts = [[tok for tok in file if not tok ==''] for file in texts]
    stopwords = nltk.corpus.stopwords.words('english')
    texts = [[tok for tok in file if tok not in stopwords] for file in texts]
    texts = [[tok for tok in file if len(tok) > 1] for file in texts]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts] 
    return texts, corpus, dictionary


# Get sentence embedding using average
def sent_vectorizer(sent, model):
    sent_vec =[]
    numw = 0
    for w in sent:
        try:
            if numw == 0:
                sent_vec = model[w]
            else:
                sent_vec = np.add(sent_vec, model[w])
            numw+=1
        except:
            pass
    return np.asarray(sent_vec) / numw


def run_kmeans_embedding(texts, w2v_model, num_cluster=10):    
    # Create embeddings for file name
    X=[]
    for sentence in texts:
        X.append(sent_vectorizer(sentence, w2v_model))
    
    # remove empty element
    X_new = []
    for i in X:
        if len(i) == 300:
            X_new.append(i)
        
    # Train the Kmeans model
    model = KMeans(n_clusters=num_cluster, init='k-means++', max_iter=100, n_init=1, random_state=3425)
    model.fit(X_new)
    
    # Output folder for clustering results
    time_stamp = time.strftime("%Y_%m_%d_%H_%M_%S")
    output_folder = "Pretrained_Embedding_cluster_output_" + time_stamp
    output_folder_path = "./data/" + output_folder
    os.mkdir(output_folder_path)
    
    # For each of the cluster, print out the documents inside of this cluster
    for i in range(num_cluster):
        cluster_folder_name = "cluster_" + str(i)
        cluster_folder_path = output_folder_path + "/" + cluster_folder_name + "/"
        os.mkdir(cluster_folder_path)
        
        print("\nDocuments in cluster {}".format(i))
        file_idx_in_cluster = np.where(model.labels_ == i)[0]
        for f in file_idx_in_cluster[:10]:
            print(" {} ".format(files[f]))
        print("\n")
        
        for f in file_idx_in_cluster:
            src = "./data/" + files[f]
            dst = cluster_folder_path + files[f]
            copyfile(src, dst)



mypath = '.\data'
files = retrieve_file_names(mypath)
file_name_cleaned = [clean_file_name(x) for x in files]
texts = data_process(file_name_cleaned)[0]
print("\nPretrained embedding using Google News data is being loaded ... ")
google_model = KeyedVectors.load_word2vec_format('.\data\GoogleNews-vectors-negative300.bin.gz', binary=True)
run_kmeans_embedding(texts, google_model)