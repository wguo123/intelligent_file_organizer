# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 19:00:04 2019

@author: weig
"""

import numpy as np
import pandas as pd

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
import gensim.corpora as corpora
from gensim import models 

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

    
def lda_cluster(texts, corpus, dictionary, num_cluster=10):
    # Train LDA model
    lda = models.LdaModel(corpus, num_topics=num_cluster, id2word=dictionary, update_every=5, chunksize=100, passes=100)
    
    # Show topics
    topics_matrix = lda.show_topics(formatted=False, num_words=20)
    topics_per_cluster = []
    for i in range(num_cluster):
        print("\nCluster {} has following topics:".format(i))
        words = []
        for j in range(len(topics_matrix[i][1])):
            words.append(topics_matrix[i][1][j][0])
            print(" {}".format(topics_matrix[i][1][j][0]))
        topics_per_cluster.append(words)

    return lda


def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


def run_lda(labels, num_cluster=10):
    time_stamp = time.strftime("%Y_%m_%d_%H_%M_%S")
    output_folder = "LDA_cluster_output_" + time_stamp
    output_folder_path = "./data/" + output_folder
    os.mkdir(output_folder_path)
    
    for i in range(num_cluster):
        cluster_folder_name = "cluster_" + str(i)
        cluster_folder_path = output_folder_path + "/" + cluster_folder_name + "/"
        os.mkdir(cluster_folder_path)
        
        print("\nDocuments in cluster {}".format(i))
        file_idx_in_cluster = np.where(labels == i)[0]
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
texts, corpus, dictionary = data_process(file_name_cleaned)
lda = lda_cluster(texts, corpus, dictionary) 
df_topic_sents_keywords = format_topics_sentences(ldamodel=lda, corpus=corpus, texts=texts)    
labels = np.array(df_topic_sents_keywords['Dominant_Topic'])    
run_lda(labels)