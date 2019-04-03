# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 18:45:44 2019

@author: weig
"""

#import nltk
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')

from os import listdir
from os.path import isfile, join
import re
import os
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN

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

def dbcan_cluster(file_name_list_cleaned):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,3))
    X = vectorizer.fit_transform(file_name_list_cleaned)
    db = DBSCAN(eps=0.0001, min_samples=5).fit(X)
    print("dbscan results are very noisy(cluster index -1 means noise/unclustered): ")
    print(Counter(db.labels_))
    
mypath = '.\data'
files = retrieve_file_names(mypath)
file_name_cleaned = [clean_file_name(x) for x in files]
dbcan_cluster(file_name_cleaned)