# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 18:31:13 2019

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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import ward, dendrogram
from sklearn.cluster import AgglomerativeClustering

import matplotlib.pyplot as plt

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

def hierarchical_cluster(files, file_name_list_cleaned, num_cluster=10):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,3))
    X = vectorizer.fit_transform(file_name_list_cleaned)
    
    dist = 1 - cosine_similarity(X)
    linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances
    
    fig, ax = plt.subplots(figsize=(15, 20)) # set size
    ax = dendrogram(linkage_matrix, orientation="right", labels=files);
    
    plt.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    
    plt.tight_layout() #show plot with tight layout
    
    #uncomment below to save figure
    plt.savefig('ward_clusters.png', dpi=500) #save figure as ward_clusters
    
    # for linkage in ('ward', 'average', 'complete', 'single'):
    linkage = 'ward'#, 'average', 'complete', 'single'):
    model = AgglomerativeClustering(linkage=linkage, n_clusters=num_cluster)
    model.fit(X.toarray())
    
    # Output folder for clustering results
    time_stamp = time.strftime("%Y_%m_%d_%H_%M_%S")
    output_folder = "Hierarchical_cluster_TFIDF_output_" + time_stamp
    output_folder_path = "./data/" + output_folder
    os.mkdir(output_folder_path)
    
    # For each of the cluster, print out the top words and documents inside of this cluster
    for i in range(num_cluster):
        
        cluster_folder_name = "cluster_" + str(i)
        cluster_folder_path = output_folder_path + "/" + cluster_folder_name + "/"
        os.mkdir(cluster_folder_path)
        
        print("Documents in cluster {}".format(i))
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
hierarchical_cluster(files, file_name_cleaned)