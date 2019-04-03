# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 18:13:14 2019

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
from sklearn.cluster import KMeans

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

def kmeans_cluster_tfidf(files, file_name_list_cleaned, num_cluster=10):
    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,3))
    X = vectorizer.fit_transform(file_name_list_cleaned)

    # Train the Kmeans model
    model = KMeans(n_clusters=num_cluster, init='k-means++', max_iter=100, n_init=1, random_state=3425)
    print("\nK-means clustering using TD-IDF vector: ")
    model.fit(X)

    # Top words in each of the cluster
    print("\nTop terms per cluster:\n")
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()

    # Output folder for clustering results
    time_stamp = time.strftime("%Y_%m_%d_%H_%M_%S")
    output_folder = "KMeans_cluster_TFIDF_output_" + time_stamp
    output_folder_path = mypath + "/" + output_folder
    os.mkdir(output_folder_path)

    # For each of the cluster, print out the top words and documents inside of this cluster
    for i in range(num_cluster):
        print("Cluster %d top words:" % i)

        cluster_folder_name = "cluster_" + str(i)
        cluster_folder_path = output_folder_path + "/" + cluster_folder_name + "/"
        os.mkdir(cluster_folder_path)

        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind])
        print("\n")

        print("Documents in cluster {}".format(i))
        file_idx_in_cluster = np.where(model.labels_ == i)[0]
        for f in file_idx_in_cluster[:10]:
            print(" {} ".format(files[f]))
        print("\n")

        for f in file_idx_in_cluster:
            src = "./data/" + files[f]
            dst = cluster_folder_path + files[f]
            copyfile(src, dst)

    # When you have new cluster coming in, decide which cluster it is most similar to
    new_file_name = "recommendation model collaboration filtering review.doc"
    print("When we have a new file coming in: {}\n".format(new_file_name))

    # Find the most closest cluster
    Y = vectorizer.transform([new_file_name])
    # print(Y)
    prediction = model.predict(Y)
    print("New file belongs to cluster {}".format(prediction))
    print("Other files in cluster {} are:".format(prediction))

    # Find other documents in this cluster
    file_idx_in_cluster = np.where(model.labels_ == prediction[0])[0]
    for f in file_idx_in_cluster[:10]:
        print(" {} ".format(files[f]))


mypath = '.\data'
files = retrieve_file_names(mypath)
file_name_cleaned = [clean_file_name(x) for x in files]
kmeans_cluster_tfidf(files, file_name_cleaned)