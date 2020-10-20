import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn import metrics
import pandas as pd
from sklearn.metrics import accuracy_score

# implement the RandIndex algorithm
def RandIndex(assignments, known):
    TruePositive = 0
    FalsePositive = 0
    TrueNegative = 0
    FalseNegative = 0

    for i in range(0,len(assignments)):
        for i in range(i+1,len(assignments)):
            if(assignments[i] == assignments[j]):
                if(known[i] == known[j]):
                    TruePositive = (TruePositive +1)
                else:
                    FalsePositive = (FalsePositive +1)
            else:
                if(known[i] != known[j]):
                    TrueNegative = (TrueNegative +1)
                else:
                    FalseNegative = (FalseNegative +1)

    rand = (TruePositive + TrueNegative)/(TruePositive + TrueNegative + FalsePositive + FalseNegative)
    return rand

# Method to implement the purity score
def purity_score(y_pred, y_true):

    purity = None
    return purity


# Method to implement the kmeans Clustering algorithm 
def kmeansClustering(X, n_clusters):

    y_kmeans = None
    ssd = None
    return y_kmeans, ssd

# Method to implement the Agglomerative Clustering algorithm
def agg_clustering(X,n_clusters ):


    y_agg = None

    return y_agg

# Method to implement the DB_Scan Clustering algorithm

def dbscan_cluster(X, eps, min_samples):


    y_db = None
    return y_db


# Main Method



"""
Read data from csv file.

-> Use the practice data to validate your answer.
-> All the outputs for practice data are given below as comments.
-> Do NOT forget to change the csv file to lab2_data.csv and 
lab2_labels.csv to answer the assignment questions.
-> Do NOT FORGET to use the correct number of clusters when 
you change the dataset.

"""

df_x = pd.read_csv('lab2_practice_data.csv', delimiter=',', header=None)
df_y = pd.read_csv('lab2_practice_labels.csv', delimiter=',', header=None)


X = df_x.values
y_true = df_y.values

# For Kmeans and Agglomerative Clustering you need to input the number of clusters (n_cluster).
# Check the labels.csv to get the correct number of clusters.
y_kmeans, ssd_kmeans = kmeansClustering(X, n_clusters=None)

y_agg = agg_clustering(X,n_clusters=None)

y_db = dbscan_cluster(X, eps=0.3, min_samples=10)

print("\n\n## Rand Index:\n")

print("Kmeans: ", RandIndex(y_kmeans, y_true))
print("Agglomerative: ", RandIndex(y_agg, y_true))
print("DBScan: ", RandIndex(y_db, y_true))

# Practice Output

# Kmeans:  1.0
# Agglomerative:  0.99
# DBScan:  0.554321608040201


print("\n## Purity Score:\n")

print("Kmeans: ", purity_score(y_kmeans, y_true))
print("Agglomerative: ", purity_score(y_agg, y_true))
print("DBScan: ", purity_score(y_db, y_true))

# Practice Output

# Kmeans:  1.0
# Agglomerative:  0.995
# DBScan:  0.705

print("\n## silhouette Score:\n")

print("Kmeans: ", metrics.silhouette_score(X, y_kmeans))
print("Agglomerative: ", metrics.silhouette_score(X, y_agg))
print("DBScan: ", metrics.silhouette_score(X, y_db))

# Practice Output

# Kmeans:  0.6705158469357315
# Agglomerative:  0.6704651744613443
# DBScan:  -0.17778119219321176

print("\n## SSD Score:\n")
print("Kmeans: ", ssd_kmeans)

# Practice Output

# Kmeans:  185.56033350015974
