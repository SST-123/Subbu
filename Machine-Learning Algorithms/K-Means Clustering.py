# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 13:16:22 2020

@author: admin
"""

"""> One of the mall need to know about type of customers and their respective
spending in their mall.
> main variables are annula income of customers and spending score given by
mall by their previous purchase and arrival and consedering some other factors"""


#Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

#Reading the data
dataset = pd.read_csv("Mall_Customers.csv")
dataset.head()

dataset.isnull().sum()

dataset.info()

X = dataset.iloc[:,[3,4]].values

#finding the optimal k value

from sklearn.cluster import KMeans

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('wcss')
plt.show()

kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state=0)
y_means = kmeans.fit_predict(X)

#visualizing the clusters
plt.figure(figsize=(6,4))
plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], s = 100, c='red',label = 'Careful')
plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], s = 100, c='blue',label = 'Standard')
plt.scatter(X[y_means == 2, 0], X[y_means == 2, 1], s = 100, c='green',label = 'Target')    
plt.scatter(X[y_means == 3, 0], X[y_means == 3, 1], s = 100, c='cyan',label = 'Careless')    
plt.scatter(X[y_means == 4, 0], X[y_means == 4, 1], s = 100, c='magenta',label = 'Sensible')    
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c='yellow', label='centroids')
plt.title('Clusters of  clients')
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    