'''
density based clustering
'''
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
%matplotlib inline

def createdatapoints(centroidlocation , numsamples, clusterDeviation):
    x,y = make_blobs(n_samples = numsamples,centers=centroidlocation,cluster_std=clusterDeviation)
     # Standardize features by removing the mean and scaling to unit variance
    x = StandardScaler().fit_transform(x)
    return x,y

x,y = createdatapoints([[4,3],[2,-1],[-1,4]],1500,0.5)

'''Modeling
DBSCAN stands for Density-Based Spatial Clustering of Applications with Noise. This technique is one of the most common clustering algorithms which works based on density of object. The whole idea is that if a particular point belongs to a cluster, it should be near to lots of other points in that cluster.

It works based on two parameters: Epsilon and Minimum Points
Epsilon determine a specified radius that if includes enough number of points within, we call it dense area
minimumSamples determine the minimum number of data points we want in a neighborhood to define a cluster.'''
epilson = 0.3
minimumsamples = 7 
db = DBSCAN(eps = epilson,min_samples=minimumsamples).fit(x)
labels  = db.labels_
labels

'''Lets Replace all elements with 'True' in core_samples_mask that are in the cluster, 'False' if the points are outliers.'''
core_samples_mask = np.zeros_like(db.labels_,dtype = bool)
core_samples_mask[db.core_sample_indices_] = True
core_samples_mask

# Number of clusters in labels, ignoring noise if present.
n_cluters_ = len(set(labels))-(1 if -1 in labels else 0)
n_cluters_

# Remove repetition in labels by turning it into a set.
unique_labels = set(labels)
unique_labels
# Create colors for the clusters.
import matplotlib.cm as cm
cmap = cm.get_cmap('nipy_spectral')

colors = cmap(np.linspace(0,1,(len(unique_labels))))

for k,col in zip(unique_labels,colors):
    
    if k == -1:
        #black used for noise
        col = 'k'
    class_member_mask=(labels == k)
    
    #plt the datapoints that are in cluster
    xy = x[class_member_mask & core_samples_mask]
    plt.scatter(xy[:,0],xy[:,1],s=50,c=[col],marker = u'o',alpha=0.5)
    
    # plot the outliers
    xy = x[class_member_mask & ~core_samples_mask]
    plt.scatter(xy[:,0],xy[:,1],s=50,c=[col],marker = u'o',alpha=0.5)
        