# imports
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import h5py
import os

plt.rcParams['lines.markersize'] = 2

def cluster_silhouette_score(h5py_file, epoch, clusters =2, save_path = None):
  with h5py.File(h5py_file) as file:
    cluster = file[f"cluster_epoch_{epoch}"][:]
  pca = PCA(n_components = 2)
  cluster_pca = pca.fit_transform(cluster)
  kmeans = KMeans(n_clusters = clusters, random_state =0, n_init = "auto")
  score = silhouette_score(cluster_pca, kmeans.fit_predict(cluster_pca))
  if save_path is not None:
    labels = kmeans.fit(cluster_pca).labels_
    sns.scatterplot(x =cluster_pca[:,0], y =cluster_pca[:, 1], hue = kmeans.labels_)
    plt.title(f'Silhouette Score: {score:.4f}')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.savefig(save_path)
  return score
  
  
  
  
