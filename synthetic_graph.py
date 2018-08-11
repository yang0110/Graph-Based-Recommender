import numpy as np
import json
import random
from random import choice
from scipy.linalg import sqrtm
import math
import time
import datetime
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from sklearn import cluster
from operator import itemgetter      #for easiness in sorting and finding max and stuff
from matplotlib.pylab import *
from scipy.sparse import csgraph 
import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import networkx as nx 
import os 
os.chdir('C:/Kaige_Research/Graph Learning/graph_sparsity_and_community_detection/code/')
from community import community_louvain
from sklearn.cluster import SpectralClustering, KMeans, DBSCAN
import pandas as pd 
import csv
from networkx.drawing.nx_pydot import write_dot
from sklearn.datasets.samples_generator import make_blobs
from networkx.algorithms.community.centrality import girvan_newman
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from collections import Counter
from sklearn.metrics.cluster import adjusted_rand_score


class Node_generator():
	def __init__(self, node_num, cluster_num, dimension, intra_cluster_noise, size_balance_parameter):
		self.dimension=dimension
		self.node_num=node_num 
		self.cluster_num=cluster_num
		self.intra_cluster_noise=intra_cluster_noise
		self.clusters=[]
		self.z=size_balance_parameter
		self.size=np.zeros(cluster_num)
		self.node_pos=[]
		self.node_f=[]

	def cluster_size(self):
		self.size+=1
		for i in range(self.cluster_num):
			a=(i+1)**(-self.z)
			b=0
			for j in range(self.cluster_num):
				b+=(j+1)**(-self.z)
			self.size[i]=(self.node_num-self.cluster_num)*(a/b)
		self.size=self.size.astype(int)
		remainder=self.node_num-np.sum(self.size)
		self.size[-1]=self.size[-1]+remainder
		return self.size



	def node_features(self):
		self.size=self.cluster_size()
		X, y = make_blobs(n_samples=self.node_num*self.cluster_num, centers=self.cluster_num, center_box=(0.0,1.0), cluster_std=self.intra_cluster_noise, n_features=self.dimension, shuffle=False)
		X=Normalizer().fit_transform(X)

		for i in range(self.cluster_num):
			features=X[i*(self.node_num):(i+1)*self.node_num][:self.size[i]]
			self.node_f.extend(features)
			self.clusters.extend(y[i*(self.node_num):(i+1)*self.node_num][:self.size[i]])

		self.node_f=np.array(self.node_f)
		return self.node_f, self.clusters,self.size


class Node_generator_diff_intra_cluster_noise():
	def __init__(self, node_num, cluster_num, dimension, intra_cluster_noise_array, size_balance_parameter):
		self.dimension=dimension
		self.node_num=node_num 
		self.cluster_num=cluster_num
		self.intra_cluster_noise=intra_cluster_noise_array
		self.clusters=[]
		self.z=size_balance_parameter
		self.size=np.zeros(cluster_num)
		self.node_pos=[]
		self.node_f=[]

	def cluster_size(self):
		for i in range(self.cluster_num):
			a=(i+1)**(-self.z)
			b=0
			for j in range(self.cluster_num):
				b+=(j+1)**(-self.z)
			self.size[i]=(self.node_num)*(a/b)
		self.size=self.size.astype(int)
		remainder=self.node_num-np.sum(self.size)
		self.size[-1]=self.size[-1]+remainder
		# self.size+=1
		# summ=np.sum(self.size)
		# rem=summ-self.node_num
		# if rem>0:
		# 	self.size[0]=self.size[0]-rem
		# else:
		# 	pass
		#print('sum',np.sum(self.size))
		return self.size


	def node_features(self):
		self.size=self.cluster_size()
		for i in range(len(self.size)):
			#print('self.size[i]', self.size[i])
			X, y = make_blobs(n_samples=self.size[i], centers=1, center_box=(0.0,1.0), cluster_std=self.intra_cluster_noise[i], n_features=self.dimension, shuffle=False)
			X=Normalizer().fit_transform(X)
			self.node_f.extend(X)
			self.clusters.extend(list(i*np.ones(self.size[i])))

		self.node_f=np.array(self.node_f)
		return self.node_f, self.clusters,self.size



def generate_position(cluster_num, cluster_size, intra_cluster_noise):

	centers=[]
	for i in range(cluster_num):
		theta=(i+1)*2*np.pi/cluster_num
		centers.append(np.array([0.0+1*np.sin(theta), 0.0+1*np.cos(theta)]))
	centers=np.array(centers)
	pos=[]
	for i in range(cluster_num):
		random=np.random.normal(loc=0.0, scale=intra_cluster_noise, size=(cluster_size[i],2))
		pos.extend(centers[i]+random)
	pos=np.array(pos)
	return pos

def generate_position_diff_intra_cluster_noise(cluster_num, cluster_size, intra_cluster_noise_array):

	centers=[]
	for i in range(cluster_num):
		theta=(i+1)*2*np.pi/cluster_num
		centers.append(np.array([0.0+1*np.sin(theta), 0.0+1*np.cos(theta)]))
	centers=np.array(centers)
	pos=[]
	for j in range(cluster_num):
		random=np.random.normal(loc=0.0, scale=intra_cluster_noise_array[j], size=(cluster_size[j],2))
		pos.extend(centers[j]+random)
	pos=np.array(pos)
	return pos


def generate_graph(adj_matrix):
	G=nx.Graph()
	G.add_nodes_from(list(range(adj_matrix.shape[0])))
	for i in range(adj_matrix.shape[0]):
		for j in range(adj_matrix.shape[1]):
			if adj_matrix[i,j]==0.0:
				pass 
			else:
				G.add_edge(i,j, weight=adj_matrix[i,j])
	#print('Graph info:', nx.info(G))
	return G, nx.info(G)


def adj_binary_sparse_thres_n(adj_matrix, thres_n):
	a=adj_matrix.shape[0]
	b=adj_matrix.shape[1]
	adj_matrix_copy=adj_matrix.copy()
	for i in range(a):
		rbf_row=adj_matrix_copy[i,:]
		big_index=np.argsort(rbf_row)[a-thres_n:]
		small_index=np.argsort(rbf_row)[:a-thres_n]
		rbf_row[small_index]=0.0
		rbf_row[big_index]=1.0
		adj_matrix_copy[i,:]=rbf_row
		adj_matrix_copy[:,i]=rbf_row
	return adj_matrix_copy

def adj_weight_sparse_thres_n(adj_matrix, thres_n):
	a=adj_matrix.shape[0]
	b=adj_matrix.shape[1]
	adj_matrix_copy=adj_matrix.copy()

	for i in range(a):
		rbf_row=adj_matrix_copy[i,:]
		big_index=np.argsort(rbf_row)[a-thres_n:]
		small_index=np.argsort(rbf_row)[:a-thres_n]
		rbf_row[small_index]=0.0
		#rbf_row[big_index]=1.0
		adj_matrix_copy[i,:]=rbf_row
		adj_matrix_copy[:,i]=rbf_row
	return adj_matrix_copy

def adj_binary_sparse_thres_w(adj_matrix, thres_w):
	a=adj_matrix.shape[0]
	b=adj_matrix.shape[1]
	adj_matrix_copy=adj_matrix.copy()

	for i in range(a):
		rbf_row=adj_matrix_copy[i,:]
		big_index=np.where(rbf_row>=thres_w)[0].tolist()
		small_index=np.where(rbf_row<thres_w)[0].tolist()
		rbf_row[small_index]=0.0
		rbf_row[big_index]=1.0
		adj_matrix_copy[i,:]=rbf_row
		adj_matrix_copy[:,i]=rbf_row
	return adj_matrix_copy

def adj_weight_sparse_thres_w(adj_matrix, thres_w):
	a=adj_matrix.shape[0]
	b=adj_matrix.shape[1]
	adj_matrix_copy=adj_matrix.copy()

	for i in range(a):
		rbf_row=adj_matrix_copy[i,:]
		big_index=np.where(rbf_row>=thres_w)[0].tolist()
		small_index=np.where(rbf_row<thres_w)[0].tolist()
		rbf_row[small_index]=0.0
		#rbf_row[big_index]=1.0
		adj_matrix_copy[i,:]=rbf_row
		adj_matrix_copy[:,i]=rbf_row
	return adj_matrix_copy

def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


'''
score_list_1=[]
score_list_2=[]
for i in range(10):

	node_generator=Node_generator(100,5, 25, 0.1, 0)
	features, clusters, cluster_sizes=node_generator.node_features()
	adj_matrix=rbf_kernel(features)

	adj1=adj_weight_sparse_thres_w(adj_matrix, 0.1)
	adj2=adj_weight_sparse_thres_n(adj_matrix, 100)
	g1, g1_info=generate_graph(adj1)
	g2, g2_info=generate_graph(adj2)


	def find_community_best_partition(graph):
		parts=community_louvain.best_partition(graph)
		clusters=[parts.get(node) for node in graph.nodes()]
		n_clusters=len(np.unique(clusters))
		cluster_size=list(Counter(clusters).values())
		del parts
		return clusters, n_clusters, cluster_size

	clusters1, n_clusters1, cluster_size1=find_community_best_partition(g1)
	clusters2, n_clusters2, cluster_size2=find_community_best_partition(g2)

	score1=adjusted_rand_score(clusters, clusters1)
	score2=adjusted_rand_score(clusters, clusters2)
	print('score1', score1)
	print('score2', score2)
	#print('clusters1', clusters1)
	#print('clusters2', clusters2)
	score_list_1.extend([score1])
	score_list_2.extend([score2])


plt.plot(score_list_1,  label='w')
plt.plot(score_list_2, label='n')
plt.ylim([0,1.0])
plt.legend()
plt.show()
elarge = [(u, v) for (u, v, d) in g.edges(data=True) if d['weight'] > 0.0]
elarge_weight=np.round([d['weight'] for (u, v, d) in g.edges(data=True) if d['weight'] > 0.0], decimals=1)
esmall = [(u, v) for (u, v, d) in g.edges(data=True) if d['weight'] <= 0.5]

pos=generate_position(10, cluster_sizes, 0.1)
nx.draw_networkx_nodes(g, pos=pos, node_color=np.array(clusters), node_size=100, cmap=plt.cm.Paired, alpha=0.5, node_labels=clusters)
nx.draw_networkx_edges(g, pos=pos, edgelist=elarge, edge_color=elarge_weight, edge_cmap=plt.cm.Blues, vmin=0.0, vmax=2.0,  alpha=0.1)
#nx.draw_networkx_edges(g, pos=pos, edgelist=esmall, edge_color='k', style='dashed', alpha=0.1)
plt.axis('off')
plt.show()
'''

