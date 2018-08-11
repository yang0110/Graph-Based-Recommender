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
import matplotlib
from scipy.sparse import csgraph 
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import networkx as nx 
from community import community_louvain
from sklearn.cluster import SpectralClustering,KMeans, DBSCAN
import pandas as pd 
import csv
from networkx.drawing.nx_pydot import write_dot
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets.samples_generator import make_blobs
from networkx.algorithms.community.centrality import girvan_newman
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel

def feature_uniform(dimension):
	vector = np.array([np.random.uniform( ) for _ in range(dimension)])
	l2_norm = np.linalg.norm(vector, ord =2)
	vector = vector/l2_norm
	return vector

def init_user_features(user_num, user_dimension, random=None):
	if random==True:
		user_features=np.zeros((user_num, user_dimension))
		for i in range(user_num):
			user_features[i]=feature_uniform(user_dimension)
	else:
		user_features=np.zeros((user_num, user_dimension))

	return user_features

def init_CBPrime(user_num):
	CBPrime=np.zeros(user_num)
	return CBPrime 

def init_cor_matrix(user_num, dimension):
	all_cor_matrix=np.array([np.identity(dimension) for i in range(user_num)])
	return all_cor_matrix

def init_bia(user_num, dimension):
	all_bias=np.zeros([user_num, dimension])
	return all_bias 

def init_cluster_cor_matrix(user_num, dimension):
	all_cluster_cor_matrix=np.array([np.identity(dimension) for i in range(user_num)])
	return all_cluster_cor_matrix

def init_cluster_bias(user_num, dimension):
	all_cluster_bias=np.zeros([user_num, dimension])
	return all_cluster_bias

def init_user_cluster_features(user_num, dimension):
	user_cluster_feature=np.zeros([user_num, dimension])
	return user_cluster_feature

def init_user_counters(user_num):
	user_counters=np.zeros(user_num)
	return user_counters 


def init_graph(user_num, cluster_init='Erdos-Renyi'):
	if cluster_init=='Erdos-Renyi':
		p=3*float(np.log(user_num))/float(user_num)
		graph=np.random.choice([0,1],size=(user_num, user_num), p=[1-p, p])
	else:
		graph=np.ones((user_num, user_num))
	return graph 

def set_user_color(clusters):
	color=list(clusters/float(np.max(list(clusters))+1))
	return color 


def init_user_json_array_and_article_time(user_json, user_num, article_num):
	user_json_array=np.zeros((user_num, article_num))
	article_time=np.zeros(article_num)
	for i in range(len(user_json.keys())):
		print('i/user_num', i, len(user_json.keys()))
		user_json_array[i, user_json[i]]=1
		article_time=np.sum(user_json_array,axis=1)+1
	return user_json_array, article_time

def mms_transform(simi):
	np.fill_diagonal(simi, np.min(simi))
	mms=MinMaxScaler()
	simi=mms.fit_transform(simi)
	np.fill_diagonal(simi, 0.0)
	return simi 

def generate_graph_from_rbf(adj_matrix):
	adj_matrix=np.matrix(adj_matrix)
	G=nx.from_numpy_matrix(adj_matrix)
	print('Graph info:', nx.info(G))
	return G

def generate_graph(simi):
	G=nx.Graph()
	nodes=range(simi.shape[0])
	edges=[]
	mean=np.mean(simi)
	maxx=np.max(simi)
	for i in nodes:
		for j in nodes:
			if simi[i,j]>0.0:
				edges.extend([(i,j,simi[i,j])])
	G.add_nodes_from(list(nodes))
	G.add_weighted_edges_from(edges)
	print('Graph info:', nx.info(G))
	return G

def generate_graph_from_cos(simi, thres):
	G=nx.Graph()
	nodes=range(simi.shape[0])
	edges=[]
	mean=np.mean(simi)
	maxx=np.max(simi)
	for i in nodes:
		for j in nodes:
			if simi[i,j]>=thres:
				edges.extend([(i,j,simi[i,j])])
	G.add_nodes_from(list(nodes))
	G.add_weighted_edges_from(edges)
	print('Graph info:', nx.info(G))
	return G

# def generate_graph_from_rbf(simi):
# 	G=nx.Graph()
# 	nodes=range(simi.shape[0])
# 	edges=[]
# 	reshape_simi=simi.ravel()
# 	index_2=list(range(len(simi)))*len(simi)
# 	index_1=[]
# 	for i in range(len(simi)):
# 		index_1+=[i]*len(simi)
# 	edges=[(index_1[i], index_2[i], reshape_simi[i]) for i in range(len(reshape_simi)) if reshape_simi[i]>0.0]
# 	G.add_nodes_from(list(nodes))
# 	G.add_weighted_edges_from(edges)
# 	del edges
# 	del nodes
# 	print('Graph info:', nx.info(G))
# 	return G

def plot_graph(graph):
	spring_pos=nx.spring_layout(graph)
	nx.draw_networkx(graph, pos=spring_pos, with_labels=False, node_size=10)
	plt.show()


def find_graph_community(graph):
	parts=community_louvain.best_partition(graph)
	values=[parts.get(node) for node in graph.nodes()]
	return values

def find_community_best_partition(graph):
	parts=community_louvain.best_partition(graph)
	values=[parts.get(node) for node in graph.nodes()]
	clusters=values
	n_clusters=len(np.unique(values))
	del parts
	del values
	return clusters, n_clusters



def find_community_generate_dendrogram(graph):
	deno=community_louvain.generate_dendrogram(graph)

	clusters=[]
	for i in community_louvain.partition_at_level(deno, len(deno)-1).keys():
		clusters.extend([community_louvain.partition_at_level(deno,len(deno)-1)[i]])

	cluster_size=[]
	for i in np.unique(clusters):
		cluster_size.extend([clusters.count(i)])
	n_clusters=len(cluster_size)
	return clusters, cluster_size, n_clusters

def find_community_girvan_newman(graph, k):
	comp=girvan_newman(graph)
	for communities in itertools.islice(comp,k):
		print(tuple(sorted(c) for c in communities))
	return comp

def plot_graph_community(graph):
	parts=community_louvain.best_partition(graph)
	values=[parts.get(node) for node in graph.nodes()]
	nx.draw_spring(graph,cmap=plt.get_cmap('jet'), node_color=values, node_size=35, with_labels=False)
	plt.axis('off')
	plt.show()

def generate_all_random_users(iterations, user_json):
	all_random_users=[]
	for i in range(iterations):
		all_random_users.extend(np.random.choice(list(user_json.keys()),1, replace=True).tolist())
	return all_random_users


def generate_all_article_pool(iterations, all_random_users, user_json, pool_size, article_num, pool):
	all_article_pool=[]
	for i in range(iterations):
		selected_user=all_random_users[i]

		article_pool=np.random.choice(pool, pool_size-1, replace=True).tolist()

		if user_json[selected_user]!=[]:
			if len(user_json[selected_user])<=1:
				article_pool.extend(list(user_json[selected_user]))
			else:
				another_article=list(choice(user_json[selected_user],1))
				article_pool.extend(another_article)
		else:
			pass
			
		all_article_pool.append(article_pool)
	return all_article_pool

def learned_similarity(clusters, simi):
	new_order=[]
	for i in np.unique(clusters):
		new_order.extend(np.where(np.array(clusters)==i)[0].tolist())

	new_rbf=np.zeros((len(clusters), len(clusters)))
	for i in range(len(clusters)):
		new_rbf[i]=simi[new_order[i], new_order]
	return new_rbf 



def find_article_graph(articles_feature_array):

	article_simi=rbf_kernel(articles_feature_array)
	article_graph=generate_graph_from_rbf(article_simi,0.0)
	return article_graph

def find_article_community(article_graph):
	article_clusters, article_cluster_size, article_n_clusters=find_community_generate_dendrogram(article_graph)
	return article_clusters, article_cluster_size, article_n_clusters 



