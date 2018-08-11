import numpy as np
import random
from random import choice, sample
from scipy.linalg import sqrtm
import time
from sklearn import cluster
from scipy.sparse import csgraph 
import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.decomposition import PCA 
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
from sklearn.preprocessing import normalize
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler

class Artificial_user():
	def __init__(self, user_num, cluster_num, article_num, dimension, intra_cluster_noise):
		self.dimension=dimension
		self.user_num=user_num 
		self.article_num=article_num
		self.cluster_num=cluster_num
		self.intra_cluster_noise=intra_cluster_noise
		self.clusters=np.zeros(self.user_num)
		self.user_json={}
		self.article_time=np.zeros(article_num)
		self.simi=np.zeros((user_num, user_num))
		self.user_json_array=np.zeros((user_num, article_num))


	def user_features(self):

		print('User Features Generation ~~~~~~~ Method 3: Blob')
		X, y = make_blobs(n_samples=self.user_num, centers=self.cluster_num, center_box=(0.0,1.0), cluster_std=self.intra_cluster_noise, n_features=self.dimension, shuffle=False)
		X=Normalizer().fit_transform(X)
		artifical_user_features=X
		self.clusters=y
		return artifical_user_features, self.clusters 

	def user_pos(self):
		node_pos, y = make_blobs(n_samples=self.user_num, centers=self.cluster_num, cluster_std=self.intra_cluster_noise, n_features=2)

		return node_pos 	


class Artificial_article():
	def __init__(self, article_num, cluster_num, dimension, intra_cluster_noise):
		self.article_num=article_num
		self.dimension=dimension
		self.cluster_num=cluster_num
		self.intra_cluster_noise=intra_cluster_noise

	def article_features(self):

		artificial_article_features=np.empty([self.article_num, self.dimension])
		for i in range(self.dimension):
			artificial_article_features[:,i]=np.random.normal(0, np.sqrt(1.0*(self.dimension-1)/self.dimension), self.article_num)
		artificial_article_features=Normalizer().fit_transform(artificial_article_features)
		return artificial_article_features

def artificial_user_json(user_features, article_features, top_n_articles, balance_level):
	user_json={}
	user_json_array=np.zeros((user_features.shape[0], article_features.shape[0]))
	print('generating user_json')

	for i in range(user_features.shape[0]):
		rewards=np.dot(np.array(user_features[i]).reshape(1,25),np.transpose(np.array(article_features))).tolist()[0]
		temp={}
		temp['rewards']=rewards
		temp_df=pd.DataFrame(temp)
		top_articles=temp_df.sort_values(by='rewards').index[-top_n_articles:].values
		articles=np.random.choice(top_articles, int(top_n_articles/balance_level), replace=False).tolist()
		user_json[i]=articles
		user_json_array[i, articles]=1
		article_time=np.sum(user_json_array, axis=0)+1
	print('Done Generation')
	return user_json, user_json_array, article_time

def artificial_simi(user_json_array):
	simi=np.zeros((user_json_array.shape[0], user_json_array.shape[0]))
	article_time=np.sum(user_json_array, axis=0)+1
	for i in range(user_json_array.shape[0]):
		print('generate simi, i/user_num', i, len(user_json_array))
		a1=np.sum(user_json_array[i])
		a2=np.sum(user_json_array, axis=1)
		a=1.0/(np.sqrt(a1*a2))
		b1=user_json_array*user_json_array[i]
		b2=(1+np.abs(user_json_array-user_json_array[i]))*article_time
		b=np.sum(b1/b2, axis=1)
		c=list(a*b)
		v_pos=tuple(np.repeat(i, len(user_json_array)))
		h_pos=tuple(range(len(user_json_array)))
		simi[v_pos, h_pos]=c 
		simi[h_pos, v_pos]=c 
	return simi


def find_simi_from_user_json(user_json, article_num):
	user_json_array=np.zeros((len(user_json.keys()), article_num))
	for i in range(len(user_json.keys())):
		articles=user_json[i]
		user_json_array[i, articles]=1
	article_time=np.sum(user_json_array, axis=0)+1	
	simi=np.zeros((user_json_array.shape[0], user_json_array.shape[0]))

	for i in range(user_json_array.shape[0]):
		print('generate simi, i/user_num', i, len(user_json_array))
		a1=np.sum(user_json_array[i])
		a2=np.sum(user_json_array, axis=1)
		a=1.0/(np.sqrt(a1*a2))
		b1=user_json_array*user_json_array[i]
		b2=(1+np.abs(user_json_array-user_json_array[i]))*article_time
		b=np.sum(b1/b2, axis=1)
		c=list(a*b)
		v_pos=tuple(np.repeat(i, len(user_json_array)))
		h_pos=tuple(range(len(user_json_array)))
		simi[v_pos, h_pos]=c 
		simi[h_pos, v_pos]=c 
	return simi, user_json_array, article_time


def find_cos_simi_from_user_json(user_json, article_num):
	user_json_array=np.zeros((len(user_json.keys()), article_num))
	for i in user_json.keys():
		articles=user_json[i]
		user_json_array[i, articles]=1
	article_time=np.sum(user_json_array, axis=0)+1	
	cos_simi=cosine_similarity(user_json_array)
	return cos_simi, user_json_array, article_time


def find_rbf_from_user_json(user_json, article_num):
	user_json_array=np.zeros((len(user_json.keys()), article_num))
	for i in user_json.keys():
		articles=user_json[i]
		user_json_array[i, articles]=1
	article_time=np.sum(user_json_array, axis=0)+1	
	rbf_simi=rbf_kernel(user_json_array)
	return rbf_simi, user_json_array, article_time

def find_rbf_from_features(features):
	rbf_simi=rbf_kernel(features)
	return rbf_simi



