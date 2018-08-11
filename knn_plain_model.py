import os 
os.chdir('C:/Kaige_Research/Graph Learning/graph_sparsity_and_community_detection/code/')
from initial_data import * 
import numpy as np
import json
import random as randomm
from random import choice
from scipy.linalg import sqrtm
import math
import time
import datetime
from scipy.sparse.csgraph import connected_components, laplacian
from scipy.sparse import csr_matrix
from sklearn import cluster
from operator import itemgetter      #for easiness in sorting and finding max and stuff
from matplotlib.pylab import *
from scipy.sparse import csgraph 
import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import networkx as nx 
from sklearn.cluster import SpectralClustering, KMeans
import pandas as pd 
import csv
from networkx.drawing.nx_pydot import write_dot
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.cluster import completeness_score
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel, euclidean_distances
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.linear_model import SGDRegressor
from scipy.linalg import sqrtm
import scipy.optimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score,normalized_mutual_info_score
import psutil
from synthetic_graph import  generate_graph, adj_weight_sparse_thres_n
from community_detection import find_community_best_partition
from community import community_louvain
import community
from numpy.linalg import eig


class Knn_plain():
	def __init__(self, user_num, article_num, pool_size, dimension,reward_noise_scale, alpha, k,artificial_article_features, neighbors=None,  real_user_features=None, random=None):

		self.dimension=dimension
		self.user_num=user_num
		self.article_num=article_num 
		self.pool_size=pool_size 
		self.alpha=1+np.sqrt(np.log(2.0/alpha)/2.0)
		self.artificial_article_features=artificial_article_features
		self.real_user_features=real_user_features
		self.random=random
		self.user_features=init_user_features(self.user_num, self.dimension, random=self.random)
		self.user_cluster_features=init_user_features(self.user_num, self.dimension, random=self.random)
		self.CBPrime=init_CBPrime(self.user_num)
		self.cor_matrix=init_cor_matrix(self.user_num, self.dimension)
		self.cluster_cor_matrix=init_cor_matrix(self.user_num, self.dimension)

		self.bias=init_bia(self.user_num, self.dimension)
		self.user_counters=init_user_counters(self.user_num)
		self.users_served_items={}
		self.served_users=[]
		self.time=0
		self.k=k
		self.neighbors=neighbors
		self.reward_noise_scale=reward_noise_scale
		self.errors={}


	def get_optimal_reward(self,selected_user, article_pool):
		if self.reward_noise_scale==0:
			noise=0
		else:
			noise=np.random.normal(loc=0.0, scale=self.reward_noise_scale)
		rewards=np.dot(self.artificial_article_features[article_pool], self.real_user_features[selected_user])+noise

		max_reward=np.max(rewards)
		return max_reward, noise



	def choose_article(self, selected_user, article_pool, time):
		rbf_row=rbf_kernel(self.user_features[selected_user].reshape(1,-1), self.user_features)
		neighbors=np.argsort(rbf_row)[0][self.user_num-self.k:]
		#neighbors=self.neighbors[selected_user]
		neighbors=list(set(neighbors)&set(self.served_users))
		if (len(neighbors)==0):
			self.user_cluster_features[selected_user]=self.user_features[selected_user]
		else:
			if (len(neighbors)==1):
				weights=[1]
			else:
				weights=rbf_row[0][neighbors]/np.sum(rbf_row[0][neighbors])
			self.user_cluster_features[selected_user]=np.average(self.user_features[neighbors], weights=weights, axis=0)

		mean=np.dot(self.artificial_article_features[article_pool], self.user_cluster_features[selected_user])
		temp1=np.dot(self.artificial_article_features[article_pool], np.linalg.inv(self.cluster_cor_matrix[selected_user]))
		temp2=np.sum(temp1*self.artificial_article_features[article_pool], axis=1)*np.log(time+1)
		var=np.sqrt(temp2)
		pta=mean+self.alpha*var
		article_picked=np.argmax(pta)
		article_picked=article_pool[article_picked]
		return article_picked, neighbors


	def get_reward(self, noise, selected_user, picked_article):

		reward=np.dot(self.real_user_features[selected_user], self.artificial_article_features[picked_article])+noise
		residual_error=reward-np.dot(self.artificial_article_features[picked_article], self.user_features[selected_user])
		return reward, residual_error

	def get_regret(self, max_reward, reward):
		regret=max_reward-reward
		return regret

	def update_user_feature(self, selected_user, picked_article, reward):
		self.cor_matrix[selected_user]+=np.outer(self.artificial_article_features[picked_article], self.artificial_article_features[picked_article])
		self.bias[selected_user]+=self.artificial_article_features[picked_article]*reward
		inv_cor_matrix=np.linalg.inv(self.cor_matrix[selected_user])
		self.user_features[selected_user]=np.dot(inv_cor_matrix, self.bias[selected_user])

	def update_cluster_parameter(self, neighbors, selected_user):

		same_cluster=neighbors
		if len(neighbors)==0:
			self.cluster_cor_matrix[selected_user]=self.cor_matrix[selected_user]
		else:
			self.cluster_cor_matrix[selected_user]=np.identity(self.dimension)+np.sum(self.cor_matrix[same_cluster]-np.identity(self.dimension), axis=0)

	def find_features_variance(self, selected_user): # sigma**2 X'X
		temp1=np.dot(np.transpose(self.artificial_article_features[self.users_served_items[selected_user]]),self.artificial_article_features[self.users_served_items[selected_user]])
		if len(self.errors[selected_user])<=(self.dimension+10):
			sigma_2=0
		else:
			sigma_2=np.sum(self.errors[selected_user])/(len(self.errors[selected_user])-self.dimension)
		covariance_matrix=sigma_2*np.linalg.inv(temp1)
		sum_var=covariance_matrix.diagonal().sum()
#		print('sum_var', sum_var)
		return sum_var


	def run(self, iterations, all_random_users, all_artilce_pool):
		cum_regret=[0]
		cum_reward=[0]
		user_features_diff={}
		user_cluster_diff={}
		regret_list=[]
		reward_list=[]
		feature_variance={}
		all_feature_variance=np.zeros((iterations, self.user_num))
		all_cluster_diff=np.zeros((iterations, self.user_num))
		regret_all_user={}
		for time in range(iterations):
			print('KNN Plain time ~~~~~~~~~~~ ', time)
			self.time=time
			user=all_random_users[time]
			if user in self.served_users:
				pass 
			else:
				self.served_users.extend([user])
				self.users_served_items[user]=[]
				self.errors[user]=[]
				feature_variance[user]=[]
				regret_all_user[user]=[]
				user_features_diff[user]=[]
				user_cluster_diff[user]=[]

			article_pool=all_artilce_pool[time]
			optimal_reward, noise=self.get_optimal_reward(user, article_pool)
			picked_article, neighbors=self.choose_article(user, article_pool, time)
			reward, residual_error=self.get_reward(noise, user, picked_article)
			regret=self.get_regret(optimal_reward, reward)
			regret_all_user[user].extend([regret])
			self.errors[user].extend([residual_error**2])
			self.users_served_items[user].extend([picked_article])
			self.update_user_feature(user, picked_article, reward)
			self.update_cluster_parameter(neighbors, user)
			sum_var=self.find_features_variance(user)
			if sum_var==0:
				pass 
			else:
				feature_variance[user].extend([sum_var])
				all_feature_variance[time, user]=sum_var

			if self.real_user_features is not None:
				diff_real_and_learned_user_features=np.linalg.norm(self.user_features[user]-self.real_user_features[user])
				user_features_diff[user].extend([diff_real_and_learned_user_features])
				diff_cluster=np.linalg.norm(self.user_cluster_features[user]-self.real_user_features[user])
				user_cluster_diff[user].extend([diff_cluster])
				all_cluster_diff[time, user]=diff_cluster

			else:
				pass

			cum_regret.extend([cum_regret[-1]+regret])
			cum_reward.extend([cum_reward[-1]+reward])
			regret_list.extend([regret])
			reward_list.extend([reward])
		return np.array(cum_regret), np.array(cum_reward), user_features_diff,user_cluster_diff, regret_list, reward_list, feature_variance, regret_all_user, all_feature_variance, all_cluster_diff