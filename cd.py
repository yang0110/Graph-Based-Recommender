import os 
os.chdir('C:/Kaige_Research/Graph_based_recommendation_system/Code/')
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
from sklearn.metrics.cluster import adjusted_rand_score
import psutil

class Cd():
	def __init__(self, user_num, article_num, pool_size, dimension, real_user, real_article, alpha, rating=None, top_n_similarity=None, affinity_matrix=None, real_data=False, binary_reward=False,real_reward=False, real_user_features=None, random=None):
		self.user_num=user_num
		self.article_num=article_num 
		self.dimension=dimension 
		self.pool_size=pool_size
		self.alpha=1+np.sqrt(np.log(2.0/alpha)/2.0)
		self.artificial_article_features=real_article
		self.user_json=real_user #user json
		self.real_user_features=real_user_features
		self.random=random
		self.user_features=init_user_features(self.user_num, self.dimension, random=self.random)
		self.cor_matrix=init_cor_matrix(self.user_num, self.dimension)
		self.bias=init_bia(self.user_num, self.dimension)
		self.cluster_cor_matrix=init_cluster_cor_matrix(self.user_num, self.dimension)
		self.cluster_bias=init_cluster_bias(self.user_num, self.dimension)
		self.user_cluster_features=init_user_cluster_features(self.user_num, self.dimension)
		self.n_cluster=None
		self.clusters=list(range(self.user_num))
		self.affinity_matrix=affinity_matrix
		self.top_n_similarity=top_n_similarity
		self.cluster_size=None
		self.binary_reward=binary_reward
		self.real_data=real_data
		self.rating=rating
		self.real_reward=real_reward
		self.users_served_items={}
		self.served_users=[]
		self.not_served_users=list(range(self.user_num))



	def get_optimal_reward(self, selected_user, article_pool):
		if self.real_data==True:
			if self.real_reward==False:
				liked_articles=self.user_json[selected_user]
				max_reward=0.0	
				common_article=set(article_pool)&set(liked_articles)
				if len(common_article)!=0:
					max_reward=1.0
				else:
					max_reward=0.0
			else:
				rates=self.rating[selected_user][article_pool]
				max_reward=np.max(rates)				
		else:
			if self.binary_reward==True:
				rewards=np.dot(self.artificial_article_features[article_pool], self.real_user_features[selected_user])
				big_index=np.where(rewards>=0.0)[0].tolist()
				small_index=np.where(rewards<0.0)[0].tolist()
				rewards[big_index]=1.0
				rewards[small_index]=0.0
				max_reward=np.max(rewards)
			else: 
				rewards=np.dot(self.artificial_article_features[article_pool], self.real_user_features[selected_user])
				max_reward=np.max(rewards)

		return max_reward

	def choose_article(self, selected_user, article_pool, time):
		mean=np.dot(self.artificial_article_features[article_pool], self.user_cluster_features[selected_user])
		temp1=np.dot(self.artificial_article_features[article_pool], np.linalg.inv(self.cluster_cor_matrix[selected_user]))
		temp2=np.sum(temp1*self.artificial_article_features[article_pool], axis=1)*np.log(time+1)
		var=np.sqrt(temp2)
		pta=mean+self.alpha*var
		article_picked=np.argmax(pta)
		article_picked=article_pool[article_picked]
		return article_picked

	def random_choose_article(self, article_pool):
		article_picked=choice(article_pool)
		return article_picked

	def get_reward(self, selected_user, picked_article):
		if self.real_data==True:

			liked_articles=self.user_json[selected_user]
			reward=0.0
			if picked_article in liked_articles:
				reward=1.0
			else:
				reward=0.0



		else:
			if self.binary_reward==True:
				reward=np.dot(self.real_user_features[selected_user], self.artificial_article_features[picked_article])

				if reward>=0.0:
					reward=1.0
				else:
					reward=0.0
			else:

				reward=np.dot(self.real_user_features[selected_user], self.artificial_article_features[picked_article])


		return reward

	def get_regret(self, max_reward, reward):
		regret=max_reward-reward
		return regret

	def find_cluster(self, time,iterations,selected_user):

		if (time%100!=0):
			pass
		else:			
			if self.affinity_matrix is None:
				self.affinity_matrix=np.zeros([self.user_num, self.user_num])
			else:
				rbf_row=rbf_kernel(self.user_features[selected_user].reshape(1,-1), self.user_features)[0]
				if len(self.not_served_users)==0:
					pass 
				else:
					rbf_row[self.not_served_users]=0
				big_index=np.argsort(rbf_row)[self.user_num-self.top_n_similarity:]
				small_index=np.argsort(rbf_row)[:self.user_num-self.top_n_similarity]
				rbf_row[small_index]=0.0
				rbf_row[big_index]=1.0
				self.affinity_matrix[selected_user,:]=rbf_row
				self.affinity_matrix[:,selected_user]=rbf_row


				del rbf_row
				del small_index
				del big_index
			graph=generate_graph_from_rbf(self.affinity_matrix)
			self.clusters, self.n_cluster=find_community_best_partition(graph)
			del graph

		print('cd cluster num ~~~~~~~~~~~~~~~~~~', self.n_cluster)
		return self.n_cluster, self.clusters


	def update_user_feature(self, selected_user, picked_article, reward):
		self.cor_matrix[selected_user]+=np.outer(self.artificial_article_features[picked_article], self.artificial_article_features[picked_article])
		self.bias[selected_user]+=self.artificial_article_features[picked_article]*reward

		inv_cor_matrix=np.linalg.inv(self.cor_matrix[selected_user])
		self.user_features[selected_user]=np.dot(inv_cor_matrix, self.bias[selected_user])
		del inv_cor_matrix


	def update_cluster_parameter(self, selected_user, reward, time):
		if (time%100!=0):
			pass 
		else:
			same_cluster=np.where(np.array(self.clusters)==self.clusters[selected_user])[0].tolist()
			self.cluster_cor_matrix[selected_user]=np.identity(self.dimension)+np.sum(self.cor_matrix[same_cluster]-np.identity(self.dimension), axis=0)
			self.cluster_bias[selected_user]=sum(self.bias[same_cluster], axis=0)
			inv_cluster_cor=np.linalg.inv(self.cluster_cor_matrix[selected_user])
			new_cluster_feature=np.dot(inv_cluster_cor, self.cluster_bias[selected_user])
			for i in same_cluster:
				self.user_cluster_features[i]=new_cluster_feature
			# same_cluster=np.where(np.array(self.clusters)==self.clusters[selected_user])[0].tolist()
			# new_cluster_feature=np.mean(self.user_features[same_cluster], axis=0)
			# for i in same_cluster:
			# 	self.user_cluster_features[i]=new_cluster_feature
			del same_cluster
			del new_cluster_feature


	def run(self, iterations, time, reward_noise_scale,all_random_users, all_artilce_pool, real_clusters):
		cum_regret=[0]
		cum_reward=[0]
		cum_n_cluster=[0]
		user_features_diff=[0]
		user_cluster_features_diff=[0]
		clustering_score=[0]
		for time in range(iterations):
			print('CD time ~~~~~~~~ ', time)
			user=all_random_users[time]
			if user in self.served_users:
				pass 
			else:
				self.served_users.extend([user])
				self.users_served_items[user]=[]
				self.not_served_users.remove(user)
			article_pool=all_artilce_pool[time]
			optimal_reward=self.get_optimal_reward(user, article_pool)
			n_cluster, clusters=self.find_cluster(time, iterations,user)


			if real_clusters is not None:
				score=adjusted_rand_score(real_clusters, clusters)
				clustering_score.extend([score])
			else:
				pass 

			picked_article=self.choose_article(user, article_pool, time)

			reward=self.get_reward(user, picked_article)
			if reward_noise_scale==0.0:
				noise_reward=reward
			else:
				noise_reward=reward+np.random.normal(loc=0.0, scale=reward_noise_scale)			
			regret=self.get_regret(optimal_reward, reward)

			if picked_article in self.users_served_items[user]:
				pass 
			else:
				self.users_served_items[user].extend([picked_article])
				self.update_user_feature(user, picked_article, noise_reward)

			self.update_cluster_parameter(user, noise_reward,time)

			if self.real_user_features is not None:

				diff_real_and_learned_user_features=np.sum(np.linalg.norm(self.user_features-self.real_user_features, axis=1))
				user_features_diff.extend([diff_real_and_learned_user_features])
				diff_real_and_learned_user_cluster_features=np.sum(np.linalg.norm(self.user_cluster_features-self.real_user_features, axis=1))
				user_cluster_features_diff.extend([diff_real_and_learned_user_cluster_features])
				del diff_real_and_learned_user_features
				del diff_real_and_learned_user_cluster_features
			else:
				pass 


			cum_n_cluster.extend([n_cluster])
			cum_regret.extend([cum_regret[-1]+regret])
			cum_reward.extend([cum_reward[-1]+reward])

		return np.array(cum_regret), cum_n_cluster, clusters, self.affinity_matrix, np.array(cum_reward), user_features_diff, user_cluster_features_diff, clustering_score, self.users_served_items, self.served_users
