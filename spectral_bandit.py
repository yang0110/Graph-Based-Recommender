import os 
os.chdir('C:/Kaige_Research/Graph Learning/graph_sparsity_and_community_detection/code/')
from initial_data import * 
import numpy as np
import json
import random as randomm
from random import choice
import time
import datetime
from scipy.sparse.csgraph import connected_components, laplacian
from scipy.sparse import csr_matrix
from sklearn import cluster
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import pandas as pd 
import csv
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import MinMaxScaler, Normalizer
from scipy.linalg import sqrtm
from numpy.linalg import eig

class Spectral_bandit():
	def __init__(self, user_num, item_num, item_features, dimension, lambda_, delta_, T, R, noise, real_alpha=None):
		self.user_num=user_num
		self.item_num=item_num
		self.item_features=item_features
		self.dimension=item_num
		self.lambda_=lambda_
		self.delta_=delta_
		self.T=T
		self.R=R
		self.alpha_=np.zeros((self.user_num, self.dimension))
		self.eigenvalues=None
		self.eigenvectors=None
		self.d=None
		self.C=np.log(self.T)
		self.item_his={}
		self.payoff_his={}
		self.served_user=[]
		self.payoff_noise_scale=noise
		self.cov_matrix={}
		self.real_alpha=real_alpha

	def find_laplacian(self):
		adj_matrix=rbf_kernel(self.item_features)
		diag=np.diag(np.sum(adj_matrix, axis=1))
		laplacian=diag-adj_matrix
		return laplacian

	def find_eigenvector(self, laplacian):
		self.eigenvalues, self.eigenvectors=np.linalg.eig(laplacian)
		return self.eigenvalues, self.eigenvectors

	def find_effective_dimension_d(self):
		d_s=np.arange(self.dimension)-1
		d_times_eigenvalues=d_s*self.eigenvalues.diagonal()
		threshold=self.T/(np.log(1+self.T/self.lambda_))
		max_d=np.where(d_times_eigenvalues<=threshold)[0][-1]
		print('d_s', d_s)
		print('self.eigenvalues.diagonal()', self.eigenvalues.diagonal())
		print('d_times_eigenvalues', d_times_eigenvalues)
		print('threshold', threshold)
		print('where', np.where(d_times_eigenvalues<=threshold)[0])
		print('max_d', max_d)
		return max_d


	def get_payoff(self, selected_item, user_index):
		eigenvector=self.eigenvectors[selected_item]
		payoff=np.dot(eigenvector, self.real_alpha[user_index])+np.random.normal(loc=0.0, scale=self.payoff_noise_scale)
		return payoff

	def get_regret(self, user_index, payoff):
		means=np.dot(self.eigenvectors, self.real_alpha[user_index])
		max_=np.max(means)
		regret=max_-payoff
		return regret

	def select_item(self, user_index,t):
		means=np.dot(self.eigenvectors, self.alpha_[user_index])
		c_t=2*self.R*np.sqrt(self.d*np.log(1+t/self.lambda_)+2*np.log(1/self.delta_))+self.C
		temp=np.sum(np.dot(self.eigenvectors, np.linalg.inv(self.cov_matrix[user_index]))*self.eigenvectors, axis=1)
		var=c_t*temp
		item_picked=np.argmax(means+var)
		return item_picked

	def update_alpha(self, user_index, payoff):
		item_list=self.item_his[user_index]
		eigenvectors=self.eigenvectors[item_list]
		payoff=self.payoff_his[user_index]

		self.cov_matrix[user_index]=np.dot(np.transpose(eigenvectors), eigenvectors)+self.eigenvalues
		self.alpha_[user_index]=np.dot(np.dot(np.linalg.inv(self.cov_matrix[user_index]), np.transpose(eigenvectors)), payoff)

	def run(self, random_user_list):
		laplacian=self.find_laplacian()
		self.eigenvalues, self.eigenvectors=self.find_eigenvector(laplacian)
		self.eigenvalues=self.eigenvalues+self.lambda_*np.identity(self.item_num)
		self.d=self.find_effective_dimension_d()
		cumulative_regret=[0]
		diff_list=[]
		for t in np.arange(self.T):
			print('Spectral_bandit Time ~~~~~~~~~~~~', t)
			print('effective_dimension', self.d)

			user_index=random_user_list[t]
			if user_index in self.served_user:
				pass 
			else:
				self.served_user.extend([user_index])
				self.item_his[user_index]=[]
				self.payoff_his[user_index]=[]
				self.cov_matrix[user_index]=np.random.normal(size=(self.dimension, self.dimension))

			selected_item=self.select_item(user_index, t)
			payoff=self.get_payoff(selected_item, user_index)
			self.item_his[user_index].extend([selected_item])
			self.payoff_his[user_index].extend([payoff])
			self.update_alpha(user_index, payoff)

			regret=self.get_regret(user_index, payoff)
			cumulative_regret.extend([cumulative_regret[-1]+regret])
			if self.real_alpha==None:
				pass
			else:
				diff=np.sum(np.linalg.norm(self.alpha_-self.real_alpha, axis=1))
				diff_list.extend([diff])
		
		return cumulative_regret, diff_list

			







