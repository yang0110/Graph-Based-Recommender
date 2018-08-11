import numpy as np 
import pandas as pd 
from sklearn.preprocessing import normalize
import os 
import matplotlib.pyplot as plt 
os.chdir('C:/Kaige_Research/Graph_based_recommendation_system/Code/')
from artificial_data import * 
from initial_data import * 
from linucb import Linucb 
from cd import Cd 
from club import Club 
import networkx as nx 
import datetime
from scipy.sparse.csgraph import laplacian 
from community import community_louvain
from numpy.linalg import eigvals
from sklearn.preprocessing import MinMaxScaler
timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M_%S') 

store_file='C:/Kaige_Research/Graph_based_recommendation_system/Data/delicious_small_data_set/'
results_folder='C:/Kaige_Research/Graph_based_recommendation_system/Result/delicious/'

newpath = results_folder+'results_'+str(timeRun)+'/'
if not os.path.exists(newpath):
    os.makedirs(newpath)
################### Original data
user_json=np.load(store_file+'delicious_popular_user_json.npy').item()#223
article_features_array=np.load(store_file+'top_1000_users_bookmarks_features.npy')
article_features_array=Normalizer().fit_transform(article_features_array)
popular_article_list=np.load(store_file+'delicious_popular_item_list.npy')
#
print(len(popular_article_list))
real_article=article_features_array
real_user=user_json

#######
real_data=True
binary_reward=True
dimension=25
user_num=len(real_user.keys())
article_num=len(real_article)
guess_cluster_num=5
pool_size=25
reward_noise_scale=0.0
top_n_similarity=700
alpha=0.05
alpha_2=0.2
iterations=100000

all_random_users=generate_all_random_users(iterations,user_json)
all_article_pool=generate_all_article_pool(iterations, all_random_users, real_user, pool_size, article_num, popular_article_list)

################### models 
linucb=Linucb(user_num, article_num, pool_size, dimension, alpha, real_user, real_article,rating=None, real_data=real_data, binary_reward=binary_reward,real_reward=False, real_user_features=None, random=False)

club=Club(user_num, article_num, pool_size, dimension,  real_user, real_article, alpha, alpha_2,rating=None, real_data=real_data, binary_reward=binary_reward,real_reward=False, real_user_features=None,  random=False)

cd=Cd(user_num, article_num, pool_size, dimension,  real_user, real_article, alpha,rating=None,top_n_similarity=top_n_similarity, affinity_matrix=None,real_data=real_data, binary_reward=binary_reward, real_reward=False, real_user_features=None, random=False)

############################################################################


linucb_cum_regret,  linucb_cum_reward, linucb_affinity_matrix, linucb_diff_user_features, linucb_users_served_items, lincub_served_users=linucb.run(iterations, time, reward_noise_scale, all_random_users,all_article_pool)


club_cum_regret,club_n_cluster, club_clusters, club_affinity_matrix, club_cum_reward, club_diff_user_features, club_diff_user_cluster_features, club_clustering_score, club_users_served_items, club_served_users=club.run(iterations, time, reward_noise_scale, all_random_users, all_article_pool, real_clusters=None)

cd_cum_regret, cd_n_cluster, cd_clusters, cd_affinity_matrix, cd_cum_reward, cd_diff_user_features, cd_diff_user_cluster_features, cd_clustering_score,cd_users_served_items, cd_served_users=cd.run(iterations, time, reward_noise_scale, all_random_users, all_article_pool, real_clusters=None)

####
np.save(newpath+'linucb_cum_regret', linucb_cum_regret)
np.save(newpath+'linucb_cum_reward', linucb_cum_reward)
np.save(newpath+'LinUCB_similarity_matrix', linucb_affinity_matrix)
np.save(newpath+'linucb_diff_user_features', linucb_diff_user_features)

np.save(newpath+'club_cum_regret', club_cum_regret)
np.save(newpath+'club_cum_reward', club_cum_reward)
np.save(newpath+'club_n_cluster', club_n_cluster)
np.save(newpath+'club_clusters', club_clusters)
np.save(newpath+'CLUB_similarity_matrix', club_affinity_matrix)
np.save(newpath+'club_diff_user_features', club_diff_user_features)


np.save(newpath+'cd_cum_regret',cd_cum_regret)
np.save(newpath+'cd_cum_reward', cd_cum_reward)
np.save(newpath+'cd_n_cluster',cd_n_cluster)
np.save(newpath+'cd_clusters',cd_clusters)
np.save(newpath+'CD_similarity_matrix', cd_affinity_matrix)
np.save(newpath+'cd_diff_user_features', cd_diff_user_features)



##########################################################################
plt.figure(figsize=(5,5))
plt.plot(linucb_cum_regret, label='LinUCB',color='b', marker='o', linewidth=1, markevery=0.1, markersize=8)
plt.plot(club_cum_regret, label='CLUB',color='orange', marker='s', linewidth=1, markevery=0.1, markersize=8)
plt.plot(cd_cum_regret, label='SCLUB-CD',color='r', marker='X', linewidth=1, markevery=0.1, markersize=8)
plt.legend(loc='best', fontsize=12)
plt.yticks(rotation=90)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Cumulative Regret', fontsize=12)
plt.savefig(newpath+'data_performance'+'.eps', dpi=600)
plt.show()

plt.figure(figsize=(5,5))
plt.plot(linucb_cum_reward, label='LinUCB',color='b', marker='o', linewidth=1, markevery=0.1, markersize=8)
plt.plot(club_cum_reward, label='CLUB',color='orange', marker='s', linewidth=1, markevery=0.1, markersize=8)
plt.plot(cd1_cum_reward, label='SCLUB-CD',color='r', marker='X', linewidth=1, markevery=0.1, markersize=8)
plt.legend(loc='best', fontsize=12)
plt.yticks(rotation=90)
plt.savefig(newpath+'data_performance_reward'+'.eps', dpi=600)
plt.show()

plt.figure(figsize=(5,5))
plt.plot(linucb_diff_user_features, label='LinUCB',color='b', marker='o', linewidth=1, markevery=0.1, markersize=8)
plt.plot(club_diff_user_features, label='CLUB',color='orange', marker='s', linewidth=1, markevery=0.1, markersize=8)
plt.plot(cd_diff_user_features, label='SCLUB-CD',color='r', marker='X', linewidth=1, markevery=0.1, markersize=8)
plt.legend(loc='best', fontsize=12)
plt.yticks(rotation=90)
plt.savefig(newpath+'diff_user_featrues'+'.eps', dpi=600)
plt.show()


plt.figure(figsize=(5,5))
plt.plot(club_n_cluster, label='CLUB',color='orange', marker='s', linewidth=1, markevery=0.1, markersize=8)
plt.plot(cd_n_cluster, label='SCLUB-CD',color='r', marker='X', linewidth=1, markevery=0.1, markersize=8)
plt.legend(loc='best',fontsize=12)
plt.ylim([0,200])
plt.yticks(rotation=90)
plt.ylabel('Cluster Number', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.savefig(newpath+'evolution_of_cluster_number'+'.eps', dpi=600)
plt.show()


