import numpy as np
import os
import matplotlib.pyplot as plt
import networkx as nx 
os.chdir('C:/Kaige_Research/Graph Learning/graph_sparsity_and_community_detection/code/')
from community import community_louvain
import pandas as pd 
from collections import Counter

def find_community_best_partition(graph):
	parts=community_louvain.best_partition(graph)
	clusters=[parts.get(node) for node in graph.nodes()]
	n_clusters=len(np.unique(clusters))
	cluster_size=list(Counter(clusters).values())
	return clusters, n_clusters, cluster_size, parts

def find_parts_from_clusters(clusters):
	parts={}
	for i in range(len(clusters)):
		parts[i]=clusters[i]
	return parts