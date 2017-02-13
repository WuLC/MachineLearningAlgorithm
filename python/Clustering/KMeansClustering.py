# -*- coding: utf-8 -*-
# @Author: WuLC
# @Date:   2017-02-13 09:03:42
# @Last Modified by:   WuLC
# @Last Modified time: 2017-02-13 10:17:27


# Clustering with KMeans algorithm

import random
from math import sqrt
from GetData import read_data

def pearson(v1,v2):
	"""use pearson coeffcient to caculate the distance between two vectors
	
	Args:
	    v1 (list): values of vector1
	    v2 (list): values of vector2
	
	Returns:
	   (flaot):1 - pearson coeffcient, the smaller, the more similar
	"""
	# Simple sums
	sum1=sum(v1)
	sum2=sum(v2)
	# Sums of the squares
	sum1Sq=sum([pow(v,2) for v in v1])
	sum2Sq=sum([pow(v,2) for v in v2])
	# Sum of the products
	pSum=sum([v1[i]*v2[i] for i in xrange(len(v1))])
	# Calculate r (Pearson score)
	num=pSum-(sum1*sum2/len(v1))
	den=sqrt((sum1Sq-pow(sum1,2)/len(v1))*(sum2Sq-pow(sum2,2)/len(v1)))
	if den==0: return 0
	return 1.0-num/den

def kMeans(blog_data, distance = pearson, k = 5):
	m, n = len(blog_data), len(blog_data[0])
	max_value = [0 for i in xrange(n)]
	min_value = [0 for i in xrange(n)]
	for i in xrange(m):
		for j in xrange(n):
			max_value[j] = max(max_value[j], blog_data[i][j])
			min_value[j] = min(min_value[j], blog_data[i][j])

    # initial random clusters
	clusters = []
	for i in xrange(k):
		clusters.append([min_value[j] + random.random()*(max_value[j] - min_value[j]) for j in xrange(n)])

	count = 0
	previous_cluster_nodes = None
	while True:
		count += 1
		print 'iteration count %s'%count
		curr_cluster_nodes = [[] for i in xrange(k)]
		for i in xrange(m):
			closest_distance = distance(blog_data[i], clusters[0])
			cluster = 0
			for j in xrange(1, k):
				d = distance(blog_data[i], clusters[j])
				if closest_distance > d:
					closest_distance = d
					cluster = j
			curr_cluster_nodes[cluster].append(i)

		if curr_cluster_nodes == previous_cluster_nodes:
			break

		previous_cluster_nodes = curr_cluster_nodes
		# modify the core of each cluster
		for i in xrange(k):
			tmp = [0 for _ in xrange(n)]
			for node in curr_cluster_nodes[i]:
				for j in xrange(n):
					tmp[j] += blog_data[node][j] 
			clusters[i] = [float(tmp[j])/len(curr_cluster_nodes) for j in xrange(n)]
	return clusters, curr_cluster_nodes

if __name__ == '__main__':
	col_names, blog_names, blog_data = read_data('Clustering_data/data')
	clusters, cluster_nodes = kMeans(blog_data)
	for i in xrange(len(cluster_nodes)):
		print '=============cluster %s==========='%i
		for node in cluster_nodes[i]:
			print blog_names[node]
		






