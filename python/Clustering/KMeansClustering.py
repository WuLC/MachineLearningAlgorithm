# -*- coding: utf-8 -*-
# @Author: WuLC
# @Date:   2017-02-13 09:03:42
# @Last Modified by:   WuLC
# @Last Modified time: 2017-02-15 20:54:58


# Clustering with KMeans algorithm

import random
from math import sqrt
from PIL import Image,ImageDraw
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


def scale_dowm(blog_data,distance=pearson,rate=0.01):
    """transform data in multiple-dimentional to two-dimentional
    
    Args:
        data (list[list[]]):  blog data in the form of a two-dimentional matrix  
        distance (TYPE, optional): standark to caculate similarity between two vectors
        rate (float, optional): rate to move the position of the nodes
    
    Returns:
        list[list[]]: position of nodes in a two dimentional coordinate
    """
    n=len(blog_data)

    # The real distances between every pair of items
    real_list=[[distance(blog_data[i],blog_data[j]) for j in xrange(n)] 
             for i in xrange(n)]

    # Randomly initialize the starting points of the locations in 2D
    loc=[[random.random(), random.random()] for i in xrange(n)]
    fake_list=[[0.0 for j in xrange(n)] for i in xrange(n)]

    lasterror=None
    for m in range(0,1000):
        # Find projected distances
        for i in range(n):
          for j in range(n):
            fake_list[i][j]=sqrt(sum([pow(loc[i][x]-loc[j][x],2) 
                                     for x in xrange(len(loc[i]))]))

        # Move points
        grad=[[0.0,0.0] for i in range(n)]

        totalerror=0
        for k in range(n):
          for j in range(n):
            if j==k or real_list[j][k] == 0: continue  # acoid the case when real_list[j][k] == 0.0
            # The error is percent difference between the distances
            error_term=(fake_list[j][k]-real_list[j][k])/real_list[j][k]
            
            # Each point needs to be moved away from or towards the other
            # point in proportion to how much error it has
            grad[k][0] += ((loc[k][0]-loc[j][0])/fake_list[j][k])*error_term
            grad[k][1] += ((loc[k][1]-loc[j][1])/fake_list[j][k])*error_term

            # Keep track of the total error
            totalerror+=abs(error_term)
        # print 'curr error {0}'.format(totalerror)

        # If the answer got worse by moving the points, we are done
        if lasterror and lasterror<totalerror: break
        lasterror=totalerror

        # Move each of the points by the learning rate times the gradient
        for k in range(n):
          loc[k][0] -= rate*grad[k][0]
          loc[k][1] -= rate*grad[k][1]

    return loc


def draw_clusters(blog_data, clusters, cluster_nodes, blog_names, jpeg_path = 'Clustering_data/mds2d.jpg'):
    """draw the result of KMeans clustering
    
    Args:
        blog_data (list[list]): blog data that had been transfromed into two-dimentional form
        clusters (list[list]): center of clusters that had been transfromed into two-dimentional form
        cluster_nodes (list[list]): nodes of each cluster
        blog_names (list[str]): blog name corresponding to each node
        jpeg_path (str, optional): path of the photo to be stored
    
    Returns:
        None
    """
    img=Image.new('RGB',(2000,2000),(255,255,255))
    draw=ImageDraw.Draw(img)
    for i in xrange(len(clusters)):
        for node in cluster_nodes[i]:
            c_x,c_y = (clusters[i][0] + 0.5)*1000, (clusters[i][1] + 0.5)*1000
            x, y =(blog_data[node][0]+0.5)*1000, (blog_data[node][1]+0.5)*1000
            draw.line((c_x, c_y, x, y),fill=(255,0,0))
            draw.text((x,y),blog_names[node],(0,0,0))   
    img.save(jpeg_path ,'JPEG') 


if __name__ == '__main__':
    cluster_num = 4
    col_names, blog_names, blog_data = read_data('Clustering_data/data')
    clusters, cluster_nodes = kMeans(blog_data, k = cluster_num)
    for i in xrange(len(cluster_nodes)):
        print '=============cluster %s==========='%i
        for node in cluster_nodes[i]:
            print blog_names[node]

    scaled_data = scale_dowm(blog_data + clusters)
    scaled_blog_data = scaled_data[:len(blog_data)]
    scaled_clusters = scaled_data[len(blog_data):]
    draw_clusters(scaled_blog_data, scaled_clusters, cluster_nodes, blog_names)
    






