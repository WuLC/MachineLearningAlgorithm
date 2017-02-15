# -*- coding: utf-8 -*-
# @Author: WuLC
# @Date:   2017-02-12 15:41:09
# @Last Modified by:   WuLC
# @Last Modified time: 2017-02-14 23:05:08

from GetData import read_data
from math import sqrt
from PIL import Image,ImageDraw

class hcluster:
	"""describe a cluster as a node in a tree"""
	def __init__(self, id, vector, distance=0, left = None, right = None):
		"""structure to describe a cluster as a node in a tree
		
		Args:
		    id (int): unique id of the node 
		    vector (list): value of the node
		    distance (int, optional): distance between left tree and right tree of the node if there exists, 0 for leaf nodes
		    left (None, optional): root of the left tree
		    right (None, optional): root of the right tree
		"""
		self.id = id
		self.vector = vector
		self.distance = distance
		self.left = left
		self.right = right


def hierarchicalClustering(blog_data, distance = pearson):
	"""hierachical clustering of data
	
	Args:
	    blog_data (list[list]): data of each blogs, a list of integers represents the data of the blog 
	    distance (TYPE, optional): standark to judge distance between data
	
	Returns:
	    (hcluster): the root of the clustering tree
	"""
	# initi clusters, each node is a cluster
	clusters = [hcluster(id = i, vector = blog_data[i]) for i in xrange(len(blog_data))] 
	# use negativ number to represent cluster with more than one node
	clust_id = -1
	# use distance to store caculated results
	distances = {}

	while len(clusters) > 1:
		similar_pairs = (0,1)
		closest_distance = distance(clusters[0].vector, clusters[1].vector)

		for i in xrange(len(clusters)):
			for j in xrange(i+1, len(clusters)):
				if (clusters[i].id, clusters[j].id) not in distances:
					distances[(clusters[i].id, clusters[j].id)] = distance(clusters[i].vector, clusters[j].vector)
				d = distances[(clusters[i].id, clusters[j].id)]
				if closest_distance > d:
					closest_distance = d
					similar_pairs = (i, j)

		merged_vector = [(clusters[similar_pairs[0]].vector[i] + clusters[similar_pairs[1]].vector[i])/2.0 
						   for i in xrange(len(clusters[similar_pairs[0]].vector))]

		new_cluster = hcluster(id = clust_id, vector = merged_vector, distance = closest_distance, 
								left = clusters[similar_pairs[0]], right = clusters[similar_pairs[1]])

		# must delete elements from higher index to lower index
		del clusters[similar_pairs[1]]
		del clusters[similar_pairs[0]]

		clusters.append(new_cluster)
		clust_id -= 1
	return clusters[0]


def print_cluster(cluster, blog_names, n):
	""" print the cluster in a rough way
	
	Args:
	    cluster (hcluster): root of the clustering tree
	    blog_names (list): name of the blogs, identified by cluster id
	    n (int): indentation of each hierarchy
	
	Returns:
	    None
	"""
	print ' '*n,
	if cluster.id < 0:
		print '-'
		print_cluster(cluster.left, blog_names, n+1)
		print_cluster(cluster.right, blog_names, n+1)
	else:
		print blog_names[cluster.id]


def getheight(cluster):
    if cluster.left==None and cluster.right==None:  return 1
    # Otherwise the height is the same of the heights of
    # each branch
    return getheight(cluster.left)+getheight(cluster.right)


def getdepth(cluster):
    # The distance of an endpoint is 0.0
    if cluster.left==None and cluster.right==None: return 0

    # The distance of a branch is the greater of its two sides
    # plus its own distance
    return max(getdepth(cluster.left),getdepth(cluster.right))+cluster.distance


def drawnode(draw,cluster,x,y,scaling,blog_names):
    if cluster.id < 0:
        h1=getheight(cluster.left)*20
        h2=getheight(cluster.right)*20
        top=y-(h1+h2)/2
        bottom=y+(h1+h2)/2
        # Line length
        ll=cluster.distance*scaling
        # Vertical line from this cluster to children    
        draw.line((x,top+h1/2,x,bottom-h2/2),fill=(255,0,0))    

        # Horizontal line to left item
        draw.line((x,top+h1/2,x+ll,top+h1/2),fill=(255,0,0))    

        # Horizontal line to right item
        draw.line((x,bottom-h2/2,x+ll,bottom-h2/2),fill=(255,0,0))        

        # Call the function to draw the left and right nodes    
        drawnode(draw,cluster.left,x+ll,top+h1/2,scaling,blog_names)
        drawnode(draw,cluster.right,x+ll,bottom-h2/2,scaling,blog_names)
    else:   
        # If this is an endpoint, draw the item label
        draw.text((x+5,y-7),blog_names[cluster.id],(0,0,0))


def draw_cluster(cluster, blog_names, jpeg_path):
    # height and width
    h=getheight(cluster)*20
    w=1200
    depth=getdepth(cluster)

    # width is fixed, so scale distances accordingly
    scaling=float(w-150)/depth

    # Create a new image with a white background
    img=Image.new('RGB',(w,h),(255,255,255))
    draw=ImageDraw.Draw(img)

    draw.line((0,h/2,10,h/2),fill=(255,0,0))    

    # Draw the first node
    drawnode(draw,cluster,10,(h/2),scaling,blog_names)
    img.save(jpeg_path,'JPEG')

	
if __name__ == '__main__':
	col_names, blog_names, blog_data = read_data('Clustering_data/data')
	cluster = hierarchicalClustering(blog_data)
	draw_cluster(cluster, blog_names, 'Clustering_data/clusters.jpg')
	