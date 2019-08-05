import networkx as nx
import numpy as np
from numpy import inf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from LP_with_input import *
from edge_crossing import *
import math
from input_functions import *
import sys

EPSILON = 0.000001
# K = 100000
K = 16
# W = 0
W = 1

# print sys.argv

GD_OPTIONS = ['VANILLA', 'MOMENTUM', 'NESTEROV', 'ADAGRAD', 'RMSPROP', 'ADAM']

THIS_GD_OPTION = 'VANILLA'
if(len(sys.argv) >= 2):
	THIS_GD_OPTION = sys.argv[1]

THIS_ALPHA = 1e-3
if(len(sys.argv) >= 5):
	THIS_ALPHA = float(sys.argv[4])


THIS_NUM_ITERS = 100
if(len(sys.argv) >= 3):
	THIS_NUM_ITERS = int(sys.argv[2])

OUTER_NUM_ITERS = 30
if(len(sys.argv) >= 4):
	OUTER_NUM_ITERS = int(sys.argv[3])

print THIS_GD_OPTION
print THIS_NUM_ITERS
print OUTER_NUM_ITERS
print THIS_ALPHA

USE_NEATO_INITIAL = True
USE_INITIAL_NODE_COORDS = False

OPTIMIZE_CROSSING_ANGLE = True

#This function returns the nodes of an edge given its index in the edge list
def getNodesforEdge(index):
	return edge_list[index][0], edge_list[index][1]


# This function extracts the edge pair in the form of matrices 
# Returns two matrices A and B
# A contains [a1x, a1y; a2x a2y] 
# B contains [b1x, b1y; b2x b2y]
def getEdgePairAsMatrix(X,i,j):
	A = np.zeros((2,2))
	B = np.zeros((2,2))
	
	i1, i2 = getNodesforEdge(i)
	j1, j2 = getNodesforEdge(j)

	A[0,:] = X[i1, :]
	A[1,:] = X[i2, :]

	B[0,:] = X[j1, :]
	B[1,:] = X[j2, :]

	return A,B


def plotGraphandStats(X):
	# print(X)
	# plt.scatter(X[:,0], X[:,1], color='red')

	# for every edge in the graph
	# draw a line joining the endpoints
	# for i in range(0,m):
	# 	i1, i2 = getNodesforEdge(i)
	# 	A = np.zeros((2,2))

	# 	A[0,:] = X[i1, :]
	# 	A[1,:] = X[i2, :]
	# 	plt.plot(A[:,0] , A[:,1], color='blue')

	num_intersections = 0
	min_angle = math.pi/2.0

	print "Number of edges: ", m

	# loop through all edge pairs
	for i in range(0,m):
		for j in range(i+1,m):

			A,B = getEdgePairAsMatrix(X,i,j)
			# doIntersect(x11, y11, x12, y12, x21, y21, x22, y22)
			if(doIntersect(A[0][0], A[0][1], A[1][0], A[1][1], B[0][0], B[0][1], B[1][0], B[1][1])):
				
				# print A
				# print B
				# print i
				# print j
				res = getIntersection(A[0][0], A[0][1], A[1][0], A[1][1], B[0][0], B[0][1], B[1][0], B[1][1])
				if not res:
					continue

				x_pt, y_pt = res 
				theta = getAngleLineSeg(A[0][0], A[0][1], B[0][0], B[0][1], x_pt, y_pt)
				if theta > math.pi/2.0:
					theta = math.pi - theta
				if theta < min_angle:
					min_angle = theta

				num_intersections += 1

	print "Number of Edge Crossings: ", num_intersections
	print "Minimum Angle: ", to_deg(min_angle)

	# plt.show()


# This function computes the stress of an embedding. It takes as input the coordinates X, 
# weights (i.e. d_{ij}^(-2)), ideal distances between the nodes, and the number of nodes
# in the graph 

def stress(X, weights, distances, n):
	# print "Parameters:", X, type(X)
	s = 0.0
	for i in range(0,n):
		for j in range(i+1,n):
			norm = X[i,:] - X[j,:]
			norm = np.sqrt(sum(pow(norm,2)))
			s += weights[i,j] * pow((norm - distances[i,j]), 2)
	return s


# This function computes the stress of an embedding X. It needs weights, ideal distances, 
# and number of nodes already initialized
def stress_X(X):
	s = 0.0
	for i in range(0,n):
		for j in range(i+1,n):
			norm = X[i,:] - X[j,:]
			norm = np.sqrt(sum(pow(norm,2)))
			s += weights[i,j] * pow((norm - distances[i,j]), 2)
	return s

# def stress_Maj(X, weights, distances, n):
# 	pass

L = None
diag_1s = None
col1s_mat = None

def setupMaj(X, weights, distances, n):
	global L
	global diag_1s
	global col1s_mat

	L = -weights
	diagL = np.diag(np.sum(weights, axis = 1))
	L = L + diagL
	# delta = np.multiply(weights,distances)

	diag_1s = np.repeat(-1,n)
	diag_1s = np.diag(diag_1s)
	col1s_mat = np.zeros([n,n])


def dStressMaj_dx(X, weights, distances, n, L, diag_1s, col1s_mat):
	LZ = np.zeros([n,n])
	Z = X
	
	for i in range(0, n):
		col1s_mat_ith = np.copy(col1s_mat)
		col1s_mat_ith[:,i] = 1
		col1s_mat_ith = col1s_mat_ith + diag_1s

		LZ_ith_row = np.matmul(col1s_mat_ith,Z)
		LZ_ith_row = pow(LZ_ith_row, 2)
		LZ_ith_row = np.sum(LZ_ith_row, axis = 1)
		LZ_ith_row = np.sqrt(LZ_ith_row)
		LZ_ith_row = 1. / LZ_ith_row
		LZ_ith_row[LZ_ith_row== -inf] = 0
		LZ_ith_row[LZ_ith_row == inf] = 0

		LZ_ith_row = LZ_ith_row.transpose();

		LZ[i,:] = LZ_ith_row

	LZ = -LZ
	diagLZ = -np.diag(np.sum(LZ, axis = 1))
	LZ = LZ + diagLZ

	grad = 2 * (np.matmul(L,X) - np.matmul(LZ, Z))
	return grad


###### Initialize and load the graphs; Compute the weights and distances

# G = nx.petersen_graph()
# G = nx.complete_graph(6)
# G = nx.complete_graph(7)
# G = nx.wheel_graph(9)

# FILENAME = 'meeting_08_21_2018/input/input1'
FILENAME = 'input18/input4'

print FILENAME

G = build_networkx_graph(FILENAME + '.txt')

# TODO: Get the position in the input file
dummy1, init_node_coords, dummy2 = take_input(FILENAME + '.txt')

n = G.number_of_nodes()
m = G.number_of_edges()
edge_list = G.edges()

distances = nx.floyd_warshall(G)

# Initialize the coordinates randomly in the range [-50, 50]
X_curr = np.random.rand(n,2)*100 - 50

if USE_NEATO_INITIAL:
	# pos = nx.nx_agraph.graphviz_layout(G)
	pos = nx.nx_agraph.graphviz_layout(G, args = '-Gstart=rand')
	# Copy the coordinates from pos to X_curr
	for i in range(0,n):
		X_curr[i] = pos[i]
	# np.save("neato_layout_input1_2017.npy" ,X_curr)

if USE_INITIAL_NODE_COORDS:
	# Copy the coordinates from initial coords to X_curr
	for i in range(0,n):
		X_curr[i] = init_node_coords[i]


plotGraphandStats(X_curr)

# Z=np.copy(X_curr)
X_prev = np.copy(X_curr)

# Copy the distances into a 2D numpy array
distances = np.array([[distances[i][j] for j in distances[i]] for i in distances])
# weights = 1/(d^2)
weights = 1/pow(distances,2)
weights[weights == inf] = 0

setupMaj(X_curr, weights, distances, n)


# Define: penalties, u_params, gammas, edgesID


# penalties: a 2D array containing the penalties for each possible edge pair
# For now the penalties start with 0 and gradually increase by 1 in the next iteration
# if the crossing persists.

penalties = np.zeros((m, m))

# u_params: a 3D array containing the u vectors for each edge pair
u_params = np.zeros((m, m, 2))

# gammas: a 2D array containing the gamma values for each possible edge pair
gammas = np.zeros((m, m))

# all these variables need to be accessed as a 2D array
# with the edge pair as the i,j index of the 2D array.

#The 2D array is of size M*M where M is the number of edges. Max edges = n*(n-1)/2
# for complete graph


# This function computes the modified objective function i.e. a sum of stress and penalty function
def modified_cost(X):
	#Reshape the 1D array to a n*2 matrix
	X = X.reshape((n,2))
	return (W*stress(X, weights, distances, n)) + K*sum_penalty(X)

def max_zero(a):
	return np.maximum(0,a)

def sum_penalty(X):
	sumPenalty = 0

	# loop through all edge pairs
	for i in range(0,m):
		for j in range(i+1,m):

			# Add the penalty
			# ||(-Au - eY)+||1 + ||(Bu + (1 + Y)e)+||1
			# z_+ = max(0,z)

			# sumPenalty += penalty_ij/2 * [||(-Ai(X)ui - Yie)+||1
			# + ||(Bi(X)ui + (1 + Yi)e)+||1] * cos^2(theta)

			A,B = getEdgePairAsMatrix(X,i,j)
			if(doIntersect(A[0][0], A[0][1], A[1][0], A[1][1], B[0][0], B[0][1], B[1][0], B[1][1])):
				# x_pt, y_pt = getIntersection(A[0][0], A[0][1], A[1][0], A[1][1], B[0][0], B[0][1], B[1][0], B[1][1])
				# theta = getAngleLineSeg(A[0][0], A[0][1], B[0][0], B[0][1], x_pt, y_pt)
				
				# cos(theta) can be computed directly using the dot product between vectors
				# a = (A[1][0] - A[0][0], A[1][1] - A[0][1])
				# b = (B[1][0] - B[0][0], B[1][1] - B[0][1])
				# cos(theta) =  a.b / (||a||*||b||)
				aVec = A[1,:] - A[0,:]
				bVec = B[1,:] - B[0,:]
				cos_sq_theta = np.dot(aVec, bVec)**2
				cos_sq_theta = cos_sq_theta / (np.dot(aVec, aVec))
				cos_sq_theta = cos_sq_theta / (np.dot(bVec, bVec))

				# sumPenalty += (penalties[i][j]/2.0) * (math.cos(theta)**2) * (np.sum(max_zero(-np.matmul(A,u_params[i][j])- gammas[i][j] * np.array([1,1]))) + np.sum(max_zero(np.matmul(B,u_params[i][j])+ (1+gammas[i][j]) * np.array([1,1]))))
				sumPenalty += (penalties[i][j]/2.0) * (cos_sq_theta) * (np.sum(max_zero(-np.matmul(A,u_params[i][j])- gammas[i][j] * np.array([1,1]))) + np.sum(max_zero(np.matmul(B,u_params[i][j])+ (1+gammas[i][j]) * np.array([1,1]))))

	return sumPenalty


def optimize(X_curr):
	# Start with pivotmds or neato stress majorization or cmdscale as in the paper
	# Or use X with a random initialization
	# Currently, we start with neato stress majorization coordinates

	NUM_ITERS = OUTER_NUM_ITERS

	# set penalty to 1 if there is an edge-crossing 
	# reset penalty to 0 if there is no edge-crossing

	X = np.copy(X_curr)

	# loop through all edge pairs
	for i in range(0,m):
		for j in range(i+1,m):
			A,B = getEdgePairAsMatrix(X,i,j)
			if(doIntersect(A[0][0], A[0][1], A[1][0], A[1][1], B[0][0], B[0][1], B[1][0], B[1][1])):
				# penalties[i][j] = penalties[i][j] + 1
				penalties[i][j] = 1
			else:
				penalties[i][j] = 0

	#TODO: Be careful that the optimization does not monotonically decrease the cost function
	# This is basically one way to phrase "Unitl Satisfied" and is a very rigid way
	# Another way is to count the number of edge crossings in the graph
	# If the no. of edge crossings remains the same for a long time 
	# or the no. of edge crossings increases significantly
	# or the no. of edge crossings remains within a same range for a long time
	# then stop the optimization and store the embedding with the best edge crossing

	while 1: 

		# For all intersecting edge pairs
		# compute optimal u and gammas using the LP subroutine
		X = np.copy(X_curr)

		# loop through all edge pairs
		for i in range(0,m):
			for j in range(i+1,m):

				A,B = getEdgePairAsMatrix(X,i,j)
				# doIntersect(x11, y11, x12, y12, x21, y21, x22, y22)
				if(doIntersect(A[0][0], A[0][1], A[1][0], A[1][1], B[0][0], B[0][1], B[1][0], B[1][1])):
					
					# Call the LP module for optimization
					#Input to the module are two edges A and B
					# A is a 2*2 matrix that contains [a1x a1y; a2x a2y]
					# B is a 2*2 matrix that contains [b1x b1y; b2x b2y]
					# In the LP module, ax = a1x, ay = a1y, bx = a2x, by = a2y
					# cx = b1x, cy = b1y, dx = b2x, dy = b2y
					#u,gamma = LP_optimize(A,B)
					#u_params[i][j] = u
					
					# ux = get_ux(A[0][0], A[0][1], A[1][0], A[1][1], B[0][0], B[0][1], B[1][0], B[1][1])
					# uy = get_uy(A[0][0], A[0][1], A[1][0], A[1][1], B[0][0], B[0][1], B[1][0], B[1][1])
					# gamma = get_gamma(A[0][0], A[0][1], A[1][0], A[1][1], B[0][0], B[0][1], B[1][0], B[1][1])
					
					ux, uy, gamma = get_u_gamma(A[0][0], A[0][1], A[1][0], A[1][1], B[0][0], B[0][1], B[1][0], B[1][1])

					u_params[i][j][0] = ux
					u_params[i][j][1] = uy
					gammas[i][j] = gamma
	
		# Use gradient descent to optimize the modified_cost function
		# keep the X as a flattened 1D array and reshape it inside the 
		# modified_cost function as a 2D array/matrix
		X = X.flatten()

		# res = minimize(modified_cost, X, method='BFGS', jac=jacobian_Mod_Cost, options={'disp': True})
		
		# Use custom GD method instead of standard gradient descent method

		# res = minimize_with_gd(X)
		# res = minimize_with_momentum(X)
		# res = minimize_with_nesterov_momentum(X)
		# res = minimize_with_adagrad(X, num_iters=50)
		# res = minimize_with_rmsprop(X, num_iters=50)
		# res = minimize_with_adam(X)

		# GD_OPTIONS = ['VANILLA', 'MOMENTUM', 'NESTEROV', 'ADAGRAD', 'RMSPROP', 'ADAM']


		if(THIS_GD_OPTION == 'VANILLA'):
			res = minimize_with_gd(X, num_iters=THIS_NUM_ITERS, alpha=THIS_ALPHA)
		elif(THIS_GD_OPTION == 'MOMENTUM'):
			res = minimize_with_momentum(X, num_iters=THIS_NUM_ITERS, alpha=THIS_ALPHA)
		elif(THIS_GD_OPTION == 'NESTEROV'):
			res = minimize_with_nesterov_momentum(X, num_iters=THIS_NUM_ITERS, alpha=THIS_ALPHA)
		elif(THIS_GD_OPTION == 'ADAGRAD'):
			res = minimize_with_adagrad(X, num_iters=THIS_NUM_ITERS, alpha=THIS_ALPHA)
		elif(THIS_GD_OPTION == 'RMSPROP'):
			res = minimize_with_rmsprop(X, num_iters=THIS_NUM_ITERS, alpha=THIS_ALPHA)
		elif(THIS_GD_OPTION == 'ADAM'):
			res = minimize_with_adam(X, num_iters=THIS_NUM_ITERS, alpha=THIS_ALPHA)


		X_prev = np.copy(X_curr)
		
		# X_curr = res.x.reshape((n,2))
		X_curr = res.reshape((n,2))

		# print(res.x)
		print "Iter:" + str(OUTER_NUM_ITERS - NUM_ITERS)
		# print(X_curr)
		plotGraphandStats(X_curr)

		X = np.copy(X_curr)

		# increase penalty by 1 if the crossing persists 
		# reset penalties to 0 if the crossing disappears

		# loop through all edge pairs
		for i in range(0,m):
			for j in range(i+1,m):
				A,B = getEdgePairAsMatrix(X,i,j)
				if(doIntersect(A[0][0], A[0][1], A[1][0], A[1][1], B[0][0], B[0][1], B[1][0], B[1][1])):
					# penalties[i][j] = penalties[i][j] + 1
					penalties[i][j] = 1
				else:
					penalties[i][j] = 0

		# if((modified_cost(X_prev) - modified_cost(X_curr)) / modified_cost(X_prev) < EPSILON):
		# 	return X_curr

		NUM_ITERS -=1 

		if(NUM_ITERS<=0):
			return X_curr


# Construct the laplacian matrix of the weights
def constructLaplacianMatrix():
	L = -weights
	L[L==-inf] = 0
	diagL = np.diag(np.sum(weights, axis = 1))
	L = L + diagL
	return L

# This function computes the jacobian/gradient of the modified cost function at the point X
def jacobian_Mod_Cost(X):
	#Reshape the 1D array to a n*2 matrix
	global weights
	global distances
	global n
	global W
	global K
	X = X.reshape((n,2))
	dmodCost_dx = np.zeros((n,2))
	dmodCost_dx = dmodCost_dx + W*dStressMaj_dx(X, weights, distances, n, L, diag_1s, col1s_mat)
	dmodCost_dx = dmodCost_dx + K*dSumPenalty_dx(X)
	return dmodCost_dx.flatten()

# This function computes the gradient of stress function at the point X
def dStress_dx(X, weights, distances, n):
	
	dStress_dxArr = np.zeros((n,2))
	# for every node/point
	for i in range(0,n):
		# for every node's contribution to this node
		for j in range(0,n):
			diff = X[i,:] - X[j,:]
			# norm_diff = np.linalg.norm(diff)
			norm_diff = np.sqrt(sum(pow(diff,2)))
			if(norm_diff!=0):
				dStress_dxArr[i,:] += weights[i,j] * (norm_diff - distances[i,j]) * diff / norm_diff
		dStress_dxArr[i,:] *= 2

	return dStress_dxArr

# This function computes the gradient of sum penalty at the point X
def dSumPenalty_dx(X):

	global u_params
	global gammas
	global penalties
	
	dSumPenalty_dxArr = np.zeros((n,2))

	# Keep a list of all edge pair (indices) that intersect
	list_crossings = []
	# list_crossings.append([5,1])
	# list_crossings = np.array(list_crossings)

	# For each vertex, Keep a list of all edge pairs that intersect where 
	# the vertex is involved
	# Store the edge that intersects
	# Keep track of whether the edge falls to the left/right side of the edge pair
	vertex_crossing_info = {}

	# loop through all edge pairs
	for i in range(0,m):
		for j in range(i+1,m):

			A,B = getEdgePairAsMatrix(X,i,j)
			# doIntersect(x11, y11, x12, y12, x21, y21, x22, y22)
			if(doIntersect(A[0][0], A[0][1], A[1][0], A[1][1], B[0][0], B[0][1], B[1][0], B[1][1])):
				list_crossings.append([i,j])

				# for each vertex appearing in the edge pair
				# store the edge pair, the edge involved, and the left/right side on the edge pair
				i1, i2 = getNodesforEdge(i)
				j1, j2 = getNodesforEdge(j)

				if(not(i1 in vertex_crossing_info)):
					vertex_crossing_info[i1] = []
				if(not(i2 in vertex_crossing_info)):
					vertex_crossing_info[i2] = []
				if(not(j1 in vertex_crossing_info)):
					vertex_crossing_info[j1] = []
				if(not(j2 in vertex_crossing_info)):
					vertex_crossing_info[j2] = []

				i1_obj = {}
				i1_obj['edge_pair'] = [i,j]
				i1_obj['edge'] = i
				i1_obj['left'] = True

				i2_obj = {}
				i2_obj['edge_pair'] = [i,j]
				i2_obj['edge'] = i
				i2_obj['left'] = True

				j1_obj = {}
				j1_obj['edge_pair'] = [i,j]
				j1_obj['edge'] = j
				j1_obj['left'] = False

				j2_obj = {}
				j2_obj['edge_pair'] = [i,j]
				j2_obj['edge'] = j
				j2_obj['left'] = False

				vertex_crossing_info[i1].append(i1_obj)
				vertex_crossing_info[i2].append(i2_obj)
				vertex_crossing_info[j1].append(j1_obj)
				vertex_crossing_info[j2].append(j2_obj)


	# Convert the 2D list into 2D numpy array
	list_crossings = np.array(list_crossings)

	# Loop through the vertex crossings list
	# For each vertex 
	
	for node_index in vertex_crossing_info:
		edge_pairs = vertex_crossing_info[node_index]

		# For each contributing edge pair
		for edge_pairObj in edge_pairs:

			# Get the coordinates of the node
			node_coords = X[node_index, :]

			this_I = edge_pairObj['edge_pair'][0]
			this_J = edge_pairObj['edge_pair'][1]

			this_edge = edge_pairObj['edge']

			A,B = getEdgePairAsMatrix(X,this_I,this_J)

			edgecross_penalty = (np.sum(max_zero(-np.matmul(A,u_params[this_I][this_J])- gammas[this_I][this_J] * np.array([1,1]))) + np.sum(max_zero(np.matmul(B,u_params[this_I][this_J])+ (1+gammas[this_I][this_J]) * np.array([1,1]))))
			d_edgecross_penalty = np.array([0,0])
	
			aVec = A[1,:] - A[0,:]
			bVec = B[1,:] - B[0,:]
			cos_sq_theta = np.dot(aVec, bVec)**2
			cos_sq_theta = cos_sq_theta / (np.dot(aVec, aVec))
			cos_sq_theta = cos_sq_theta / (np.dot(bVec, bVec))

			d_cos_sq_theta = np.array([0,0])

			# Retrieve the u_params, gamma and penalty
			# this_ux = u_params[this_I][this_J][0]
			# this_uy = u_params[this_I][this_J][1]
			this_u = u_params[this_I][this_J]
			this_gamma = gammas[this_I][this_J]
			this_penalty = penalties[this_I][this_J]

			# Also include the penalty term in the gradient

			# If the vertex is on left side i.e. associated with -Au - gamma*e
			# if (-Au-gamma*e) > 0, add the derivative -ux, -uy for xi & yi
			# (-xi*ux - yi*uy - gamma > 0)

			if(edge_pairObj['left']):
				if(-np.dot(node_coords, this_u) - this_gamma > 0):
					# dSumPenalty_dxArr[node_index, :] += (-this_u)*this_penalty/2.0
					d_edgecross_penalty = (-this_u)

			# If the vertex is on right side i.e. associated with Bu + (1+gamma)*e
			# if (xi*ux + yi*uy + (1+gamma)) > 0, add the derivative
			# ux, uy for xi & yi
			else:
				if(np.dot(node_coords, this_u) + 1 + this_gamma > 0):
					# dSumPenalty_dxArr[node_index, :] += (this_u)*this_penalty/2.0
					d_edgecross_penalty = (this_u)

			# Note: Add a separate term for crossing angle penalty
			# cos^theta = (a.b)^2 / (||a||^2 * ||b||^2)

			# cos^theta = [(cx - ax)(dx-bx) + (cy-ay)(dy-by)]^2 
			# /[[(cx-ax)^2 + (cy-ay)^2][(dx-bx)^2 + (dy-by)^2]]

			# a = (cx-ax, cy-ay); b = (dx-bx, dy-by)
			
			# d/dcx = d/dcx (constant * f(cx)/ g(cx))
			
			# constant = 1/[(dx-bx)^2 + (dy-by)^2] 
			
			# d/dcx = d/dcx (constant * f(cx) * 1/g(cx))
			
			# = constant * (f(cx)*d/dcx(1/g(cx)) + (1/g(cx)) * d/dcx(f(cx)))
			
			# df(cx)/dcx = 2*sqrt(f(cx))*(dx-bx)
			
			# df(cy)/dcy = 2*sqrt(f(cx))*(dy-by)
			
			# d(1/g(cx))/dcx = - (1 / g(cx)^2) * 2(cx-ax)
			
			# d(1/g(cy))/dcy = - (1 / g(cy)^2) * 2(cy-ay)

			# Start the angle derivative from here
			# if(OPTIMIZE_CROSSING_ANGLE):

			A, B = getEdgePairAsMatrix(X, this_I, this_J)

			if(not(edge_pairObj['left'])):
				A, B = B, A

			temp_ind1, temp_ind2 = getNodesforEdge(this_edge)

			if(temp_ind2 == node_index):
				# Swap the rows of A 
				temp = np.zeros((2,2))
				temp[0,:] = A[1,:]
				temp[1,:] = A[0,:]
				A = temp

			diff_A = A[0,:] - A[1,:]
			diff_B = B[0,:] - B[1,:]

			# fAB = (cx-ax)(dx-bx) + (cy-ay)(dy-by)
			# gAB = 1/[(cx-ax)^2 + (cy-ay)^2]
			# constant = 1/[(dx-bx)^2 + (dy-by)^2]

			# The function cos^2(theta) = constant * fAB^2 * gAB
			# dcos2theta = constant * (fAB^2 * dg_AB + dfsq_AB * gAB)

			fAB = np.dot(diff_A, diff_B)

			dfsq_AB = 2.0 * fAB * diff_B

			gAB = 1.0/((np.dot(diff_A, diff_A)))
			
			# dg_ab = - 2.0 * (1/(sum(np.dot(diff_A, diff_A)))**2) * (diffA)
			dg_ab = -2.0 * (gAB ** 2) * (diff_A) 

			const_term = 1.0 / (np.dot(diff_B, diff_B))

			d_cos_sq_theta = const_term * (fAB**2 * dg_ab + dfsq_AB * gAB)

			## Use the product rule to combine these two derivatives
			dSumPenalty_dxArr[node_index, :] += (this_penalty/2.0) * (edgecross_penalty*d_cos_sq_theta + d_edgecross_penalty*cos_sq_theta)

	return dSumPenalty_dxArr

def minimize_with_gd(X, num_iters=100, beta1 = 0.9, beta2 = 0.999, alpha = 1e-3, eps = 1e-8):
	for t in range(1, num_iters+1):
		grad = jacobian_Mod_Cost(X)
		X -= alpha * grad
		
	return X

def minimize_with_momentum(X, num_iters=100, gamma = 0.9, alpha = 1e-3, eps = 1e-8):
	V = np.zeros(X.shape)

	for t in range(1, num_iters+1):
		grad = jacobian_Mod_Cost(X)
		V = gamma * V + alpha *grad
		X -= V
		
	return X

def minimize_with_nesterov_momentum(X, num_iters=100, gamma = 0.9, alpha = 1e-3, eps = 1e-8):
	V = np.zeros(X.shape)

	for t in range(1, num_iters+1):
		grad_ahead = jacobian_Mod_Cost(X + gamma*V)
		
		V = gamma * V + alpha *grad_ahead
		X -= V
		
	return X

def minimize_with_adagrad(X, num_iters=100, beta1 = 0.9, beta2 = 0.999, alpha = 1e-3, eps = 1e-8):
	R = np.zeros(X.shape)
	
	for t in range(1, num_iters+1):
		grad = jacobian_Mod_Cost(X)
		
		R += grad**2
		
		X -= alpha * grad / (np.sqrt(R) + eps)
		
	return X

def minimize_with_rmsprop(X, num_iters=100, gamma = 0.9, beta2 = 0.999, alpha = 1e-3, eps = 1e-8):
	R = np.zeros(X.shape)
	
	for t in range(1, num_iters+1):
		grad = jacobian_Mod_Cost(X)
		
		R = gamma*R + (1-gamma)*(grad**2)
		
		X -= alpha * grad / (np.sqrt(R) + eps)
		
	return X

# This function is a variant of gradient descent which uses adam optimizer
# This uses both the momentum of  updates andparameter the rate of 
# individual parameter updates

def minimize_with_adam(X, num_iters=100, beta1 = 0.9, beta2 = 0.999, alpha = 1e-3, eps = 1e-8):
	M = np.zeros(X.shape)
	R = np.zeros(X.shape)
	
	for t in range(1, num_iters+1):
		grad = jacobian_Mod_Cost(X)
		
		M = beta1*M + (1-beta1)*grad
		R = beta2*R + (1-beta2)*(grad**2)
		
		m_hat = M / (1 - beta1**(t))
		r_hat = R / (1 - beta2**(t))
		
		X -= alpha * m_hat / (np.sqrt(r_hat) + eps)
		
	return X


X_curr = optimize(X_curr)
plotGraphandStats(X_curr)













