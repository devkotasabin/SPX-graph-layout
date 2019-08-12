import networkx as nx
import numpy as np
from numpy import inf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from LP_with_input import *
from edge_crossing import *
from networkx.drawing.nx_pydot import write_dot
from plot_layout_statistics import parse_dot_file
from input_functions import *
import math
import sys
import time

if len(sys.argv)<4:
 print('usage:python3 main_reyan.py normalize(0/1) file_prefix(er_10_0.6) output_folder')
 quit()

EPSILON = 0.000001

NUM_ITERATIONS = 3
#NUM_ITERATIONS = 10
#NUM_ITERATIONS = 5

USE_NUM_ITERS = True
IS_LOG_COST_FUNCTION = True
NORMALIZE = int(sys.argv[1])

K = 1000
# K = 1
# W = 0
W = 1

#NUM_RUNS = 5
# NUM_RUNS = 2
# NUM_RUNS = 10
#NUM_RUNS = 6
NUM_RUNS = 1

# G = []#graph
# n = []
# m = []
edge_list = []
distances = []
X_curr = []
X_prev = []
weights = []
penalties = []
u_params = []
gammas = []



#USE_NEATO_INITIAL = True
USE_NEATO_INITIAL = False

# 5 runs for each parameter
# 10 different parameters combination of K and W
# W = [0, 1]
# K = [1, 10, 100, 1000, 10000]
# Cost Function for every run
# Number of Crossings

#FILENAME = 'er_10_0.6'
FILENAME = sys.argv[2]
OUTPUT_FOLDER = sys.argv[3]

#G = nx.erdos_renyi_graph(10, 0.4)
G = build_directed_networkx_graph(OUTPUT_FOLDER+'/'+FILENAME+'.txt')
G_undirected = build_networkx_graph(OUTPUT_FOLDER+'/'+FILENAME+'.txt')
n = G.number_of_nodes()
if n!=max(G.nodes())+1:
 print('The graph is probably disconnected')
 quit()
m = G.number_of_edges()

#W_start = 0
W_start = 1
W_end = 2

#K_start = -5
#K_end = 6

K_start = -3
K_end = 4

#K_start = 0
#K_end = 2

if len(G.edges())>50:
 NUM_ITERATIONS = 1
 K_start = -2
 K_end = 3

NUMBER_OF_CROSSINGS = -np.ones((W_end-W_start, K_end-K_start, NUM_RUNS));
COST_FUNCTIONS = -np.ones((W_end-W_start, K_end-K_start, NUM_RUNS, NUM_ITERATIONS));
X_new = np.zeros((W_end-W_start, K_end-K_start, NUM_RUNS, n, 2));
init_X = np.zeros((W_end-W_start, K_end-K_start, NUM_RUNS, n, 2));

total_u_gamma_time = 0
total_gradient_descent_time = 0
total_stress_time = 0
total_sum_penalty_time = 0
total_modified_cost_time = 0
total_penalties_after_grad_desc_time = 0

def is_upward_drawing(G, X):
  for e in G.edges():
    u, v = e
    if X[u][1] > X[v][1]:
      return False
  return True


def runOptimizer(G, W, K, wi, ki, ii):
	###### Initialize and load the graphs; Compute the weights and distances

	# G = nx.petersen_graph()
	# G = nx.complete_graph(10)
	# G = nx.complete_graph(7)
	# G = nx.wheel_graph(9)

	global edge_list
	global distances
	global X_curr
	global X_prev
	global weights
	global penalties
	global u_params
	global gammas
	global n
	global m

	n = G.number_of_nodes()
	m = G.number_of_edges()
	#edge_list = G.edges()
	for e in G.edges():
		u, v = e
		tmp = []
		tmp.append(u)
		tmp.append(v)
		edge_list.append(tmp)

	distances = nx.floyd_warshall(G_undirected)

	# Initialize the coordinates randomly in the range [-50, 50]
	X_curr = np.random.rand(n,2)*100 - 50

	if USE_NEATO_INITIAL:
		#pos = nx.nx_agraph.graphviz_layout(G)
		node_coords, edge_list = parse_dot_file(OUTPUT_FOLDER+'/run_neato_'+FILENAME+'_'+str(ii)+'.dot')
		# Copy the coordinates from pos to X_curr
		for i in range(0,n):
			#X_curr[i] = pos[i]
			tmp = np.zeros((2))
			tmp[0] = node_coords[i][0]
			tmp[1] = node_coords[i][1]
			X_curr[i] = tmp

	init_X[wi][ki][ii] = X_curr

	# plotGraphandStats(X_curr)

	# Z=np.copy(X_curr)
	X_prev = np.copy(X_curr)

	# Copy the distances into a 2D numpy array
	distances = np.array([[distances[i][j] for j in distances[i]] for i in distances])
	# weights = 1/(d^2)
	weights = 1/pow(distances,2)
	weights[weights == inf] = 0


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

	return optimize(X_curr, wi, ki, ii)
	# plotGraphandStats(X_curr)



#This function returns the nodes of an edge given its index in the edge list
def getNodesforEdge(index):
	#print(index)
	#print(edge_list)
	#print('type(edge_list)')
	#print(type(edge_list))
	#print('type(edge_list[index])')
	#print(type(edge_list[index]))
	#print(str(edge_list[index]))
	#print(str(edge_list[index][0]))
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

def num_crossings(G,X):
	num_intersections = 0

	# print X

	# print "Number of edges: ", m

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
				
				num_intersections += 1

	return num_intersections


def plotGraphandStats(X):
	print(X)
	plt.scatter(X[:,0], X[:,1], color='red')

	# for every edge in the graph
	# draw a line joining the endpoints
	for i in range(0,m):
		i1, i2 = getNodesforEdge(i)
		A = np.zeros((2,2))

		A[0,:] = X[i1, :]
		A[1,:] = X[i2, :]
		plt.plot(A[:,0] , A[:,1], color='blue')

	num_intersections = 0

	print("Number of edges: ", m)

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
				
				num_intersections += 1

	print("Number of Edge Crossings: ", num_intersections)
	plt_title = "Number of Edge Crossings: " + str(num_intersections) 
	plt.title(plt_title)
	plt.show()


# This function computes the stress of an embedding. It takes as input the coordinates X, 
# weights (i.e. d_{ij}^(-2)), ideal distances between the nodes, and the number of nodes
# in the graph 

def stress(X, weights, distances, n):
	global total_stress_time
	start_time = time.time()
	# print "Parameters:", X, type(X)
	s = 0.0
	for i in range(0,n):
		for j in range(i+1,n):
			norm = X[i,:] - X[j,:]
			norm = np.sqrt(sum(pow(norm,2)))
			#if distances[i,j]==math.inf:continue
			s += weights[i,j] * pow((norm - distances[i,j]), 2)
	total_stress_time = total_stress_time + (time.time() - start_time)
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




# This function computes the modified objective function i.e. a sum of stress and penalty function
def modified_cost(X):
	global total_modified_cost_time
	start_time = time.time()
	global NORMALIZE
	#Reshape the 1D array to a n*2 matrix
	X = X.reshape((n,2))
	return_val = 0.0
	if NORMALIZE==0:
		return_val = (W*stress(X, weights, distances, n)) + K*sum_penalty(X)
	else:
		s = stress(X, weights, distances, n)
		return_val = (W*s) + K*sum_penalty(X)/(m*m)
	#print('Time to compute modified_cost: ' + str((time.time() - start_time)) + ' seconds')
	total_modified_cost_time = total_modified_cost_time + (time.time() - start_time)
	return return_val

def max_zero(a):
	return np.maximum(0,a)

def sum_penalty(X):
	global total_sum_penalty_time
	start_time = time.time()
	sumPenalty = 0

	# loop through all edge pairs
	for i in range(0,m):
		for j in range(i+1,m):

			# Add the penalty
			# ||(-Au - eY)+||1 + ||(Bu + (1 + Y)e)+||1
			# z_+ = max(0,z)

			# sumPenalty += penalty_ij/2 * [||(-Ai(X)ui - Yie)+||1
			# + ||(Bi(X)ui + (1 + Yi)e)+||1]

			A,B = getEdgePairAsMatrix(X,i,j)
			if(doIntersect(A[0][0], A[0][1], A[1][0], A[1][1], B[0][0], B[0][1], B[1][0], B[1][1])):
				sumPenalty += (penalties[i][j]/2.0) * (np.sum(max_zero(-np.matmul(A,u_params[i][j])- gammas[i][j] * np.array([1,1]))) + np.sum(max_zero(np.matmul(B,u_params[i][j])+ (1+gammas[i][j]) * np.array([1,1]))))
	total_sum_penalty_time = total_sum_penalty_time + (time.time() - start_time)
	return sumPenalty

def optimize(X_curr, wi, ki, ii):
	# Start with pivotmds or neato stress majorization or cmdscale as in the paper
	# Or use X with a random initialization
	# Currently, we start with neato stress majorization coordinates

	# set penalty to 1 if there is an edge-crossing 
	# reset penalty to 0 if there is no edge-crossing

	global total_u_gamma_time, total_gradient_descent_time, total_penalties_after_grad_desc_time

	X = np.copy(X_curr)

	start_time = time.time()
	# loop through all edge pairs
	for i in range(0,m):
		for j in range(i+1,m):

			#print(m, i, j)

			A,B = getEdgePairAsMatrix(X,i,j)
			if(doIntersect(A[0][0], A[0][1], A[1][0], A[1][1], B[0][0], B[0][1], B[1][0], B[1][1])):
				# penalties[i][j] = penalties[i][j] + 1
				penalties[i][j] = 1
			else:
				penalties[i][j] = 0
	print('Time to count crossings at the beginning: ' + str((time.time() - start_time)) + ' seconds')

	rows = []
	lb = []
	ub = []
	for e in G.edges():
		u, v = e
		row = []
		for w in G.nodes():
			row.append(0)
			if w==u:
				row.append(-1)
			elif w==v:
				row.append(1)
			else:
				row.append(0)
		#X[v][1] = X[u][1] + .001
		rows.append(row)
		lb.append(.001)
		ub.append(np.inf)
	cons = LinearConstraint(rows, lb, ub)

	#TODO: Be careful that the optimization does not monotonically decrease the cost function
	# This is basically one way to phrase "Unitl Satisfied" and is a very rigid way
	# Another way is to count the number of edge crossings in the graph
	# If the no. of edge crossings remains the same for a long time 
	# or the no. of edge crossings increases significantly
	# or the no. of edge crossings remains within a same range for a long time
	# then stop the optimization and store the embedding with the best edge crossing

	num_iters = 0
	while 1: 
		num_iters += 1

		# For all intersecting edge pairs
		# compute optimal u and gammas using the LP subroutine
		X = np.copy(X_curr)

		start_time = time.time()
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
		print('Time to determine ux, uy and gamma: ' + str((time.time() - start_time)) + ' seconds')
		total_u_gamma_time = total_u_gamma_time + (time.time() - start_time)
	
		# Use gradient descent to optimize the modified_cost function
		# keep the X as a flattened 1D array and reshape it inside the 
		# modified_cost function as a 2D array/matrix
		X = X.flatten()

		start_time = time.time()
		#res = minimize(modified_cost, X, method='BFGS', options={'disp': True})
		res = minimize(modified_cost, X, method='trust-constr', options={'disp': True}, constraints=cons)
		print('Time to optimize using BFGS: ' + str((time.time() - start_time)) + ' seconds')
		total_gradient_descent_time = total_gradient_descent_time + (time.time() - start_time)
		X_prev = np.copy(X_curr)
		X_curr = res.x.reshape((n,2))
		
		print (str(W) + " " + str(K) + " " + str(ii) + " " + str(num_iters)) 
		print(res.x)


		if IS_LOG_COST_FUNCTION:
			COST_FUNCTIONS[wi][ki][ii][num_iters-1] = modified_cost(res.x)

		X = np.copy(X_curr)

		# increase penalty by 1 if the crossing persists 
		# reset penalties to 0 if the crossing disappears

		start_time = time.time()
		# loop through all edge pairs
		num_intersections = 0
		for i in range(0,m):
			for j in range(i+1,m):
				A,B = getEdgePairAsMatrix(X,i,j)
				if(doIntersect(A[0][0], A[0][1], A[1][0], A[1][1], B[0][0], B[0][1], B[1][0], B[1][1])):
					# penalties[i][j] = penalties[i][j] + 1
					penalties[i][j] = 1
					num_intersections += 1
				else:
					penalties[i][j] = 0
		print('num_intersections: ', num_intersections)
		print('is_upward_drawing: ', is_upward_drawing(G, X))
		total_penalties_after_grad_desc_time = total_penalties_after_grad_desc_time + (time.time() - start_time)
		

		if (not USE_NUM_ITERS):
			if((modified_cost(X_prev) - modified_cost(X_curr)) / modified_cost(X_prev) < EPSILON):
				return X_curr
		else:
			if(num_iters >= NUM_ITERATIONS):
				return X_curr


# Construct the laplacian matrix of the weights
def constructLaplacianMatrix():
	L = -weights
	L[L==-inf] = 0
	diagL = np.diag(np.sum(weights, axis = 1))
	L = L + diagL
	return L

wi=0
#for W in range(0,2):
for W in range(W_start, W_end):
	ki=0
	#for K in [1, 10, 100, 1000, 10000]:
	for K in [math.pow(2,i) for i in range(K_start,K_end)]:
		ii=0
		for i in range(0,NUM_RUNS):
			if(ii<(NUM_RUNS/2)):
				#USE_NEATO_INITIAL = True
				USE_NEATO_INITIAL = False
			else:
				USE_NEATO_INITIAL = False
			resultX = runOptimizer(G, W, K, wi, ki, ii)	
			X_new[wi][ki][ii] = resultX
			NUMBER_OF_CROSSINGS[wi][ki][ii]=num_crossings(G,resultX)

			ii = ii+1
		ki = ki+1
	wi = wi+1

# Write the graph into the dot file
#write_dot(G, 'output/' + FILENAME + '.dot')
write_networx_graph(G, 'output/' + FILENAME + '.txt')

normalize_str = '_wo_norm'
if NORMALIZE==1:
 normalize_str = '_norm'
# Write all the other arrays into another file
np.save(OUTPUT_FOLDER + '/' + FILENAME + '_ncr' + normalize_str, NUMBER_OF_CROSSINGS)
np.save(OUTPUT_FOLDER + '/' + FILENAME + '_cost' + normalize_str, COST_FUNCTIONS)
np.save(OUTPUT_FOLDER + '/' + FILENAME + '_xy' + normalize_str, X_new)
np.save(OUTPUT_FOLDER + '/' + FILENAME + '_init_xy' + normalize_str, init_X)

# To Read it back
# np.load(fname + '.npy')

print('total_u_gamma_time: '+str(total_u_gamma_time))
print('total_gradient_descent_time: ' + str(total_gradient_descent_time))
print('total_stress_time: ' + str(total_stress_time))
print('total_sum_penalty_time: ' + str(total_sum_penalty_time))
print('total_modified_cost_time: ' + str(total_modified_cost_time))
print('total_penalties_after_grad_desc_time: ' + str(total_penalties_after_grad_desc_time))








