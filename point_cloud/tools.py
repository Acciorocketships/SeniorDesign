import math
import numpy as np

'''
linear algebra matrix tool functions -------------------------------------------------------------------------------------------------
'''

#http://planning.cs.uiuc.edu/node102.html
def yaw(vector, angle, do_round=True):
	result = [math.cos(angle) * vector[0] - math.sin(angle) * vector[1],
			math.sin(angle) * vector[0] + math.cos(angle) * vector[1],
			vector[2]]
	if do_round:
		return vector_round(result, 14)
	else:
		return result

def pitch(vector, angle, do_round=True):
	result = [math.cos(angle) * vector[0] + math.sin(angle) * vector[2],
			vector[1],
			-math.sin(angle) * vector[0] + math.cos(angle) * vector[2]]
	if do_round:
		return vector_round(result, 14)
	else:
		return result

def roll(vector, angle, do_round=True):
	result = [vector[0],
			math.cos(angle) * vector[1] - math.sin(angle) * vector[2],
			math.sin(angle) * vector[1] + math.cos(angle) * vector[2]]
	if do_round:
		return vector_round(result, 14)
	else:
		return result

def identity():
	return [[1, 0, 0],
			[0, 1, 0],
			[0, 0, 1]]

def matrix_multiplier(matrix, mult):
	result = identity()
	for i in range(3):
		for j in range(3):
			result[i][j] = matrix[i][j] * mult
	return result

def matrix_add(m1, m2):
	result = identity();
	for i in range(3):
		for j in range(3):
			result[i][j] = m1[i][j] + m2[i][j]
	return result

'''
TODO figure out the kwargs thing
'''
def generate_image(func, max_i, max_j):
	t = np.zeros((max_i, max_j))
	#for i in range(max_i):
	#	for j in range(max_j):
			#t[i][j] = i + j

	return t

# https://math.stackexchange.com/questions/142821/matrix-for-rotation-around-a-vector
def rmatrix_to_vector(v, angle):
	w = [[0, -v[2], v[1]],
		 [v[2], 0, -v[0]],
		 [-v[1], v[0], 0]]
	mult = math.sin(angle)
	first = matrix_multiplier(w, mult)

	mult = 2 * pow(math.sin(angle/2), 2)
	col1 = [w[0][0], w[1][0], w[2][0]]
	col2 = [w[0][1], w[1][1], w[2][1]]
	col3 = [w[0][2], w[1][2], w[2][2]]
	w2 = [[dot_product(w[0], col1), dot_product(w[0], col2), dot_product(w[0], col3)],
		  [dot_product(w[1], col1), dot_product(w[1], col2), dot_product(w[1], col3)],
		  [dot_product(w[2], col1), dot_product(w[2], col2), dot_product(w[2], col3)]]
	second = matrix_multiplier(w2, mult)

	result = matrix_add(identity(), first)
	return matrix_add(result, second)

def normalize_matrix(matrix, high, low):
	i, j = np.shape(matrix)
	highest = -float("inf")
	lowest = float("inf")
	for x in range(i):
		highest = max(highest, max(matrix[x]))
		lowest = min(lowest, min(matrix[x]))

	current = highest - lowest
	available = high - low
	result = np.zeros((i, j))

	for x in range(i):
		for y in range(j):
			result[x][y] = (matrix[x][y] - lowest)/current * available + low
	return result

'''
linear algebra vector tool functions -------------------------------------------------------------------------------------------------
'''

def vector_round(vector, decs):
	result = []
	for num in vector:
		result.append(round(num, decs))
	return result

def dot_product(v, u):
	return v[0] * u[0] + v[1] * u[1] + v[2] * u[2]

def make_ints(v):
	return [int(v[0]), int(v[1]), int(v[2])]

def make_unit_vector(v):
	size = len_vector(v)
	return normalize(v, size)

def ortho(u, v):
	return [u[1] * v[2] - v[1] * u[2],
			-(u[0] * v[2] - v[0] * u[2]),
			u[0] * v[1] - v[0] * u[1]]

def len_vector(v):
	return math.sqrt(pow(v[0], 2) + pow(v[1], 2) + pow(v[2], 2))

def normalize(v, n):
	return [v[0]/ n, v[1]/n, v[2]/n]

def vector_multiplier(v, dist):
	return [v[0] * dist, v[1] * dist, v[2] * dist]

def invert(v):
	return [-v[0], -v[1], -v[2]]

def sum_vectors(v, u):
	return [v[0] + u[0], v[1] + u[1], v[2] + u[2]]

def angle_2_vectors(v, u):
	try:
		top = dot_product(v, u)
		bottom = len_vector(v) * len_vector(u)
		return math.acos(top/bottom)
	except:
		return 0

#projection of vector u along v direction
def projection_vector(u, v):
	top = dot_product(u, v)
	bottom = pow(len_vector(v), 2)
	mult = top/bottom
	return (v[0] * mult, v[1] * mult, v[2] * mult)

'''
point cloud tool functions -------------------------------------------------------------------------------------------------
'''

'''
Purpose: given list of points, find 2 points that defines the box that bounds all points
points = list of tuples of xyz
'''
def find_min_max(points_list):
    min_x = min([tup[0] for tup in points_list])
    max_x = max([tup[0] for tup in points_list])
    min_y = min([tup[1] for tup in points_list])
    max_y = max([tup[1] for tup in points_list])
    min_z = min([tup[2] for tup in points_list])
    max_z = max([tup[2] for tup in points_list])
    return (min_x, max_z), (min_y, max_y), (min_z, max_z)

def pts_dist(pt1, pt2):
	x = pt1[0] - pt2[0]
	y = pt1[1] - pt2[1]
	z = pt1[2] - pt2[2]
	return len_vector((x, y, z))



