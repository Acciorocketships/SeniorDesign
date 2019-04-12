import math
import numpy as np
import os
import cv2
import subprocess
import copy
from PIL import Image
from sklearn.neighbors import NearestNeighbors


'''
general math tool functions -------------------------------------------------------------------------------------------------
'''

def round_down_multiple_of(to_round, base):
    multiplier = int(to_round/base)
    return multiplier * base

def round_up_multiple_of(to_round, base):
    multiplier = int(to_round/base) + 1
    return multiplier * base



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

#https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
def flann_sift(prev_img, new_img, fi_kd=0, trees=5, checks=50, k=2):
    img1 = prev_img
    img2 = new_img
    tester = prev_img

    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    index_params = dict(algorithm=fi_kd, trees=trees)
    search_params = dict(checks=checks)  

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=k)
    matched_kp1 = []
    matched_kp2 = []

    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matched_kp1.append(tuple(round_coords(kp1[i].pt)))
            matched_kp2.append(tuple(round_coords(kp2[m.trainIdx].pt)))

    #cv2.imwrite("test1.png", prev_img)
    #cv2.imwrite("test2.png", new_img)
    return matched_kp1, matched_kp2


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


'''
TODO fix this
'''
def normalize_matrix(matrix, high, low):
    i, j = np.shape(matrix)
    highest = -float("inf")
    lowest = float("inf")
    for x in range(i):
        highest = max(highest, max(matrix[x]))
        lowest = min(lowest, min(matrix[x]))

    print(highest)
    print(lowest)

    current = highest - lowest
    available = high - low
    result = np.zeros((i, j))

    for x in range(i):
        for y in range(j):
            if matrix[x][y] != 0:
                result[x][y] = 255 - int(float(matrix[x][y] - lowest)/current * available + low)
            else:
                result[x][y] = 0
    print(result)
    return result

def make_greyscale(arr):
    return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

def padding_dist(m, d):
    return m - d

#p r y = pitch roll yaw of camera's view
def pry_to_view(view, do_round):
    p, r, y = view
    vect = pitch([0,0,1], p, do_round=do_round)
    vect = roll(vect, r, do_round=do_round)
    vect = yaw(vect, y, do_round=do_round)
    return vect

#lp, lr, ly = pitch roll raw of left side of camera, needed to determine rotation of frame about direction of view vector
def pry_to_horiz(left_view, do_round):
    lp, lr, ly = left_view
    horiz = pitch([-1,0,0], lp, do_round=do_round)
    horiz = roll(horiz, lr, do_round=do_round)
    horiz = yaw(horiz, ly, do_round=do_round)
    return horiz


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
    bottom = pow(float(len_vector(v)), 2.0)
    mult = float(top)/float(bottom)
    return (float(v[0]) * float(mult), float(v[1]) * float(mult), float(v[2]) * float(mult))

'''
point cloud tool functions -------------------------------------------------------------------------------------------------
'''

'''
Purpose: given list of points, find 2 points that defines the box that bounds all points
points = list of tuples of xyz
'''
def find_min_max(points_list):
    if len(points_list) == 0:
        return (0, 0), (0, 0), (0, 0)
    min_x = min([tup[0] for tup in points_list])
    max_x = max([tup[0] for tup in points_list])
    min_y = min([tup[1] for tup in points_list])
    max_y = max([tup[1] for tup in points_list])
    min_z = min([tup[2] for tup in points_list])
    max_z = max([tup[2] for tup in points_list])
    return (min_x, max_x), (min_y, max_y), (min_z, max_z)

def find_bounding_box_center(points_list):
    x, y, z = find_min_max(points_list)
    mid_x = (x[0] + x[1])/2
    mid_y = (y[0] + y[1])/2
    mid_z = (z[0] + z[1])/2
    return (mid_x, mid_y, mid_z)

def pts_dist(pt1, pt2):
    x = pt1[0] - pt2[0]
    y = pt1[1] - pt2[1]
    z = pt1[2] - pt2[2]
    return len_vector((x, y, z))

#https://stackoverflow.com/questions/20120384/iterative-closest-point-icp-implementation-on-python?fbclid=IwAR17PnUyQlR28q9GbiAGvyL25i6cmMHe4MytMQvRnjDnuK9tdn8gQz593qk
def icp(a, b, init_pose=(0,0,0), no_iterations = 13):
    src = np.array([a.T], copy=True).astype(np.float32)
    dst = np.array([b.T], copy=True).astype(np.float32)

    Tr = np.array([[np.cos(init_pose[2]),-np.sin(init_pose[2]),init_pose[0]],
                   [np.sin(init_pose[2]), np.cos(init_pose[2]),init_pose[1]],
                   [0,                    0,                   1          ]])

    src = cv2.transform(src, Tr[0:2])

    for i in range(no_iterations):
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(dst[0])
        distances, indices = nbrs.kneighbors(src[0])
        T = cv2.estimateRigidTransform(src, dst[0, indices.T], False)
        src = cv2.transform(src, T)
        Tr = np.dot(Tr, np.vstack((T,[0,0,1])))
    return Tr[0:2]


def round_coords(tup):
    result = []
    for x in range(len(tup)):
        result.append(int(tup[x]))
    return result

'''
detection class -------------------------------------------------------------------------------------------------
'''

def in_coords(frame, pt):
    bl = frame[0]
    tr = frame[1]

    if pt[0] >= bl[0] and pt[0] <= tr[0] and pt[1] >= bl[1] and pt[1] <= tr[1]:
        return true
    else:
        return false

'''
saved = file containing 4 numbers on each line to identify agent locations
'''
def read_agents(saved="detect_target.txt"):
    results = dict()
    counter = 0
    with open(saved, "r") as f:
        for line in f:
            results[counter] =line.split()
            counter+= 1
    return results

'''
miscellaneous tools ---------------------------------------------------------------------------------------------
'''

def get_directory():
    python_command = "readlink -f CloudManager.py"  # launch your python2 script using bash
    process = subprocess.Popen(python_command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    l = output.split("CloudManager.py\n");
    return l[0]

def get_random_cloud(num=50, scale=1000):
    temp = np.array(scale*np.random.random((num,3)))
    temp = np.float32(temp)
    return temp
    


