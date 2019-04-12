import sys
import pcl
import math
import tools
from CloudManager import CloudManager
import sys
import cv2
import numpy as np
import os
from PIL import Image
from scipy.misc import imread

cloud = CloudManager(math.pi/3, math.pi/4, 20, do_round=False)

def test_in_view():
    vect = [0, 0, 1]
    horz = [-1, 0, 0]
    test = cloud.in_view((0, 0, 0), (0, 0, 9.999), vect, horz, tools.ortho(vect, horz))
    print(test)

def test_out_of_view():
    vect = [0, 0, 1]
    horz = [-1, 0, 0]
    test = cloud.in_view((0, 0, 0), (0, 0, 10.001), vect, horz, tools.ortho(vect, horz))
    print(test)


def test_depth_map():

    t = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            t[i][j] = i + j

    t = np.array(t)
    print(t)
    t=tools.normalize_matrix(t, 255, 55)
    cloud.concat_depth_map((1, 7, -5), t, (math.pi, math.pi/2, 0), (math.pi/2, 0, 0))

    t = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            t[i][j] = abs(i - j)

    t = np.array(t)
    print(t)
    t=tools.normalize_matrix(t, 255, 55)
    cloud.concat_depth_map((1, 1, 1), t, (math.pi, 0, 0), (0, math.pi, 0))

    t = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            if i > j:
                t[i][j] = 50
            else:
                t[i][j] = 200

    t = np.array(t)
    print(t)
    t=tools.normalize_matrix(t, 255, 55)
    cloud.concat_depth_map((3, 3, 5), t, (math.pi, math.pi, math.pi/2), (0, math.pi, 0))
    cloud.concat_points_list([(0, 0, 0)])
    cloud.save_cloud_points("t.txt")

def test_append():
    t = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            t[i][j] = i + j

    t = np.array(t)
    print(t)
    t=tools.normalize_matrix(t, 255, 55)
    cloud.append_depth_map((1, 7, -5), t, (math.pi, math.pi/2, 0), (math.pi/2, 0, 0))

    t = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            t[i][j] = abs(i - j)

    t = np.array(t)
    print(t)
    t=tools.normalize_matrix(t, 255, 55)
    cloud.append_depth_map((1, 1, 1), t, (math.pi, 0, 0), (0, math.pi, 0))

    t = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            if i > j:
                t[i][j] = 50
            else:
                t[i][j] = 200

    t = np.array(t)
    print(t)
    t=tools.normalize_matrix(t, 255, 55)
    cloud.append_depth_map((3, 3, 5), t, (math.pi, math.pi, math.pi/2), (0, math.pi, 0))
    cloud.concat_points_list([(0, 0, 0)])
    cloud.save_cloud_points("t.txt")

def test_cube():
    t = [(x, y, z) for x in range(-10, 10) for y in range(-10, 10) for z in range(-10, 10)]
    s = [(x, y, z) for x in range(-2, 2) for y in range(-2, 2) for z in range(0, 50)]
    cloud.cloud = pcl.PointCloud(t+s)
    arr = np.random.rand(100,100) * 255
    cloud.concat_depth_map((0, 0, 0), arr, (math.pi, 0, 0), (0, math.pi, 0))
    arr = np.random.rand(100,100) * 255
    cloud.concat_depth_map((-3, 2, 3), arr, (math.pi/4, 0, 0), (0, math.pi/2, 0))
    cloud.save_cloud_points("t.txt")

def test_radius_search():
    t = [(x, y, z) for x in range(-10, 10) for y in range(-10, 10) for z in range(-10, 10)]
    s = [(x, y, z) for x in range(-2, 2) for y in range(-2, 2) for z in range(0, 50)]
    cloud.cloud = pcl.PointCloud(t+s)
    coords = cloud.radius_search((0,0,0), 5, 5000)
    cloud.save_points("t.txt", coords)

def test_depth_map3():
    m = np.zeros((100, 100))
    cloud.concat_points_list([(0, 0, 0)])
    m = m + 50
    arr = cloud.process_depth_map((0, 0, 0), m, (math.pi, math.pi, 0), (math.pi, 0, 0))
    m = m + 10
    arr += cloud.process_depth_map((0, 0, 0), m, (math.pi, math.pi, 0), (math.pi, 0, 0))
    m = m + 150
    arr += cloud.process_depth_map((0, 0, 0), m, (math.pi, math.pi, 0), (math.pi, 0, 0))
    m = m + 10
    arr += cloud.process_depth_map((0, 0, 0), m, (math.pi, math.pi, 0), (math.pi, 0, 0))
    
    cloud.save_points("t.txt", arr)

def test_depth_map_merge():
    im = imread("res/rabbit.png")
    if len(im.shape) > 2:
        im = tools.make_greyscale(im)
    arr = cloud.process_depth_map((0, 0, 0), im, (math.pi, 0, 0), (math.pi, 0, 0))
    im = imread("res/lines.png")
    if len(im.shape) > 2:
        im = tools.make_greyscale(im)
    arr2 = cloud.process_depth_map((2, -4, 1), im, (math.pi, 0, 0), (0, math.pi, 0))

    cloud.save_points("t.txt", arr+arr2)



def test_crop_box():
    im = imread("res/rabbit.png")
    if len(im.shape) > 2:
        im = tools.make_greyscale(im)
    arr = cloud.process_depth_map((0, 0, 0), im, (math.pi, math.pi, 0), (math.pi, 0, 0))
    cloud.cloud = pcl.PointCloud(arr)
    cbf = cloud.cloud.make_cropbox()
    cbf.set_Min(-1, -1, -1, 1.0)
    cbf.set_Max(1, 1, 1, 1.0)
    cloud_out = cbf.filter()
    cloud.cloud = cloud_out
    cloud.save_points("t.txt", cloud.cloud.to_list())

def test_conditional():
    im = imread("res/rabbit.png")
    if len(im.shape) > 2:
        im = tools.make_greyscale(im)
    arr = cloud.process_depth_map((0, 0, 0), im, (math.pi, math.pi, 0), (math.pi, 0, 0))
    cloud.cloud = pcl.PointCloud(arr)


def test_extract():
    im = imread("res/rabbit.png")
    if len(im.shape) > 2:
        im = tools.make_greyscale(im)
    arr = cloud.process_depth_map((0, 0, 0), im, (math.pi, math.pi, 0), (math.pi, 0, 0))
    cloud.cloud = pcl.PointCloud(arr)
    

def test_read_agents():
    cloud.read_agents()

def test_write_cloud():
    cloud.save_cloud("test.txt")

def test_write_cloud2():
    a = np.random.randn(100, 3).astype(np.float32)
    p1 = pcl.PointCloud(a)
    pcl.save(p1, "temppcl.pcd")

def test_read_pcd():
    with open("temppcl.pcd") as f:
        for line in f:
            print(line)

def test_distance():
    m = np.zeros((50, 50))
    m += 200
    arr = cloud.process_depth_map((0, 0, 0), m, (math.pi, math.pi, 0), (math.pi, 0, 0))
    cloud.save_points("t.txt", arr)

def test_manual():
    m = np.zeros((50, 50))
    for x in range(50):
        for y in range(50):
            m[x][y] = abs(0.5*x + 0.5*y)
    arr = cloud.process_depth_map((0, 0, 0), m, (math.pi, math.pi, 0), (math.pi, 0, 0))
    cloud.save_points("t.txt", arr)

def find_function():
    cloud.get_min_max_3D()

def test_reduce_map():
    options = [5000, -5000, 10000, -10000, 15000, -15000, 20000, -20000, 25000, -25000, 30000, -30000]
    pts = [(x, y, z) for x in options for y in options for z in options]
    pts.append((0, 0, 0))
    cloud.cloud = pcl.PointCloud(pts)
    cloud.print_cloud()
    cloud.save_voxel((0, 0, 0))
    cloud.print_cloud()

def test_load_map():
    cloud.cloud = pcl.PointCloud([(0, 0, 0)])
    cloud.load_voxel((10000, 10000, 0))
    cloud.print_cloud()

def test_load_one():
    cloud.cloud = pcl.load("x15000-y25000.pcd")
    cloud.print_cloud()

def test_path():
    tools.get_directory()

def test_random():
    tools.get_random_cloud()

def test_sift():
    img1 = cv2.imread('res/food.jpg',0)
    img2 = cv2.imread('res/food_crop.jpg',0)
    tools.flann_sift(img1, img2)

def test_depth_map2():
    m = []
    for x in range(-20, 20):
        for y in range(-20, 20):
            m.append((x, y, 0))
    cloud.cloud = pcl.PointCloud(m)
    im = imread("res/scene.jpg")
    if len(im.shape) > 2:
        im = tools.make_greyscale(im)
    arr = cloud.process_depth_map((0, 0, -10), im, (0, 0, 0), (0, 0, 0))
    cloud.save_points("t.txt", arr + m)
    #cloud.concat_depth_map((0, 0, 0), im, (math.pi, math.pi, 0), (math.pi, 0, 0))
    #cloud.save_cloud("t.txt")

def conv_rel():
    im = imread("res/rabbit.png")
    if len(im.shape) > 2:
        im = tools.make_greyscale(im)
    arr = cloud.convert_relatively(im)
    arr = np.array(arr)
    arr.flatten()
    print(arr.shape)
    #print(arr)
    arr = [arr[x][y] for x in range(len(arr)) for y in range(len(arr[0]))]
    cloud.save_points("t.txt", arr)

def test_icp():
    ang = np.linspace(-np.pi/2, np.pi/2, 320)
    a = np.array([ang, np.sin(ang)])
    print(a.shape)


    th = np.pi/2
    rot = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
    b = np.dot(rot, a) + np.array([[0.2], [0.3]])

    M2 = tools.icp(a, b, [0.1,  0.33, np.pi/2.2], 30)

    print("MATRIX")
    print(M2)

    src = np.array([a.T]).astype(np.float32)
    res = cv2.transform(src, M2)
    print(src.shape)
    print(res.shape)
    #print(b)

    new_b = list(zip(b[0], b[1]))
    new_b = [(float(tup[0]), float(tup[1]), 0) for tup in new_b]
    #print(new_b)

    #new_r = list(zip(res[0].T[0], res[0].T[1]), 'r.')
    new_r = [(tup[0], tup[1], 0) for tup in res[0]]
    #print(new_r)

    new_a = list(zip(a[0], a[1]))
    new_a = [(tup[0], tup[1], 0) for tup in new_a]
    #print(new_a)

    cloud.save_points("t.txt", new_b + new_r + new_a)

#test_depth_map()
#test_append()
#test_cube()
#test_depth_map_merge()
#test_in_view()
#test_out_of_view()
#test_depth_map2()
#test_depth_map3()
#test_read_agents()
#test_write_cloud2()
#test_crop_box()
#test_radius_search()
#test_read_pcd()
#test_distance()
#test_manual()
#find_function()
#test_reduce_map()
#test_load_map()
#test_path()
#test_load_one()
#test_random()
#test_sift()
#conv_rel()
test_icp()




print("test end")




