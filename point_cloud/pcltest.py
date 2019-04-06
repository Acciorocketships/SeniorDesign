import sys
import pcl
import math
import tools
from CloudManager import CloudManager
import sys
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


def test_depth_map2():
    m = []
    for x in range(-20, 20):
        for y in range(-20, 20):
            m.append((x, y, 0))
    cloud.cloud = pcl.PointCloud(m)
    im = imread("res/rabbit.png")
    if len(im.shape) > 2:
        im = tools.make_greyscale(im)
    arr = cloud.process_depth_map((0, 0, -10), im, (0, 0, 0), (0, 0, 0))
    cloud.save_points("t.txt", arr + m)
    #cloud.concat_depth_map((0, 0, 0), im, (math.pi, math.pi, 0), (math.pi, 0, 0))
    #cloud.save_cloud("t.txt")

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

#test_depth_map()
#test_append()
#test_cube()
#test_depth_map_merge()
#test_in_view()
#test_out_of_view()
test_depth_map2()
#test_depth_map3()
#test_read_agents()
#test_write_cloud2()
#test_crop_box()
#test_radius_search()
#test_read_pcd()
#test_distance()
#test_manual()
#find_function()












print("test end")