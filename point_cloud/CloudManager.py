import sys
sys.path.append('/usr/local/Cellar/')
import pcl
import math
import argparse
import sys
import os
from PIL import Image
from agent import Agent
from camera import Camera
import tools
#from AgentsManager import AgentsManager

class CloudManager:

    '''
    TODO
    - check movement
    - deal with agents problem
    - function to replace currently replaces a cube/rectangle space instead of cone, so removing incorrect things
    need to check if points are reachable by cone of view
    - fix the in_view thing 

    Note
    - may need to change the way the remove function works, if too many points in cloud and too slow.
    instead of converting all cloud to list, remove all coordinates of empty space in incoming image
    - default rounds to nearest int but can toggle to have decimals when testing
    '''
    def __init__(self, wr, hr, view, do_round=True):
        self.cloud = pcl.PointCloud()
        self.view = view
        self.wr = wr
        self.hr = hr
        self.do_round = do_round
        self.octree = pcl.OctreePointCloudSearch(0.1)
        self.camera = Camera(wr, hr, view, do_round=do_round)
        #self.agents = AgentsManager()

    '''
    Purpose: given a point cloud object, merge it into the saved big point cloud
    new_cloud = pcl.PointCloud()
    '''
    def concat_literal_cloud(self, new_cloud):
        new_pts = new_cloud.to_list()
        x, y, z = tools.find_min_max(new_pts)
        self.replace(new_cloud, x, y, z)

    '''
    Purpose: given a list of points, merge it into the saved big point cloud
    points_list = list of tuples of (x, y, z)
    '''
    def concat_points_list(self, points_list):
        all_pts = self.cloud.to_list()
        all_pts = all_pts + points_list
        self.cloud = pcl.PointCloud(all_pts)

    '''
    Purpose: given a list of points, replace the block of space defined by these points
    points_list = list of tuples of (x, y, z)
    '''
    def replace_points_list(self, points_list):
        x, y, z, = tools.find_min_max(points_list)
        self.replace(points_list, x, y, z)

    '''
    Purpose: return if a point is within view of the camera cone
    '''
    def in_view(self, pos, pt, view, left_view, vert, horz):
        #verify not out of range
        dist = tools.pts_dist(pt, pos)
        if dist > self.view:
            return False

        p, r, y = view
        lp, lr, ly = left_view

        vect = tools.pitch([0,0,1], p, do_round=self.do_round)
        vect = tools.roll(vect, r, do_round=self.do_round)
        vect = tools.yaw(vect, y, do_round=self.do_round)

        #parallel to top and bottom of view frame, runs from right to left
        horz = tools.pitch([-1,0,0], lp, do_round=self.do_round)
        horz = tools.roll(horz, lr, do_round=self.do_round)
        horz = tools.yaw(horz, ly, do_round=self.do_round)

        #parallel to left and right of view frame, runs from bottom to top
        vert = tools.ortho(vect, horz)

        #verify not too far left/right/high/low
        pt_vector = (pos[0] - pt[0], pos[1] - pt[1], pos[2] - pt[2])
        angle = tools.angle_2_vectors(pt_vector, vect)
        vert_proj = tools.projection_vector(pt_vector, vert)
        horz_proj = tools.projection_vector(pt_vector, horz)

        vert_angle = tools.angle_2_vectors(vert_proj, vect)
        horz_angle = tools.angle_2_vectors(horz_proj, vect)
        if vert_angle > self.hr/2 or horz_angle > self.wr/2:
            return False
        else:
            return True   

    '''
    Purpose: given a whole bunch of parameters (see camera.py), convert depth image to points and merge into saved big cloud
    view, leftview = rotations pitch roll yaw
    '''
    def concat_depth_map(self, pos, projection, view, left_view):
        points_list, vert, horz = self.camera.convert(pos, projection, view, left_view)
        x, y, z = tools.find_min_max(points_list)
        p, r, y = view
        lp, lr, ly = left_view

        my_pts = self.cloud.to_list()
        for point in my_pts:
            #if in space and keeping points, don't remove, else remove
            if self.in_view(pos, point, view, left_view, vert, horz) and point not in points_list:
                my_pts.remove(point)

        #add new points
        for point in points_list:
            if point not in my_pts:
                my_pts.append(point)

        self.cloud = pcl.PointCloud(list(my_pts))

    '''
    Purpose: returns num instances of coordinates of points in radius r of input coordinate pt_pos
    pt_pos = point tuple (x, y, z)
    r = radius float
    num = max number of points to fetch
    '''
    def radius_search(self, pt_pos, r, num):
        self.octree.set_input_cloud(self.cloud)
        self.octree.add_points_from_input_cloud()
        indices, rads = self.octree.radius_search(pt_pos, r, num)
        l = self.cloud.to_list()
        coords = []
        for i in indices:
            coords.append(l[i])
        return coords

    '''
    Purpose: returns T/F if my_pos and pt_pos are within r of each other
    my_pos = point tuple (x, y, z)
    pt_pos = point tuple (x, y, z)
    '''
    def is_in_radius(self, my_pos, pt_pos, r):
        dist = pow(my_pos[0]-pt_pos[0], 2) + pow(my_pos[1]-pt_pos[1], 2) + pow(my_pos[2]-pt_pos[2], 2)
        return dist <= r

    def cloud_size(self):
        return len(self.cloud.to_list())

    '''
    Purpose: replace piece of cloud using new info
    points_list = list of points that are in new snapshot of environment
    arr_thing = (min max) of that dimension
    '''
    def replace(self, points_list, arr_x, arr_y, arr_z):
        temp_removed = []
        #remove points that aren't there anymore
        my_pts = self.cloud.to_list()
        for point in my_pts:
            #if in space and keeping points, don't remove, else remove
            if (point not in points_list and 
                point[0] >= arr_x[0] and point[0] <= arr_x[1] and
                point[1] >= arr_y[0] and point[1] <= arr_y[1] and
                point[2] >= arr_z[0] and point[2] <= arr_z[1]):
                temp_removed.append(point)
                my_pts.remove(point)

        #add new points
        for point in points_list:
            if point not in my_pts:
                my_pts.append(point)

        self.cloud = pcl.PointCloud(list(my_pts))

    ''''
    Purpose: utility function to write cloud to file because matplotlib is incompatible with docker instance needed for pcl
    f = string file name to write to
    '''
    def save_cloud(self, f):
        with open(f, 'w+') as doc:
            for p in self.cloud.to_list():
                doc.write(str(p[0]) + ',' + str(p[1]) + ',' + str(p[2]) + '\n')

    '''
    Purpose: print out points in cloud with their index numbers, good for verifying radius search
    '''
    def print_cloud(self):
        l = self.cloud.to_list()

        for x in range(len(l)):
            print(str(x) + ' ' + str(l[x]))

    def return_cloud(self):
        return self.cloud.to_list()





