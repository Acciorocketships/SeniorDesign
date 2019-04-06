import sys
sys.path.append('/usr/local/Cellar/')
import pcl
import math
import argparse
import sys
import os
from PIL import Image
import tools
import numpy as np
from agent import Agent

class CloudManager:

    '''
    Note
    - may need to change the way the remove function works, if too many points in cloud and too slow.
    instead of converting all cloud to list, remove all coordinates of empty space in incoming image
    - default rounds to nearest int but can toggle to have decimals when testing
    '''
    def __init__(self, wr, hr, view, do_round=True):
        self.cloud = pcl.PointCloud()
        self.agents = []
        self.view = view
        self.wr = wr #max width view in radians
        self.hr = hr #max height view in radians
        self.do_round = do_round #if true, round coordinates to nearest int
        self.octree = pcl.OctreePointCloudSearch(0.1)
        self.mi = None
        self.mj = None
        self.hasmatrix = False
        self.voxel_size = 5000 #10 meters

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
    Purpose: returns T/F if my_pos and pt_pos are within r of each other
    my_pos = point tuple (x, y, z)
    pt_pos = point tuple (x, y, z)
    '''
    def is_in_radius(self, my_pos, pt_pos, r):
        dist = pow(my_pos[0]-pt_pos[0], 2) + pow(my_pos[1]-pt_pos[1], 2) + pow(my_pos[2]-pt_pos[2], 2)
        return dist <= r

    def cloud_size(self):
        return len(self.cloud.to_list())

    ''''
    Purpose: utility function to write cloud to file because matplotlib is incompatible with docker instance needed for pcl
    f = string file name to write to
    '''
    def save_cloud(self, f):
        pcl.save(self.cloud, f)

    def save_points(self, f, arr):
        with open(f, 'w+') as doc:
            for p in arr:
                doc.write(str(p[0]) + ',' + str(p[1]) + ',' + str(p[2]) + '\n')

    def save_cloud_points(self, f):
        self.save_points(f,self.cloud.to_list())

    def load_pcd_cloud(self, f):
        self.cloud = pcl.load(f)

    '''
    Purpose: print out points in cloud with their index numbers, good for verifying radius search
    '''
    def print_cloud(self):
        l = self.cloud.to_list()

        for x in range(len(l)):
            print(str(x) + ' ' + str(l[x]))

    def return_cloud(self):
        return self.cloud.to_list()

        '''
    same params concat_depth_map, directly returns list instead for debugging
    '''
    def process_depth_map(self, pos, projection, view, left_view):
        points_list = self.convert(pos, projection, view, left_view)
        return points_list

    def make_radian_matrix(self, shape):
        i_mid = shape[0]/2
        j_mid = shape[1]/2
        self.mi = np.zeros(shape)
        self.mj = np.zeros(shape)
        self.hasmatrix = True

        for j in range(shape[1]):
            for i in range(shape[0]):
                #get radians difference from center
                self.mi[i][j] = -(i + 0.5 - i_mid)/shape[0] * self.wr 
                self.mj[i][j] = (j + 0.5 - j_mid)/shape[1] * self.hr 

    '''
    Purpose: return if a point is within view of the camera cone
    pos = location of self
    pt = location of point being considered
    vect, vert, horiz = vectors representing parallel to view and orientation of screen along that view
    '''
    def in_view(self, pos, pt, vect, vert, horiz):
        #verify not out of range
        dist = tools.pts_dist(pt, pos)
        if dist > self.view:
            return False

        #vector from camera to point
        pt_vector = (pt[0] - pos[0], pt[1] - pos[1], pt[2] - pos[2])

        #angle between view and vector
        angle = tools.angle_2_vectors(pt_vector, vect)

        #projection vectors in the vert and horiz directions
        dist_proj = tools.projection_vector(pt_vector, vect)
        vert_proj = tools.projection_vector(pt_vector, vert)
        nvp = (vert_proj[0] + dist_proj[0], vert_proj[1] + dist_proj[1], vert_proj[2] + dist_proj[2]) 
        horiz_proj = tools.projection_vector(pt_vector, horiz)
        nhp = (horiz_proj[0] + dist_proj[0], horiz_proj[1] + dist_proj[1], horiz_proj[2] + dist_proj[2])

        #angle of projections to view
        vert_angle = tools.angle_2_vectors(nvp, vect)
        horiz_angle = tools.angle_2_vectors(nhp, vect)

        if vert_angle > self.hr/2 or horiz_angle > self.wr/2:
            return False
        else:
            return True  

    '''
    Purpose: given a whole bunch of parameters (see camera.py), convert depth image to points and merge into saved big cloud
    pos = your own location
    projection = 2D matrix representing input
    view, leftview = rotations pitch roll yaw
    '''
    def append_depth_map(self, pos, projection, view, left_view):
        points_list = self.convert(pos, projection, view, left_view)
        self.cloud = pcl.PointCloud(self.cloud.to_list() + points_list)

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
    DONE ABOVE
    ----------------------------------------------------------------------------------------------------
    WIP BELOW
    '''

    '''
    Given depth image, convert to points in 3D space
    pos = camera x y z in objective 3D space
    projection = 2D numpy matrix of depth values
    ags = list of agent bounding boxes
    '''
    def convert(self, pos, projection, view, left_view):
        #direction of camera's view
        vect = tools.pry_to_view(view, self.do_round)
        #parallel to top and bottom of view frame, runs from right to left
        horiz = tools.pry_to_horiz(left_view, self.do_round)
        #parallel to left and right of view frame, runs from bottom to top
        vert = tools.ortho(vect, horiz)

        new_pts = set()

        if not self.hasmatrix:
            self.make_radian_matrix(projection.shape)

        temp_matrix = np.zeros(projection.shape).tolist()

        adjustment_high = 0.99 #max 1
        adjustment_low = 0 #min 0

        #j = all the rows, i = all the columns
        for j_counter in range(projection.shape[1]):
            for i_counter in range(projection.shape[0]):
                try:
                    pix_dist = self.view - float(projection[i_counter][j_counter]) / 255 * self.view
                    if pix_dist <= adjustment_high * self.view and pix_dist >= adjustment_low * self.view:
                        #get rotation
                        m_i = tools.rmatrix_to_vector(tools.invert(tools.make_unit_vector(vert)), self.mj[i_counter][j_counter])
                        m_j = tools.rmatrix_to_vector(tools.invert(tools.make_unit_vector(horiz)), self.mi[i_counter][j_counter])

                        max_vect = tools.normalize(vect, tools.len_vector(vect))
                        max_vect = tools.vector_multiplier(max_vect, self.view)

                        #get vector from camera to pixel, with vector in objective 3D space
                        pv = [tools.dot_product(vect, m_i[0]), tools.dot_product(vect, m_i[1]), tools.dot_product(vect, m_i[2])]
                        pv = [tools.dot_product(pv, m_j[0]), tools.dot_product(pv, m_j[1]), tools.dot_product(pv, m_j[2])]
                        pv = tools.make_unit_vector(pv)
                        pix_vect = tools.vector_multiplier(pv, pix_dist)

                        #get position of this point in 3D space
                        pix_pt = tools.sum_vectors(pos, pix_vect)
                        if self.do_round:
                            new_pts.add(tuple(tools.make_ints(pix_pt)))
                            temp_matrix[i][j] = tuple(tools.make_ints(pix_pt))
                        else:
                            new_pts.add(tuple(pix_pt))
                            temp_matrix[i][j] = tuple(pix_pt)
                except:
                    pass

            '''
            TODO make this actually useful
            
            agents = tools.read_agents()
            for a in agents:
                bounds = [int(x) for x in agents[a]]
                items = []
                for n in range(max(0, bounds[0]), min(projection.shape[0], bounds[2])):
                    for m in range(max(0, bounds[1]), min(projection.shape[1], bounds[3])):
                        items.append(temp_matrix[n][m])
                ag = Agent(items)
                self.add_agent(ag)
            '''

        return list(new_pts)

     

    '''
    Purpose: given a whole bunch of parameters (see camera.py), convert depth image to points and replace space into saved big cloud
    pos = your own location
    projection = 2D matrix representing input
    view, leftview = rotations pitch roll yaw
    '''
    def concat_depth_map(self, pos, projection, view, left_view):
        points_list = self.convert(pos, projection, view, left_view)
        
        result = []
        my_pts = self.cloud.to_list()
        vect = tools.pry_to_view(view, self.do_round)
        horiz = tools.pry_to_horiz(left_view, self.do_round)
        vert = tools.ortho(vect, horiz)

        for point in my_pts:
            if not self.in_view(pos, point, vect, vert, horiz):
                result.append(point)
        result += points_list

        self.cloud = pcl.PointCloud(result)


    '''
    Purpose: replace cubic piece of cloud using new info's given boundary space
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

    def get_min_max_3D(self):
        x, y, z = tools.find_min_max(self.cloud.to_list())

    '''
    new_cloud = input from depth map
    cropped_cloud = current points occupying the same space
    purpose: if there are few enough current points that we can use append instead of concat, speed up run time
    '''
    def concat_vs_append(self, new_cloud, cropped_cloud):
        do_concat = True
        '''
        TODO determine similarity
        https://ieeexplore.ieee.org/document/6238913
        '''
        if do_concat:
            return "concat"
        else:
            return "append"

    '''
    pos = current coordinate position
    purpose: if map includes distant space, save it and delete it to speed up processing
    '''
    def save_voxel(self, pos):
        x, y, z = self.get_min_max_3D()

        begin_x = tools.round_down_multiple_of(x[0], self.voxel_size)
        end_x = tools.round_up_multiple_of(x[1], self.voxel_size)
        begin_y = tools.round_down_multiple_of(y[0], self.voxel_size)
        end_y = tools.round_up_multiple_of(y[1], self.voxel_size)

        safe_begin_x = tools.round_down_multiple_of(pos[0] - 3*self.voxel_size, self.voxel_size)
        safe_end_x = tools.round_up_multiple_of(pos[0] + 3*self.voxel_size, self.voxel_size)
        safe_begin_y = tools.round_down_multiple_of(pos[1] - 3*self.voxel_size, self.voxel_size)
        safe_end_y = tools.round_up_multiple_of(pos[1] + 3*self.voxel_size, self.voxel_size)

        for i in range((end_x - begin_x)/self.voxel_size):
            for j in range((end_y - begin_y)/self.voxel_size):
                temp_x = begin_x + i * self.voxel_size
                temp_y = begin_y + j * self.voxel_size

                mid_x = temp_x - 0.5 * self.voxel_size
                mid_y = temp_y - 0.5 * self.voxel_size

                if not (mid_x > safe_begin_x and mid_x < safe_end_x and mid_y > safe_begin_y and mid_y < safe_end_y):
                    cbf = self.cloud.make_cropbox()
                    cbf.set_Min(temp_x, temp_y, z[0], 1.0)
                    cbf.set_Max(temp_x + self.voxel_size, temp_y + self.voxel_size, z[1], 1.0)
                    cloud_out = cbf.filter()
                    pcl.save(cloud_out, temp_x + "-" + temp_y + ".pcd")
        
        cbf = self.cloud.make_cropbox()
        cbf.set_Min(safe_begin_x, safe_begin_y, z[0], 1.0)
        cbf.set_Max(safe_end_x, safe_end_y, z[1], 1.0)
        cloud_out = cbf.filter()
        self.cloud = cloud_out

        #TODO delete irrelevant agents

    '''
    pos = current coordinate position
    purpose: if nearby map includes unknown space, check if saved map exists and load it
    '''
    def load_voxel(self, pos):
        x, y, z = self.get_min_max_3D()

        begin_x = tools.round_down_multiple_of(x[0], self.voxel_size)
        end_x = tools.round_up_multiple_of(x[1], self.voxel_size)
        begin_y = tools.round_down_multiple_of(y[0], self.voxel_size)
        end_y = tools.round_up_multiple_of(y[1], self.voxel_size)

        safe_begin_x = tools.round_down_multiple_of(pos[0] - 2*self.voxel_size, self.voxel_size)
        safe_end_x = tools.round_up_multiple_of(pos[0] + 2*self.voxel_size, self.voxel_size)
        safe_begin_y = tools.round_down_multiple_of(pos[1] - 2*self.voxel_size, self.voxel_size)
        safe_end_y = tools.round_up_multiple_of(pos[1] + 2*self.voxel_size, self.voxel_size)

        for i in range((end_x - begin_x)/self.voxel_size):
            for j in range((end_y - begin_y)/self.voxel_size):
                temp_x = begin_x + i * self.voxel_size
                temp_y = begin_y + j * self.voxel_size
                try:
                    name = temp_x + "-" + temp_y + ".pcd"
                    f = open(name)
                    temp_c = pcl.load(name)
                    self.cloud = self.cloud + temp_c
                except:
                    pass


   