import sys
import pcl
import math
import argparse
import sys
import os
from PIL import Image


class Agent:

    #this agent is defined by these points
    def __init__(self, pts_list, vector = None):
        #last known location
        self.pts_list = pts_list
        #direction of travel
        self.vector = vector
        self.center, self.radius = self.sphere()

    #update info about this agent
    def update(self, pts_list, vector = None):
        self.pts_list = pts_list
        if vector is not None:
            self.vector = vector
        self.com = self.center_of_mass()

    def return_agent(self):
        return self.pts_list

    def sphere(self):
        min_x = min([tup[0] for tup in self.pts_list])
        max_x = max([tup[0] for tup in self.pts_list])
        min_y = min([tup[1] for tup in self.pts_list])
        max_y = max([tup[1] for tup in self.pts_list])
        min_z = min([tup[2] for tup in self.pts_list])
        max_z = max([tup[2] for tup in self.pts_list])
        
        c = ((min_x + max_x)/2, (min_y + max_y)/2, (min_z + max_z)/2)

        r = math.sqrt(pow(c[0]-min_x, 2), pow(c[1]-min_y, 2), pow(c[2]-min_z, 2))

        return c, r