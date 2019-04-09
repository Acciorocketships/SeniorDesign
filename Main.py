from __future__ import absolute_import
from threading import Thread
import numpy as np
from Map import *
from Planning import *


class Mainloop(object):


	def __init__(self):

		self.map = Map()

		self.target = {u'pos':np.zeros((3,)), 
					   u'vel':np.zeros((3,)),
					   u'speed':1}

		self.state = {u'pos':np.zeros((3,)), 
					  u'vel':np.array((3,))}

		self.path = {u'x': np.zeros((0,3)),
					 u'v': np.zeros((0,3)),
					 u't': np.zeros((0,)),
					 u'roughx': np.zeros((0,3)),
					 u'rought': np.zeros((0,3))}

		self.planner = Planner(map=self.map,target=self.target,state=self.state,path=self.path)



	def updateMap(self,lidar):

		pass

		## Emily's logic here ##

		## updates the self.map member variable, and all the objects inside it ##
		## updates the pos, vel, rot states ##
		## calls updatePath ##



	def updatePath(self):

		self.planner.plan(dt_out=0.01, dt_astar=0.1)




if __name__ == u'__main__':
	main = Mainloop()
	main.updatePath()