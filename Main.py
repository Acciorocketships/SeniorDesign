from threading import Thread
import numpy as np
from Map import *
from Planning import *


class Mainloop:


	def __init__(self):

		self.map = Map()

		self.target = {'pos':np.zeros((3,)), 
					   'vel':np.zeros((3,)),
					   'speed':1}

		self.state = {'pos':np.zeros((3,)), 
					  'vel':np.array((3,))}

		self.path = {'x': np.zeros((0,3)),
					 'v': np.zeros((0,3)),
					 't': np.zeros((0,)),
					 'roughx': np.zeros((0,3)),
					 'rought': np.zeros((0,3))}

		self.planner = Planner(map=self.map,target=self.target,state=self.state,path=self.path)



	def updateMap(self,lidar):

		pass

		## Emily's logic here ##

		## updates the self.map member variable, and all the objects inside it ##
		## updates the pos, vel, rot states ##
		## calls updatePath ##



	def updatePath(self):

		self.planner.plan(dt_out=0.01, dt_astar=0.1)




if __name__ == '__main__':
	main = Mainloop()
	main.updatePath()