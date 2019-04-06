from threading import Thread
import numpy as np


class Mainloop:


	def __init__(self):

		self.map = Map()

		self.targetpos = np.array([0,0,0])
		self.targetvel = np.array([0,0,0])

		self.pos = np.array([0,0,0])
		self.vel = np.array([0,0,0])
		self.rot = np.array([0,0,0])

		self.path = []



	def update(self,lidar):

		pass

		## Emily's logic here ##

		## updates the self.map member variable, and all the objects inside it ##
		## updates the pos, vel, rot states ##
		## calls updatePath ##



	def updatePath(self):

		pass

		## Ryan's logic here ##

		## updates the self.path member variable, a list of positions from the current position to the goal ##


