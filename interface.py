import pcl
import numpy as np
from math import *
try:
	from Queue import PriorityQueue # Python 2
except:
	from queue import PriorityQueue # Python 3

# https://nlesc.github.io/python-pcl/



class Map:

	def __init__(self):

		self.terrain = pcl.PointCloud_PointXYZ() # a point cloud of the terrain in the world frame
		self.octree = self.terrain.make_octree(0.1)
		self.belief = pcl.PointCloud_PointXYZI() # a point cloud with intensity = belief
		self.moving = set() # a set of moving Objects currently in the world
		self.intelligent = set() # a set of intelligent Objects currently in the world



class Object:

	def __init__(self):

		self.type = 0 # 0 for moving, 1 for intelligent
		self.classification = None # classification from RCNN

		self.time = 0 # the time at which the object was last observed

		# Initial Position, Velocity, and Acceleration of the Object
		self.pos = np.zeros((3,1))
		self.vel = np.zeros((3,1))
		self.accel = np.zeros((3,1))

		self.map = Map()
		self.path = []


	# The position of the object t seconds in the future
	def pos(self,t=0):

		if self.type == 0:
			return self.extrapolate(t)
		elif self.type == 1:
			return self.pathpos(t)


	# Predicts the position of the object given its initial pos, vel, accel
	def extrapolate(self,t):

		return self.pos + self.vel * t + self.accel * 0.5 * t**2


	def nextvel(self,vel):

		theta = atan2(vel[1],vel[0])
		speed = np.linalg.norm(self.vel)
		
		for dtheta in [-pi/6, -pi/12, 0, pi/12, pi/6]:
			newvel = np.array([cos(theta + dtheta), sin(theta + dtheta), 0,])
			yield newvel

		yield (np.array([0,0,0]), dtheta)


	def pathpos(self,t):
		# using splines on self.path (constant timestep tRes), interpolate for the position at time t
		pass
		

	def pathplan(self,T):

		destination = self.extrapolate(t)
		frontier = PriorityQueue()
		tRes = 0.1

		# Starting point
		pos = self.pos
		J = np.linalg.norm(np.array(pos)-np.array(destination))
		pathJ = 0
		t = 0
		initial = (J, pathJ, t, pos, vel)
		frontier.put(initial)

		while True:
			J, pathJ, t, pos, vel = frontier.get()
			for nextvel, dtheta in self.nextvel(vel):
				nextpathJ = pathJ + np.linalg.norm(nextvel) * tRes
				nextpos = pos + nextvel * tRes
				# TODO
