try:
	import pcl
except:
	print('Could not import PCL')
import numpy as np
from math import *
from itertools import count
try:
	from Queue import PriorityQueue # Python 2
except:
	from queue import PriorityQueue # Python 3
from GaussND import *

# https://nlesc.github.io/python-pcl/



class Map:

	def __init__(self):

		# self.terrain = pcl.PointCloud(np.array([[1, 2, 3], [3, 4, 5]], dtype=np.float32)) # a point cloud of the terrain in the world frame
		# self.octree = pcl.OctreePointCloud(0.1)
		# self.belief = pcl.PointCloud_PointXYZI(np.array([[1,1,1,0],[1,0,0,1]], dtype=np.float32)) # a point cloud with intensity = belief
		self.objects = set() # a set of moving Objects currently in the world


	def proxCost(self, pos, t=0, ignore=None):
		cost = 0
		for obj in self.objects:
			if ignore is not None and obj != ignore:
				displacement = pos - obj.pos(t)
				cost += obj.proxCost(displacement)
		return cost


	def plotObects(self, t=0):
		if not isinstance(t, np.ndarray):
			t = np.array([t])
		t = np.reshape(t,(-1,))
		for obj in self.objects:
			color = np.random.rand(3)
			for i in range(t.size):
				pos = obj.pos(t[i])
				plt.plot(pos[0], pos[1], 'o', c=color, alpha=(i+1)/t.size)



class Object:

	def __init__(self, map=None):

		self.m = map

		# Initial Position, Velocity, and Acceleration of the Object
		self.position = np.zeros((3,1))
		self.velocity = np.zeros((3,1))
		self.acceleration = np.zeros((3,1))

		# Other Attributes
		self.type = 0 # 0 for moving, 1 for intelligent
		self.classification = None # classification from RCNN
		self.gaussian = GaussND(numN=(np.array([[0,0,0]]).T,0.1*np.identity(3)))

		# Internal variables
		self.path = None

		# Options
		self.tRes = 0.05


	# The cost associated with being withing a given proximity of the object
	def proxCost(self,pos):
		return self.gaussian[np.reshape(pos,(-1,3))]


	# The position of the object t seconds in the future
	def pos(self,t=0):
		if self.type == 0:
			return self.extrapolate(t)
		elif self.type == 1:
			return self.pathpos(t)


	# Predicts the position of the object given its initial pos, vel, accel
	def extrapolate(self,t):
		return self.position + self.velocity * t + self.acceleration * 0.5 * t**2


	def nextvel(self,vel):
		theta = atan2(vel[1],vel[0])
		speed = np.linalg.norm(self.velocity)
		dangle = pi/12
		for dpsi in [-dangle, 0, dangle]:
			for dtheta in [-dangle, 0, dangle]:
				oldvel = np.array([[cos(theta), sin(theta), 0.]]).T
				rothoriz = np.array([[cos(dtheta), -sin(dtheta), 0], [sin(dtheta), cos(dtheta), 0], [0, 0, 1]])
				rotvert = np.array([[cos(dpsi), 0, sin(dpsi)], [0, 1, 0], [-sin(dpsi), 0, cos(dpsi)]])
				newvel = rothoriz @ rotvert @ oldvel
				yield newvel


	def pathpos(self,t):
		ilow = floor(t/self.tRes)
		ihigh = ceil(t/self.tRes)
		if ilow < 0:
			return self.path[0,:]
		if ihigh >= self.path.shape[0]:
			return self.path[-1,:]
		u = (t - self.tRes * ilow) / self.tRes
		return (u * self.path[ilow,:]) + ((1-u) * self.path[ihigh,:])


	def distance(self,pos1,pos2):
		return np.linalg.norm(np.array(pos1)-np.array(pos2))
		

	def pathplan(self,T=10,destination=None):

		# Initialization
		if destination is None:
			destination = self.extrapolate(T)
		frontier = PriorityQueue()
		epsilon = 0.1
		maxsteps = int(T / self.tRes)

		# Starting point
		ties = count()
		pos = self.position
		heuristic = self.distance(pos,destination)
		data = {'pos': self.position, 'vel': self.velocity, 't': 0, 'dist': 0}
		J = heuristic
		initial = (J, next(ties), data)
		frontier.put(initial)

		# Path planning
		path = np.zeros((maxsteps,self.position.size))
		for i in range(maxsteps):
			J, _, data = frontier.get()
			idx = int(data['t'] / self.tRes)
			path[i,:] = np.reshape(data['pos'],(3,))
			# import pdb; pdb.set_trace()
			if self.distance(data['pos'],destination) < epsilon:
				break
			for nextvel in self.nextvel(data['vel']):
				nextt = data['t'] + self.tRes
				nextpos = data['pos'] + nextvel * self.tRes
				nextdist = data['dist'] + np.linalg.norm(nextvel) * self.tRes
				nextJ = self.distance(nextpos,destination) + self.m.proxCost(pos=nextpos,t=nextt,ignore=self)
				nextdata = {'pos': nextpos, 'vel': nextvel, 't': nextt, 'dist': nextdist}
				frontier.put((nextJ, next(ties), nextdata))
		path = path[:i,:]
		self.path = path
		return path


if __name__ == '__main__':
	from matplotlib import pyplot as plt

	m = Map()

	# o1 = Object(m)
	# r1 = 0.1
	# o1.position = np.array([[0,1,0]]).T
	# o1.velocity = np.array([[0,0,0]]).T
	# mu1 = np.array([[0,0,0]]).T
	# cov1 = r1*np.identity(3)
	# g1 = 1*GaussND(numN=(mu1,cov1))
	# o1.gaussian = g1
	# m.objects.add(o1)

	o1 = Object(m)
	o1.position = np.array([[0,1,0]]).T
	m.objects.add(o1)

	o2 = Object(m)
	o2.position = np.array([[-1,3,0]]).T
	m.objects.add(o2)

	o3 = Object(m)
	o3.position = np.array([[-1.5,2.6,0]]).T
	m.objects.add(o3)

	o = Object(m)
	o.type = 1
	o.velocity = np.array([[1,0,0]]).T
	m.objects.add(o)

	dest = np.array([[-2,5,0]]).T
	path = o.pathplan(10,dest)

	T = (path.shape[0]-1) * o.tRes
	t = np.arange(0,T,o.tRes)
	m.plotObects(t)
	plt.plot(path[:,0],path[:,1])
	plt.show()
