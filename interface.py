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
from matplotlib.widgets import Slider

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
			if obj != ignore:
				# import pdb; pdb.set_trace()
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

	def plot(self,T=3,lim=[-5,5,-5,5]):
		self.fig, ax = plt.subplots()
		plt.xlim(lim[0],lim[1])
		plt.ylim(lim[2],lim[3])
		plt.subplots_adjust(bottom=0.25)
		colors = [np.random.rand(3) for i in range(len(self.objects))]
		self.axes = [None for i in range(len(self.objects))]
		for i, obj in enumerate(self.objects):
			pos = obj.pos(0)
			self.axes[i] = plt.plot(pos[0], pos[1], 'o', c=colors[i])[0]
		axt = plt.axes([0.125, 0.1, 0.775, 0.03])
		tslider = Slider(axt, 'Time', 0, T, valinit=0)
		tslider.on_changed(self.update)
		plt.show()

	def update(self,t):
		for i, obj in enumerate(self.objects):
			pos = obj.pos(t)
			self.axes[i].set_ydata(pos[1])
			self.axes[i].set_xdata(pos[0])
		self.fig.canvas.draw_idle()



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
		self.gaussian = 100*GaussND(numN=(np.array([[0,0,0]]).T,3*np.identity(3)))

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


	def pathpos(self,t):
		ilow = floor(t/self.tRes)
		ihigh = ceil(t/self.tRes)
		if ilow < 0:
			return self.path[0,:]
		if ihigh >= self.path.shape[0]:
			return self.path[-1,:]
		u = (t - self.tRes * ilow) / self.tRes
		return ((1-u) * self.path[ilow,:]) + (u * self.path[ihigh,:])


	def distance(self,pos1,pos2):
		return np.linalg.norm(np.array(pos1)-np.array(pos2))


	def heuristic(self,pos,t,destination):
		return self.distance(pos,destination) + self.m.proxCost(pos=pos,t=t,ignore=self)


	def discretize(self,val,step):
		for i in range(len(val)):
			val[i,0] = round(val[i,0] / step) * step
		return val


	def nextpos(self,pos,step):
		for dx in [-step, 0, step]:
			for dy in [-step, 0, step]:
				for dz in [-step, 0, step]:
					if dx==0 and dy==0 and dz==0:
						continue
					yield pos + np.array([[dx],[dy],[dz]])


	def pathplan(self,T=3,destination=None):

		# Initialization
		if destination is None:
			destination = self.extrapolate(T)
		frontier = PriorityQueue()
		maxsteps = 3*int(T / self.tRes)
		visited = {}
		self.speed = np.linalg.norm(self.velocity)
		step = round(0.1 * self.speed, 2)

		# Starting point
		ties = count()
		pos = self.discretize(self.position,step)
		heuristic = self.heuristic(pos,0,destination)
		data = {'pos': pos, 't': 0, 'dist': 0, 'prev': None}
		prev = {totuple(pos): None}
		J = heuristic
		initial = (J, next(ties), data)
		frontier.put(initial)

		# Path planning
		for i in range(maxsteps):
			# Pop Next Value
			J, _, data = frontier.get()
			idx = int(data['t'] / self.tRes)
			pos = data['pos']
			# Check if Visited
			if visited.get(totuple(pos),float('inf')) > J:
				visited[totuple(pos)] = J
			else:
				continue
			prev[totuple(pos)] = data['prev']
			if self.distance(pos,destination) < step:
				break
			# Add Children to Frontier
			for nextpos in self.nextpos(pos,step):
				nextt = data['t'] + self.tRes
				nextdist = data['dist'] + self.distance(nextpos,pos) * self.tRes
				# direction = destination - data['pos']
				# direction /= np.linalg.norm(direction)
				nextJ = self.heuristic(nextpos,nextt,destination) + nextdist # - 0.2*np.dot(direction[:,0],(nextpos-pos)[:,0])/nextt
				nextdata = {'pos': nextpos, 't': nextt, 'dist': nextdist, 'prev': data['pos']}
				frontier.put((nextJ, next(ties), nextdata))
		
		# Convert Linked List to NP Array
		curr = data['pos']
		path = []
		while not np.all(curr == None):
			path.append(curr)
			curr = prev[totuple(curr)]
		path.reverse()
		path = np.array(path)[:,:,0]
		
		self.path = path
		return path


def totuple(arr):
	return tuple(np.reshape(arr,(-1,)))


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
	o1.velocity = np.array([[-0.5,1,0]]).T
	m.objects.add(o1)

	o2 = Object(m)
	o2.position = np.array([[-1,3,0]]).T
	o2.velocity = np.array([[-1,-0.5,0]]).T
	m.objects.add(o2)

	o3 = Object(m)
	o3.position = np.array([[-1.5,1.5,0]]).T
	o3.velocity = np.array([[0.5,0.5,0]]).T
	m.objects.add(o3)

	o = Object(m)
	o.type = 1
	o.velocity = np.array([[-1,2,0]]).T
	o.position = np.array([[0,0,0]]).T
	m.objects.add(o)

	# dest = np.array([[-2,5,0]]).T
	path = o.pathplan()
	m.plot(T=3,lim=[-3,3,-1,6])
	# import code; code.interact(local=locals())

	# T = 1
	# t = np.arange(0,T,o.tRes)
	# m.plotObects(t)
	# plt.plot(path[:,0],path[:,1])
	# plt.show()
