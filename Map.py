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
				displacement = pos - obj.pos(t)
				cost += obj.proxCost(displacement)
		return cost


	def plotObects(self, t=0):
		if not isinstance(t, np.ndarray):
			t = np.array([t])
		t = np.reshape(t,(-1,))
		for obj in self.objects:
			if obj.type == 0:
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
			if obj.type == 0:
				self.axes[i] = plt.plot(pos[0], pos[1], 'o', c=colors[i])[0]
			else:
				self.axes[i] = plt.plot(pos[0], pos[1], '*', c=colors[i])[0]
				dest = obj.pos(T)
				plt.plot(dest[0],dest[1],'k.')
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
		self.gaussian = 30*GaussND(numN=(np.array([[0,0,0]]).T,1*np.identity(3)))

		# Pathfinding Weights
		self.Ctime = 0.5
		self.Cprox = 2
		self.Cdist = 1

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
		return self.distance(pos,destination)


	def discretize(self,val,step):
		for i in range(len(val)):
			val[i,0] = round(val[i,0] / step) * step
		return val


	def nextpos(self,pos,step):
		for dx in [-step, 0, step]:
			for dy in [-step, 0, step]:
				#for dz in [-step, 0, step]:
				dz = 0
				yield pos + np.array([[dx],[dy],[dz]])


	def pathplan(self,T=3,destination=None,returnall=False):

		# Initialization
		if destination is None:
			destination = self.extrapolate(T)
		frontier = PriorityQueue()
		maxsteps = 100000 #3*int(T / self.tRes)
		visited = {}
		self.speed = np.linalg.norm(self.velocity)
		step = round(0.1 * self.speed, 2)

		# Starting point
		ties = count()
		pos = self.discretize(self.position,step)
		heuristic = self.heuristic(pos,0,destination)
		data = {'pos': pos, 't': 0, 'togo': heuristic, 'dist': 0, 'J': 0, 'prox': 0, 'prev': None}
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
			if self.distance(pos,destination) < step:
				break
			# Add Children to Frontier
			for nextpos in self.nextpos(pos,step):
				nextt = data['t'] + self.tRes
				nextdist = data['dist'] + self.distance(nextpos,pos)
				nextProx = self.m.proxCost(pos=nextpos,t=nextt,ignore=self)
				heuristic = self.heuristic(nextpos,nextt,destination)
				nextJ = heuristic + self.Cdist * nextdist + self.Cprox * nextProx + self.Ctime * nextt
				nextdata = {'pos': nextpos, 't': nextt,'togo': heuristic, 'dist': nextdist, 'J': nextJ, 'prox': nextProx, 'prev': data}
				frontier.put((nextJ, next(ties), nextdata))
		
		# Convert Linked List to NP Array
		curr = data
		path = []
		if returnall:
			dist = []
			J = []
			prox = []
			t = []
			togo = []
		while not np.all(curr == None):
			path.append(curr['pos'])
			if returnall:
				dist.append(curr['dist'])
				J.append(curr["J"])
				prox.append(curr["prox"])
				t.append(curr["t"])
				togo.append(curr["togo"])
			curr = curr['prev']
		path.reverse()
		path = np.array(path)[:,:,0]
		if returnall:
			dist.reverse()
			dist = np.array(dist)
			J.reverse()
			J[0] = J[1]
			J = np.array(J)
			prox.reverse()
			prox[0] = prox[1]
			prox = np.array(prox)
			t.reverse()
			t = np.array(t)
			togo.reverse()
			togo = np.array(togo)
		
		self.path = path

		if returnall:
			return (path,dist,J,prox,togo,t)
		else:
			return path


def totuple(arr):
	return tuple(np.reshape(arr,(-1,)))








if __name__ == '__main__':

	from matplotlib import pyplot as plt

	m = Map()

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

	o4 = Object(m)
	o4.position = np.array([[0.8,-0.6,0]]).T
	o4.velocity = np.array([[-0.5,0.5,0]]).T
	m.objects.add(o4)

	o5 = Object(m)
	o5.position = np.array([[0.5,0.7,0]]).T
	o5.velocity = np.array([[0.2,-0.4,0]]).T
	m.objects.add(o5)

	o = Object(m)
	o.type = 1
	o.velocity = np.array([[-1,2,0]]).T
	o.position = np.array([[0,0,0]]).T
	m.objects.add(o)

	dest = np.array([[0,3,0]]).T
	path, dist, J, prox, togo, t = o.pathplan(destination=dest,returnall=True)

	m.plot(T=t[-1],lim=[-3,3,-1,6])

	f = plt.figure()
	f.add_subplot(2,2,1)
	plt.plot(t, J)
	plt.title("J")
	f.add_subplot(2,2,2)
	plt.plot(t, dist)
	plt.title("Dist Travelled")
	f.add_subplot(2,2,3)
	plt.plot(t, prox)
	plt.title("Proximity")
	f.add_subplot(2,2,4)
	plt.plot(t, togo)
	plt.title("Dist to Go")
	plt.show()

	m.plotObects(np.linspace(0,7,30))
	plt.plot(path[:,0],path[:,1])
	plt.xlim(-3,3)
	plt.ylim(-2,4)
	plt.show()
