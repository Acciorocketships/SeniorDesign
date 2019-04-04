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
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
				cost += obj.proxCost(pos=pos,t=t)
		return cost


	def plotObects(self, t=0):
		if not isinstance(t, np.ndarray):
			t = np.array([t])
		t = np.reshape(t,(-1,))
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		for obj in self.objects:
			if obj.type == 0:
				color = np.random.rand(3)
				for i in range(t.size):
					pos = obj.pos(t[i])
					ax.plot([pos[0]], [pos[1]], [pos[2]], 'o', c=color, alpha=(i+1)/t.size)


	def plot(self,T=3,lim=[-5,5,-5,5],plane=[0,1]):
		self.plane = plane
		self.fig, ax = plt.subplots()
		plt.xlim(lim[0],lim[1])
		plt.ylim(lim[2],lim[3])
		plt.subplots_adjust(bottom=0.25)
		colors = [np.random.rand(3) for i in range(len(self.objects))]
		self.axes = [None for i in range(len(self.objects))]
		for i, obj in enumerate(self.objects):
			pos = obj.pos(0)
			if obj.type == 0:
				self.axes[i] = plt.plot(pos[plane[0]], pos[plane[1]], 'o', c=colors[i])[0]
			else:
				self.axes[i] = plt.plot(pos[plane[0]], pos[plane[1]], '*', c=colors[i])[0]
				dest = obj.pos(T)
				plt.plot(dest[plane[0]],dest[plane[1]],'k.')
		axt = plt.axes([0.125, 0.1, 0.775, 0.03])
		tslider = Slider(axt, 'Time', 0, T, valinit=0)
		tslider.on_changed(self.update)
		plt.show()


	def update(self,t):
		for i, obj in enumerate(self.objects):
			pos = obj.pos(t)
			self.axes[i].set_xdata(pos[self.plane[0]])
			self.axes[i].set_ydata(pos[self.plane[1]])
		self.fig.canvas.draw_idle()






class Object:

	def __init__(self, map=None):

		self.m = map

		# Initial Position, Velocity, and Acceleration of the Object
		self.position = np.zeros(3)
		self.velocity = np.zeros(3)
		self.acceleration = np.zeros(3)
		self.speed = 1

		# Other Attributes
		self.type = 0 # 0 for moving, 1 for intelligent
		self.classification = None # classification from RCNN
		self.radius = 0.6
		self.gaussian = GaussND(numN=(np.array([[0,0,0]]).T,self.radius*np.identity(3)))
		self.gaussian = self.gaussian / self.gaussian[[0,0,0]]

		# Pathfinding Weights
		self.Cheur = 1
		self.Ctimedist = 1.5
		self.Cprox = 0.1
		self.Cdist = 0.2
		self.Cdir = 0.1
		self.Ctime = 0.2


		# Internal variables
		self.path = None

		# Options
		self.tRes = 0.05


	# The cost associated with being withing a given proximity of the object
	def proxCost(self,pos,t=0):
		disp = self.distance(pos,self.pos(t))
		if disp < self.radius:
			return float('inf')
		else:
			return self.gaussian[np.reshape(pos-self.pos(t),(-1,3))]


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


	def discretize(self,val,step):
		for i in range(len(val)):
			val[i] = round(val[i] / step) * step
		return val


	def nextpos(self,pos,step):
		for dx in [-step, 0, step]:
			for dy in [-step, 0, step]:
				for dz in [-step, 0, step]:
					yield pos + np.array([dx,dy,dz])


	def pathplan(self,destination=None,dt=0.1,returnlevel=1):

		# Initialization
		if destination is None:
			destination = self.extrapolate(3)
		frontier = PriorityQueue()
		maxsteps = 100000
		visited = {}
		if self.speed == 0 and np.linalg.norm(self.velocity) != 0:
			self.speed = np.linalg.norm(self.velocity)
		elif self.speed == 0:
			self.speed = 1
		self.tRes = dt
		step = max(round(self.tRes * self.speed, 2), 0.01)
		totaldist = self.distance(self.position,destination)
		totaltime = totaldist / self.speed
		if totaldist == 0:
			totaldist = 1
		if totaltime == 0:
			totaltime = 1

		# Starting point
		ties = count()
		pos = self.discretize(self.position,step)
		heuristic = self.distance(pos,destination)
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
			if self.distance(pos,destination) < step-1E-12:
				break
			# Add Children to Frontier
			for nextpos in self.nextpos(pos,step):
				nextt = data['t'] + self.tRes
				nextdist = data['dist'] + self.distance(nextpos,pos)
				nextProx = self.m.proxCost(pos=nextpos,t=nextt,ignore=self)
				heuristic = self.distance(nextpos,destination)

				vel = (nextpos-pos)
				lastvel = (pos - data['prev']['pos']) if data['prev'] != None else self.velocity / self.speed * np.linalg.norm(vel)
				mag = np.linalg.norm(vel) * np.linalg.norm(lastvel)
				dirchange = (1 - np.dot(vel,lastvel) / mag) if mag != 0 else 0

				nextJ = self.Cheur * (heuristic / totaldist) + \
						self.Cdist * (nextdist / totaldist) + \
						self.Ctimedist * (heuristic / totaldist * nextt / totaltime) + \
						self.Ctime * (nextt / totaltime) + \
						self.Cprox * nextProx + \
						self.Cdir * dirchange

				nextdata = {'pos': nextpos, 
							't': nextt,
							'togo': self.Cheur * (heuristic / totaldist), 
							'dist': nextdist / totaldist, 
							'J': nextJ, 
							'prox': self.Cprox * nextProx, 
							'prev': data}
				frontier.put((nextJ, next(ties), nextdata))
		
		# Convert Linked List to NP Array
		curr = data
		path = []
		if returnlevel > 0:
			t = []
		if returnlevel > 1:
			dist = []
			J = []
			prox = []
			togo = []
		while not np.all(curr == None):
			path.append(curr['pos'])
			if returnlevel > 0:
				t.append(curr["t"])
			if returnlevel > 1:
				dist.append(curr['dist'])
				J.append(curr["J"])
				prox.append(curr["prox"])
				togo.append(curr["togo"])
			curr = curr['prev']
		path.reverse()
		if returnlevel > 0:
			t.reverse()
			t = np.array(t)
		if returnlevel > 1:
			dist.reverse()
			dist = np.array(dist)
			J.reverse()
			J[0] = J[1]
			J = np.array(J)
			prox.reverse()
			prox[0] = prox[1]
			prox = np.array(prox)
			togo.reverse()
			togo = np.array(togo)
		
		path = np.array(path)
		t = np.array(t)
		self.path = path
		self.t = t

		if returnlevel > 1:
			return (path,t,dist,J,prox,togo)
		elif returnlevel > 0:
			return (path,t)
		else:
			return path


def totuple(arr):
	return tuple(np.reshape(arr,(-1,)))


def plotPaths(paths):
	if type(paths) != tuple:
		paths = (paths,)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	for i in range(len(paths)):
		ax.plot(paths[i][:,0],paths[i][:,1],paths[i][:,2],'o-')
	plt.show()



if __name__ == '__main__':

	m = Map()

	o1 = Object(m)
	o1.position = np.array([0,1,0])
	o1.velocity = np.array([-0.5,1,0])
	m.objects.add(o1)

	o2 = Object(m)
	o2.position = np.array([-1,3,0])
	o2.velocity = np.array([-1,-0.5,0])
	m.objects.add(o2)

	o3 = Object(m)
	o3.position = np.array([-1.5,1.5,0])
	o3.velocity = np.array([0.5,0.5,0])
	m.objects.add(o3)

	o4 = Object(m)
	o4.position = np.array([0.8,-0.6,0])
	o4.velocity = np.array([-0.5,0.5,0])
	m.objects.add(o4)

	o5 = Object(m)
	o5.position = np.array([0.5,0.7,0])
	o5.velocity = np.array([0.2,-0.4,0])
	m.objects.add(o5)

	o6 = Object(m)
	o6.position = np.array([-0.2,-0.5,0])
	o6.velocity = np.array([1,0,0])
	m.objects.add(o6)

	o7 = Object(m)
	o7.position = np.array([-0.7,0,0])
	o7.velocity = np.array([0.8,0.4,0])
	m.objects.add(o7)

	o = Object(m)
	o.type = 1
	o.velocity = np.array([2,0,0])
	o.position = np.array([0,0,0])
	m.objects.add(o)

	dest = np.array([1,1.5,0])
	path, t, dist, J, prox, togo = o.pathplan(destination=dest,dt=0.1,returnlevel=2)

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

	m.plotObects(t)
	plt.plot(path[:,0],path[:,1])
	plt.xlim(-4,4)
	plt.ylim(-4,4)
	plt.show()
