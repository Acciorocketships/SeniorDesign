from __future__ import division
from __future__ import absolute_import
from gekko import GEKKO as solver
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Map import *



class Planner(object):


	def __init__(self, map=Map(), target=None, state=None, path=None):

		self.map = map

		# Target
		if target is not None:
			self.target = target
		else:
			self.target = {u'pos':np.zeros((3,)), 
						  u'vel':np.array((3,)),
						  u'speed': 1}

		# Current State
		if state is not None:
			self.state = state
		else:
			self.state = {u'pos':np.zeros((3,)), 
						  u'vel':np.array((3,))}

		# Planned Path
		if path is not None:
			self.path = path
		else:
			self.path = {u'x': np.zeros((0,3)),
						 u'v': np.zeros((0,3)),
						 u't': np.zeros((0,)),
						 u'roughx': np.zeros((0,3)),
						 u'rought': np.zeros((0,3)),}

		self.p = None
		self.v = None
		self.a = None




	# Plan a path using differential flatness
	def plan(self, dt_astar=0.1, dt_out=None):

		discpath, disct = self.astar(dt=dt_astar)

		self.path[u'roughx'], self.path[u'rought'] = prunePath(discpath, disct)

		p, v, a = calcHermiteQP(self.path[u'roughx'], v0=self.state[u'vel'], vT=self.target[u'vel'], speed=self.target[u'speed'])

		if dt_out is not None:
			self.path[u't'] = np.arange(self.path[u'rought'][0], self.path[u'rought'][-1], dt_out)
		else:
			self.path[u't'] = self.disct
		self.path[u'x'] = hermite(p=p, v=v, a=a, t=self.path[u'rought'], teval=self.path[u't'], nderiv=0)
		self.path[u'v'] = hermite(p=p, v=v, a=a, t=self.path[u'rought'], teval=self.path[u't'], nderiv=1)



	# Helper for spline
	def astar(self, dt=0.1):

		o = Object(self.map)
		o.position = self.state[u'pos']
		o.speed = self.target[u'speed']
		self.map.objects.add(o)
		path, t = o.pathplan(destination=self.target[u'pos'], dt=dt, returnlevel=1)
		return (path, t)






# Removes all collinear points in a path
def prunePath(path,t):
	pathout = [path[0,:]]
	tout = [t[0]]
	for i in xrange(1,path.shape[0]-1):
		if not isCollinear(pathout[-1],path[i,:],path[i+1,:]):
			pathout.append(path[i,:])
			tout.append(t[i])
	pathout.append(path[-1,:])
	tout.append(t[-1])
	pathout = np.array(pathout)
	tout = np.array(tout)
	return (pathout,tout)

# Helper function for prunePath
def isCollinear(p1,p2,p3):
	v1 = p2 - p1
	v2 = p3 - p2
	mag1 = np.linalg.norm(v1)
	mag2 = np.linalg.norm(v2)
	if mag1 == 0 or mag2 == 0:
		return True
	v1 = v1 / mag1
	v2 = v2 / mag2
	return abs(abs(np.dot(v1.reshape((-1,)),v2.reshape((-1,)))) - 1) < 1E-12

# Calculates the value of the spline at any time teval, 
# given the hermite coefficients (position, velocity, acceleration, corresponding time)
def hermite(p,v,a,t,teval,nderiv=0,equalsteps=False):
	p = np.array(p)
	v = np.array(v)
	a = np.array(a)
	t = np.array(t)
	teval = np.array(teval)
	# Calculate the time indices
	if equalsteps:
		tidx = np.floor((teval-t[0]) / (t[1] - t[0])).astype(np.int)
	# Calculate the time indices for non-constant time steps
	else:
		tidx = []
		for i in xrange(teval.size):
			if teval[i] <= t[0]:
				tidx.append(0)
			elif teval[i] >= t[-1]:
				tidx.append(t.size-2)
			else:
				tidx.append(np.argmax(t>teval[i])-1)
		tidx = np.array(tidx)
	M = np.array([[1,   0,   0, -10,  15,  -6],
		 		  [0,   0,   0,  10, -15,   6],
		 		  [0,   1,   0,  -6,   8,  -3],
		 		  [0,   0,   0,  -4,   7,  -3],
				  [0,   0, 0.5,-1.5, 1.5,-0.5],
		 		  [0,   0,   0, 0.5,  -1, 0.5]])
	result = []
	if len(tidx.shape)==0:
		tidx = np.array([tidx])
	for i in xrange(tidx.size-1):
		G = np.stack((p[tidx[i],:], p[tidx[i]+1,:], v[tidx[i],:], v[tidx[i]+1,:], a[tidx[i],:], a[tidx[i]+1,:]), axis=1)
		u = (teval[i]-t[tidx[i]]) / (t[tidx[i]+1]-t[tidx[i]])
		U = np.array([nPr(0,nderiv) * (u ** max(0,0-nderiv)),
					  nPr(1,nderiv) * (u ** max(0,1-nderiv)),
					  nPr(2,nderiv) * (u ** max(0,2-nderiv)),
					  nPr(3,nderiv) * (u ** max(0,3-nderiv)),
					  nPr(4,nderiv) * (u ** max(0,4-nderiv)),
					  nPr(5,nderiv) * (u ** max(0,5-nderiv))])
		result.append(G.dot(M.dot(U)))
	result = np.array(result)
	return result


def calcHermiteQP(p, v0=np.zeros((1,3)), vT=np.zeros((1,3)), speed=1):

	p = np.array(p)
	v0 = np.array(v0)
	vT = np.array(vT)
	N = p.shape[0]-1

	# Constraints

	E, d = Ab(p=p, v0=v0, vT=vT, speed=speed)
	E = E[:4*(N-1)+2,:]
	d = d[:4*(N-1)+2,:]


	# Jerk Cost Function

	Qj = np.zeros((4*N,4*N))
	cj = np.zeros((4*N,3))
	for i in xrange(N):

		p0 = p[i,:]
		p1 = p[i+1,:]

		Qj[4*i+0,4*i+0] = 192
		Qj[4*i+0,4*i+1] = 336
		Qj[4*i+0,4*i+2] = 72
		Qj[4*i+0,4*i+3] = -48

		Qj[4*i+1,4*i+1] = 192
		Qj[4*i+1,4*i+2] = 48
		Qj[4*i+1,4*i+3] = -72

		Qj[4*i+2,4*i+2] = 9
		Qj[4*i+2,4*i+3] = -6

		Qj[4*i+3,4*i+3] = 9

		cj[4*i+0,:] = 720*(p0-p1)
		cj[4*i+1,:] = 720*(p0-p1)
		cj[4*i+2,:] = 120*(p0-p1)
		cj[4*i+3,:] = -120*(p0-p1)

	Qj *= 2


	# Position Cost Function

	# Qp = np.zeros((4*N,4*N))
	# cp = np.zeros((4*N,3))
	# for i in range(N):

	# 	p0 = p[i,:]
	# 	p1 = p[i+1,:]

	# 	Qp[4*i+0,4*i+0] = 52/3465
	# 	Qp[4*i+0,4*i+1] = -19/990
	# 	Qp[4*i+0,4*i+2] = 23/9240
	# 	Qp[4*i+0,4*i+3] = 13/6930

	# 	Qp[4*i+1,4*i+1] = 52/3465
	# 	Qp[4*i+1,4*i+2] = -13/6930
	# 	Qp[4*i+1,4*i+3] = -23/9240

	# 	Qp[4*i+2,4*i+2] = 1/9240
	# 	Qp[4*i+2,4*i+3] = 1/5544

	# 	Qp[4*i+3,4*i+3] = 1/9240

	# 	cp[4*i+0,:] = -4/385*p0 + -5/462*p1
	# 	cp[4*i+1,:] = -101/2310*p0 + -5/462*p1
	# 	cp[4*i+2,:] = -37/27720*p0 + -17/27720*p1
	# 	cp[4*i+3,:] = 1/6930*p0 + 17/27720*p1

	# Qp *= 2

	# a = 0
	# Q = a*10000*Qp + (1-a)*Qj
	# c = a*10000*cp + (1-a)*cj

	Q = Qj
	c = cj

	# Quadratic Programming Formula

	cost = np.concatenate((Q, E.T), axis=1)
	constraints = np.concatenate((E, np.zeros((E.shape[0],E.shape[0]))), axis=1)
	A = np.concatenate((cost, constraints), axis=0)
	b = np.concatenate((-c, d), axis=0)

	x = np.linalg.solve(A,b)

	v = np.zeros((N+1,3))
	a = np.zeros((N+1,3))
	for i in xrange(N):
		v[i,:] = x[4*i,:]
		a[i,:] = x[4*i+2,:]
	v[N,:] = x[4*(N-1)+1,:]
	a[N,:] = x[4*(N-1)+3,:]

	return (p,v,a)



# Calculates the hermite coefficients (velocity and acceleration) given a list of positions
def calcHermite(p, v0=np.zeros((1,3)), vT=np.zeros((1,3)), speed=1):

	p = np.array(p)
	v0 = np.array(v0)
	vT = np.array(vT)
	N = p.shape[0]-1

	A, b = Ab(p,v0,vT,speed=speed)

	c = np.linalg.solve(A,b)

	v = np.zeros((N+1,3))
	a = np.zeros((N+1,3))
	for i in xrange(N):
		v[i,:] = c[4*i,:]
		a[i,:] = c[4*i+2,:]
	v[N,:] = c[4*(N-1)+1,:]
	a[N,:] = c[4*(N-1)+3,:]

	return (p,v,a)


# Calculates the A and b matrices for a Hermite Spline given a list of knots
def Ab(p, v0=np.zeros((1,3)), vT=np.zeros((1,3)), speed=1):

	N = p.shape[0]-1
	A = np.zeros((4*N,4*N))
	b = np.zeros((4*N,3))

	# Main Equations
	for i in xrange(N-1):

		# v_{j}(1) = v_{j+1}(0)
		A[4*i,4*i+1] = 1
		A[4*i,4*(i+1)] = -1

		# a_{j}(1) = a_{j+1}(0)
		A[4*i+1,4*i+3] = 1
		A[4*i+1,4*(i+1)+2] = -1

		# Continuous 3rd Derivative
		A[4*i+2,4*i] = 24
		A[4*i+2,4*i+1] = 36
		A[4*i+2,4*i+2] = 3
		A[4*i+2,4*i+3] = -9
		A[4*i+2,4*(i+1)] = -36
		A[4*i+2,4*(i+1)+1] = -24
		A[4*i+2,4*(i+1)+2] = -9
		A[4*i+2,4*(i+1)+3] = 3
		b[4*i+2,:] = 60*(-p[i,:]+2*p[i+1,:]-p[i+2,:])

		# Continuous 4th Derivative
		# A[4*i+3,4*i] = 168
		# A[4*i+3,4*i+1] = 192
		# A[4*i+3,4*i+2] = 24
		# A[4*i+3,4*i+3] = -36
		# A[4*i+3,4*(i+1)] = -192
		# A[4*i+3,4*(i+1)+1] = -168
		# A[4*i+3,4*(i+1)+2] = 36
		# A[4*i+3,4*(i+1)+3] = -24
		# b[4*i+3,:] = 360*(p[i,:]-p[i+2,:])

		# Catmull-Rom
		s0 = (p[i+1,:]-p[i,:]) / np.linalg.norm(p[i+1,:]-p[i,:])
		s1 = (p[i+2,:]-p[i+1,:]) / np.linalg.norm(p[i+2,:]-p[i+1,:])
		maxspeeddist = speed*1.0 # speed * time. The length of the next segment for which the speed will be at max
		speedratio = ((np.dot(s0,s1) + 1) / 2)  *  min(np.linalg.norm(p[i+1,:]-p[i,:]), maxspeeddist)*(min(np.linalg.norm(p[i+2,:]-p[i+1,:]), maxspeeddist) / maxspeeddist)

		A[4*i+3,4*i+1] = 1
		b[4*i+3,:] = 0.5 * (s1+s0)
		b[4*i+3,:] *= speed * speedratio


	# Set Start Velocity
	ca = 0
	A[4*(N-1)+ca,0] = 1
	b[4*(N-1)+ca,:] = v0.reshape((1,3))

	# Set End Velocity
	cb = 1
	A[4*(N-1)+cb,4*(N-1)+1] = 1
	b[4*(N-1)+cb,:] = vT.reshape((1,3))


	# Extra constraints (removed in optimization approach)


	# Set Start Acceleration to 0
	# cc = 2
	# A[4*(N-1)+cc,2] = 1
	# b[4*(N-1)+cc,:] = np.zeros((1,3))

	# Set End Acceleration to 0
	cd = 2
	A[4*(N-1)+cd,4*(N-1)+3] = 1
	b[4*(N-1)+cd,:] = np.zeros((1,3))


	# Set Start Jerk to 0
	# ce = 3
	# A[4*(N-1)+ce,0] = -36
	# A[4*(N-1)+ce,1] = -24
	# A[4*(N-1)+ce,2] = -9
	# A[4*(N-1)+ce,3] = 3
	# b[4*(N-1)+ce,:] = 60*(p[0,:]-p[1,:])

	# Set End Jerk to 0
	cf = 3
	A[4*(N-1)+cf,4*(N-1)] = -24
	A[4*(N-1)+cf,4*(N-1)+1] = -36
	A[4*(N-1)+cf,4*(N-1)+2] = -3
	A[4*(N-1)+cf,4*(N-1)+3] = 9
	b[4*(N-1)+cf,:] = 60*(p[N-2,:]-p[N-1,:])

	return (A,b)


def nPr(n,r):
	if r > n:
		return 0
	ans = 1
	for k in xrange(n,max(1,n-r),-1):
		ans = ans * k
	return ans






def create_map():

	m = Map()

	o1 = Object(m)
	o1.position = np.array([0,1,2])
	o1.velocity = np.array([-0.5,1,-0.4])
	m.objects.add(o1)

	o2 = Object(m)
	o2.position = np.array([-1,3,-1])
	o2.velocity = np.array([-1,-0.5,0.2])
	m.objects.add(o2)

	o3 = Object(m)
	o3.position = np.array([-1.5,1.5,0.5])
	o3.velocity = np.array([0.5,0.5,-0.1])
	m.objects.add(o3)

	o4 = Object(m)
	o4.position = np.array([0.8,-0.6,-1.5])
	o4.velocity = np.array([-0.5,0.5,0.5])
	m.objects.add(o4)

	o5 = Object(m)
	o5.position = np.array([0.5,0.7,0])
	o5.velocity = np.array([0.2,-0.4,0.2])
	m.objects.add(o5)

	o6 = Object(m)
	o6.position = np.array([-0.2,-0.5,0.3])
	o6.velocity = np.array([1,0,0.1])
	m.objects.add(o6)

	o7 = Object(m)
	o7.position = np.array([-0.7,0,0.4])
	o7.velocity = np.array([0.8,0.4,-0.1])
	m.objects.add(o7)

	o = Object(m)
	o.type = 1
	o.speed = 1
	o.position = np.array([-1,0,0])
	m.objects.add(o)

	dest = np.array([1,1.5,1])
	path, t = o.pathplan(destination=dest,dt=0.1,returnlevel=1)

	return m


def test_astar():
	planner = Planner(map=create_map())
	planner.pos = np.array([2,2,0])
	planner.targetpos = np.array([0,0,0])
	planner.vel = np.array([1,0,0])
	planner.astar()
	plotPaths(planner.roughpath)


def test_calcHermite():
	path = np.array([[0,0,0],[1,0,0],[2,0,0],[1.5,1,0],[1,1.5,0]])
	t = np.array([1,2,3,4,5])
	p, v, a = calcHermite(path)
	spline = hermite(p,v,a,t,np.linspace(1,3.9,100),nderiv=0)
	plotPaths(spline)


def test_mpc():
	planner = Planner()
	planner.targetpos = np.array([2,0,0])
	planner.vel = np.array([0,1,0])
	planner.astar()
	planner.mpc()
	plotPaths((planner.roughpath,planner.path))


def test_prunePath():
	path = np.array([[0,0,0],[1,0,0],[2,0,0],[1.5,1,0],[1,1.5,0]])
	t = np.array([1,2,3,4,5])
	newpath, newt = prunePath(path,t)
	print newpath
	print newt


def test_spline():
	planner = Planner(map=create_map())
	planner.state[u'pos'] = np.array([1,3,0])
	planner.target[u'pos'] = np.array([-1,-1,0])
	planner.state[u'vel'] = np.array([0,0,0])
	planner.target[u'vel'] = np.array([0,0,0])
	planner.plan(dt_out=0.01)
	ax = planner.map.plotObjects(t=planner.path[u'rought'], ax=None)
	plotPaths((planner.path[u'x'],planner.path[u'roughx']), ax=ax)


def plot_spline():
	planner = Planner(map=create_map())
	planner.state[u'pos'] = np.array([-2,-2,0.5])
	planner.target[u'pos'] = np.array([2,2,0.5])
	planner.state[u'vel'] = np.array([0,0,0])
	planner.target[u'vel'] = np.array([0,0,0])
	planner.plan(dt_out=0.01)
	viewer = Viewer(path=planner.path[u'x'],t=planner.path[u't'],map=planner.map)
	viewer.show()

if __name__ == u'__main__':
	
	plot_spline()




