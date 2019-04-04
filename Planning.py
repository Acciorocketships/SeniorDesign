from gekko import GEKKO as solver
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Map import *



class Planner:


	def __init__(self, map=Map()):

		self.map = map

		# Target
		self.targetpos = np.array([0,0,0])
		self.targetvel = np.array([0,0,0])
		self.targetacc = np.array([0,0,0])

		# Current State
		self.pos = np.array([0,0,0])
		self.vel = np.array([0,0,0])
		self.acc = np.array([0,0,0])

		# More Current State Values (for MPC)
		self.rot = np.array([0,0,0])
		self.angvel = np.array([0,0,0])
		self.propangvel = np.array([0,0,0,0])

		# Planned Path
		self.p = np.zeros((1,3))
		self.v = np.zeros((1,3))
		self.t = np.zeros((1,))
		# Path Planning Intermediates
		self.roughpath = np.zeros((1,3))
		self.rought = np.zeros((1,))




	# Plan a path using differential flatness
	def spline(self, t=None):

		self.astar()

		path_pruned, t_pruned = prunePath(self.roughpath, self.rought)

		p, v, a = calcHermite(path_pruned, v0=self.vel, vT=self.targetvel, a0=self.acc, aT=self.targetacc)

		if t is not None:
			self.t = t
		else:
			self.t = self.rought
		self.p = hermite(p, v, a, t_pruned, self.t, nderiv=0)
		self.v = hermite(p, v, a, t_pruned, self.t, nderiv=1)



	# Plan a path using model predictive control (nonlinear optimization)
	def mpc(self):

		s = solver()

		## Quadcopter Parameters ##

		l = s.Param(value=0.25, name="Quadcopter Arm Length")
		b = s.Param(value=3E-5, name="Thrust Coefficient")
		d = s.Param(value=1.1E-6, name="Drag Coefficient")
		m = s.Param(value=2.0, name="Mass")
		Ixy = s.Param(value=0.0337, name="Moment of Inertia about X and Y Axes")
		Iz = s.Param(value=0.0185, name="Moment of Inertia about Z Axis")
		Jr = s.Param(value=2.74E-4, name="Inertia of Rotor") # pg.32 http://www.diva-portal.org/smash/get/diva2:1020192/FULLTEXT02.pdf
		g = s.Param(value=9.81, name="Gravity")


		## Variables ##

		# Control Inputs
		W = s.Array(s.MV,lb=0,ub=8200,dim=4) # Angular velocity of the 4 motors
		for i in range(4):
			W[i].value = self.propangvel[i]
			W[i].STATUS = 1
			# Add DCOST (penalization for change) and DMAX (max change per step)

		# Thrust, Pitch, Roll, Yaw from Control Inputs
		U1 = s.Intermediate(b * (W[0]**2 + W[1]**2 + W[2]**2 + W[3]**3), name="Thrust Control")
		U2 = s.Intermediate(-l*b * (1/s.sqrt(2)*(-W[1]**2 + W[3]**2) + 1/s.sqrt(2)*(W[0]**2 - W[2]**2)), name="Pitch Control")
		U3 = s.Intermediate(-l*b * (-1/s.sqrt(2)*(-W[1]**2 + W[3]**2) + 1/s.sqrt(2)*(W[0]**2 - W[2]**2)), name="Roll Control")
		U4 = s.Intermediate(d * (-W[0]**2 + W[1]**2 - W[2]**2 + W[3]**3), name="Yaw Control")
		Wr = s.Intermediate(-W[0] + W[1] - W[2] + W[3], name="Residual Rotor Speed")

		# State Variables
		phi = s.SV(value=self.rot[0], name="Pitch")
		theta = s.SV(value=self.rot[1], name="Roll")
		psi = s.SV(value=self.rot[2], name="Yaw")

		# Rotations
		ux = s.Intermediate(s.cos(phi)*s.sin(theta)*s.cos(psi) + s.sin(phi)*s.sin(psi))
		uy = s.Intermediate(s.cos(phi)*s.sin(theta)*s.sin(psi) - s.sin(phi)*s.cos(psi))

		phidot = s.SV(value=self.angvel[0], name="Pitch Derivative")
		thetadot = s.SV(value=self.angvel[1], name="Roll Derivative")
		psidot = s.SV(value=self.angvel[2], name="Yaw Derivative")

		vx = s.SV(value=self.vel[0], name="Vx")
		vy = s.SV(value=self.vel[1], name="Vy")
		vz = s.SV(value=self.vel[2], name="Vz")


		## Set Target Path ##

		if self.t.size == 0:
			# Linear path to goal if none already
			self.t = np.linspace(0,3,100)
			self.roughpath = np.array(self.pos.reshape(1,3) + self.t*(self.targetpos-self.pos).reshape(1,3))
		roughpath = self.roughpath.T

		s.time = self.t
		x = s.CV(value=roughpath[0,:], name="X")
		y = s.CV(value=roughpath[1,:], name="Y")
		z = s.CV(value=roughpath[2,:], name="Z")

		x.STATUS = 1
		y.STATUS = 1
		z.STATUS = 1
		x.FSTATUS = 1
		y.FSTATUS = 1
		z.FSTATUS = 1

		s.Equation(x.dt() == vx)
		s.Equation(y.dt() == vy)
		s.Equation(z.dt() == vz)


		## Equations of Motion ##

		# Orientation Derivative
		s.Equation(phi.dt() == phidot)
		s.Equation(theta.dt() == thetadot)
		s.Equation(psi.dt() == psidot)

		# Angular Velocity Derivative
		s.Equation(phidot.dt() == (thetadot * psidot * (Ixy-Iz)/Ixy) + (thetadot * Jr/Ixy * Wr) - (1/Ixy * U2))
		s.Equation(thetadot.dt() == (phidot * psidot * (Iz-Ixy)/Ixy) - (phidot * Jr/Ixy * Wr) - (1/Ixy * U3))
		s.Equation(psidot.dt() == 1/Iz * U4)

		# Velocity Derivative
		s.Equation(vx.dt() == ux/m * U1)
		s.Equation(vy.dt() == uy/m * U1)
		s.Equation(vz.dt() == g - s.cos(phi)*s.cos(theta)/m * U1)


		## Solve ##

		s.options.CV_TYPE = 1 # squared error
		s.options.IMODE = 6 # dynamic control
		s.options.SOLVER = 3
		s.options.MAX_ITER = 10000
		s.solve(disp=True)

		self.p = np.array([x,y,z]).T




	def astar(self):

		o = Object(self.map)
		o.position = self.pos
		o.speed = 1
		self.roughpath, self.rought = o.pathplan(destination=self.targetpos,dt=0.1,returnlevel=1)







# Removes all collinear points in a path
def prunePath(path,t):
	pathout = [path[0,:]]
	tout = [t[0]]
	for i in range(1,path.shape[0]-1):
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
	v1 /= mag1
	v2 /= mag2
	return np.dot(v1.reshape((-1,)),v2.reshape((-1,))) == 1

# Calculates the value of the spline at any time teval, 
# given the hermite coefficients (position, velocity, acceleration, corresponding time)
def hermite(p,v,a,t,teval,nderiv=0):
	p = np.array(p)
	v = np.array(v)
	a = np.array(a)
	t = np.array(t)
	teval = np.array(teval)
	# Calculate the time indices
	dt = t[1] - t[0]
	tidx = np.floor((teval-t[0]) / dt).astype(np.int)
	# Calculate the time indices for non-constant time steps
	if not np.linalg.norm(t[tidx] - (t[0] + tidx * dt)) < 1E-12:
		tidx = []
		for i in range(teval.size):
			if teval[i] <= t[0]:
				tidx.append(0)
			elif teval[i] >= t[-1]:
				tidx.append(t.size-1)
			else:
				tidx.append(np.argmax(t>teval[i]))
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
	for i in range(tidx.size-1):
		G = np.stack((p[tidx[i],:], p[tidx[i]+1,:], v[tidx[i],:], v[tidx[i]+1,:], a[tidx[i],:], a[tidx[i]+1,:]), axis=1)
		u = (teval[i]-t[tidx[i]]) / (t[tidx[i]+1]-t[tidx[i]])
		U = np.array([nPr(0,nderiv) * (u ** max(0,0-nderiv)),
					  nPr(1,nderiv) * (u ** max(0,1-nderiv)),
					  nPr(2,nderiv) * (u ** max(0,2-nderiv)),
					  nPr(3,nderiv) * (u ** max(0,3-nderiv)),
					  nPr(4,nderiv) * (u ** max(0,4-nderiv)),
					  nPr(5,nderiv) * (u ** max(0,5-nderiv))])
		result.append(G @ (M @ U))
	result = np.array(result)
	return result

# Calculates the hermite coefficients (velocity and acceleration) given a list of positions
def calcHermite(p, v0=np.zeros((1,3)), vT=np.zeros((1,3)), a0=np.zeros((1,3)), aT=np.zeros((1,3))):

	p = np.array(p)
	v0 = np.array(v0)
	vT = np.array(vT)
	a0 = np.array(a0)
	aT = np.array(aT)

	A, b = Ab(p,v0,vT,a0,aT)

	c = np.linalg.solve(A,b)

	N = p.shape[0]
	v = np.zeros((N,3))
	a = np.zeros((N,3))
	for i in range(N):
		v[i,:] = c[4*i,:]
		a[i,:] = c[4*i+2,:]
	v[N-1,:] = c[4*(N-1)+1,:]
	a[N-1,:] = c[4*(N-1)+3,:]

	return (p,v,a)


# Calculates the A and b matrices for a Hermite Spline given a list of knots
def Ab(p, v0=np.zeros((1,3)), vT=np.zeros((1,3)), a0=np.zeros((1,3)), aT=np.zeros((1,3))):

	N = p.shape[0]
	A = np.zeros((4*N,4*N))
	b = np.zeros((4*N,3))
	p = np.concatenate((p,np.array([2*p[-1,:]-p[-2,:]])),axis=0)

	for i in range(N-1):

		A[4*i,4*i+1] = 1
		A[4*i,4*(i+1)] = -1

		A[4*i+1,4*i+3] = 1
		A[4*i+1,4*(i+1)+2] = -1

		A[4*i+2,4*i] = 24
		A[4*i+2,4*i+1] = 36
		A[4*i+2,4*i+2] = 3
		A[4*i+2,4*i+3] = -9
		A[4*i+2,4*(i+1)] = -36
		A[4*i+2,4*(i+1)+1] = -24
		A[4*i+2,4*(i+1)+2] = -9
		A[4*i+2,4*(i+1)+3] = 3
		b[4*i+2,:] = 60*(-p[i,:]+2*p[i+1,:]-p[i+2,:])

		A[4*i+3,4*i] = 168
		A[4*i+3,4*i+1] = 192
		A[4*i+3,4*i+2] = 24
		A[4*i+3,4*i+3] = -36
		A[4*i+3,4*(i+1)] = -192
		A[4*i+3,4*(i+1)+1] = -168
		A[4*i+3,4*(i+1)+2] = 36
		A[4*i+3,4*(i+1)+3] = -24
		b[4*i+3,:] = 360*(p[i,:]-p[i+2,:])

	A[4*(N-1),0] = 1
	b[4*(N-1),:] = v0.reshape((1,3))

	A[4*(N-1)+1,4*(N-1)] = 1
	b[4*(N-1)+1,:] = vT.reshape((1,3))

	A[4*(N-1)+2,2] = 1
	b[4*(N-1)+2,:] = a0.reshape((1,3))

	A[4*(N-1)+3,4*(N-1)+3] = 1
	b[4*(N-1)+3,:] = aT.reshape((1,3))

	return (A,b)

def nPr(n,r):
	if r > n:
		return 0
	ans = 1
	for k in range(n,max(1,n-r),-1):
		ans = ans * k
	return ans






def create_map():
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
	o.pathplan(destination=dest,dt=0.1)

	return m


def test_astar():
	planner = Planner(map=create_map())
	planner.targetpos = np.array([2,0,0])
	planner.vel = np.array([0,1,0])
	planner.astar()
	import pdb; pdb.set_trace()
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
	print(newpath)
	print(newt)


def test_spline(map=create_map()):
	planner = Planner()
	planner.targetpos = np.array([1,1.5,0])
	planner.vel = np.array([1,0,0])
	planner.spline()
	plotPaths((planner.p, planner.roughpath))


if __name__ == '__main__':
	
	test_astar()




