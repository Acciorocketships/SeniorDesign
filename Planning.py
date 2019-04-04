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
		self.targetvel = None
		self.targetrot = None

		# Current State
		self.pos = np.array([0,0,0])
		self.vel = np.array([0,0,0])
		self.rot = np.array([0,0,0])

		# More Current State Values
		self.angvel = np.array([0,0,0])
		self.propangvel = np.array([0,0,0,0])

		# Planned Path
		self.roughpath = np.array([])
		self.path = np.array([])
		self.control = np.array([])
		self.motorcontrol = np.array([])
		self.t = np.array([])




	def astar(self):

		o = Object(self.map)
		o.position = self.pos
		o.speed = 1
		self.roughpath, self.t = o.pathplan(destination=self.targetpos,dt=0.1,returnlevel=1)




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

		self.path = np.array([x,y,z]).T










def Hermite(p,v,a,t,teval,nderiv=0):
	p = np.array(p)
	v = np.array(v)
	a = np.array(a)
	t = np.array(t)
	teval = np.array(teval)
	# Calculate the time indices
	dt = t[1] - t[0]
	tidx = np.floor((teval-t[0]) / dt).astype(np.int)
	# Calculate the time indices for non-constant time steps
	if not np.all(t[tidx] == t[0] + tidx * dt):
		tidx = []
		for i in range(teval.size):
			if teval[i] <= t[0]:
				tidx.append(0)
			elif teval >= t[-1]:
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
	for i in range(tidx.size):
		G = np.stack((p[tidx[i],:], p[tidx[i]+1,:], v[tidx[i],:], v[tidx[i]+1,:], a[tidx[i],:], a[tidx[i]+1,:]), axis=1)
		u = (teval[i]-t[tidx[i]]) / (t[tidx[i]+1]-t[tidx[i]])
		U = np.array([nPr(0,nderiv) * (u ** max(0,0-nderiv)),
					  nPr(1,nderiv) * (u ** max(0,1-nderiv)),
					  nPr(2,nderiv) * (u ** max(0,2-nderiv)),
					  nPr(3,nderiv) * (u ** max(0,3-nderiv)),
					  nPr(4,nderiv) * (u ** max(0,4-nderiv)),
					  nPr(5,nderiv) * (u ** max(0,5-nderiv))])
		result.append(G @ M @ U)
	result = np.array(result)
	return result


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








def test_astar():
	planner = Planner()
	planner.targetpos = np.array([2,0,0])
	planner.vel = np.array([0,1,0])
	planner.astar()
	plotPaths(planner.roughpath)


def test_calcHermite():
	path = np.array([[0,0,0],[1,0,0],[1.5,1,0],[1,1.5,0]])
	t = np.array([1,2,3,4])
	p, v, a = calcHermite(path)
	spline = Hermite(p,v,a,t,np.linspace(1,3.9,100),nderiv=0)
	plotPaths(spline)


def test_mpc():
	planner = Planner()
	planner.targetpos = np.array([2,0,0])
	planner.vel = np.array([0,1,0])
	planner.astar()
	planner.mpc()
	plotPaths((planner.roughpath,planner.path))


if __name__ == '__main__':
	
	test_calcHermite()




