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




	def pathplan(self):
		
		pass




	def astar(self):

		o = Object(self.map)
		o.position = self.pos
		o.speed = 1
		self.roughpath, self.t = o.pathplan(destination=self.targetpos,dt=0.1,returnlevel=1)



	def spline(self):

		pass



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




def calcHermite(p, v0=np.zeros((3,1)), vT=np.zeros((3,1)), a0=np.zeros((3,1)), aT=np.zeros((3,1))):

	N = p.shape[0]
	A = np.zeros((4*N,4*N))
	b = np.zeros((4*N,3))

	for i in range(N-1):

		A[4*i,4*i+1] = 1
		A[4*i,4*(i+1)] = -1

		A[4*i+1,4*i+3] = 1
		A[4*i+1,4*(i+1)+2] = -1

		A[4*i+2,4*i] = 12
		A[4*i+2,4*i+1] = -12
		A[4*i+2,4*i+2] = 6
		A[4*i+2,4*i+3] = 6

		A[4*i+3,4*i] = -24
		A[4*i+3,4*i+1] = 24
		A[4*i+3,4*i+2] = 60
		A[4*i+3,4*i+3] = -60
		b[4*i+3,:] = 720*(p[i+1,:] - p[i,:])

	A[4*(N-1),0] = 1
	b[4*(N-1),:] = v0.reshape((1,3))

	A[4*(N-1)+1,4*(N-1)] = 1
	b[4*(N-1)+1,:] = vT.reshape((1,3))

	A[4*(N-1)+2,2] = 1
	b[4*(N-1)+2,:] = a0.reshape((1,3))

	A[4*(N-1)+3,4*(N-1)+3] = 1
	b[4*(N-1)+3,:] = aT.reshape((1,3))

	c = np.linalg.solve(A,b)

	v = np.zeros((N,3))
	a = np.zeros((N,3))
	for i in range(N):
		v[i,:] = c[4*i,:]
		a[i,:] = c[4*i+2,:]
	v[N-1,:] = c[4*(N-1)+1,:]
	a[N-1,:] = c[4*(N-1)+3,:]

	return (p,v,a)


def test_astar():
	planner = Planner()
	planner.targetpos = np.array([2,0,0])
	planner.vel = np.array([0,1,0])
	planner.astar()
	plotPaths(planner.roughpath)


def test_calcHermite():
	path = np.array([[0,0,0],[1,0,0],[1.5,1,0],[1,1.5,0]])
	p, v, a = calcHermite(path)
	plotPaths(path)


def test_mpc():
	planner = Planner()
	planner.targetpos = np.array([2,0,0])
	planner.vel = np.array([0,1,0])
	planner.astar()
	planner.mpc()
	plotPaths((planner.roughpath,planner.path))


if __name__ == '__main__':
	
	test_calcHermite()




