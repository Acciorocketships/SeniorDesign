from gekko import GEKKO as solver
import numpy as np
from Map import *


class Planner:


	def __init__(self, map=None):

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
		pathdata = o.pathplan(destination=self.targetpos,T=5,dt=0.01,returnall=True)
		self.roughpath = pathdata[0]
		self.t = pathdata[1]


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
			self.t = np.linspace(0,5,21)
			self.roughpath = np.array(self.pos.reshape(3,1) + self.t*(self.targetpos-self.pos).reshape(3,1))

		s.time = self.t
		x = s.CV(value=self.roughpath[0,:], name="X")
		y = s.CV(value=self.roughpath[1,:], name="Y")
		z = s.CV(value=self.roughpath[2,:], name="Z")

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



		## Fix Target State ##

		# Fix Destination
		#s.fix(x,s.time.size-1,x[-1])
		#s.fix(y,s.time.size-1,y[-1])
		#s.fix(z,s.time.size-1,z[-1])

		# Fix Target Velocity
		# if self.targetvel != None:
		# 	s.fix(vx,s.time.size-1,vx[-1])
		# 	s.fix(vy,s.time.size-1,vy[-1])
		# 	s.fix(vz,s.time.size-1,vz[-1])

		# # Fix Target Rotation
		# if self.targetrot != None:
		# 	s.fix(phi,s.time.size-1,phi[-1])
		# 	s.fix(theta,s.time.size-1,theta[-1])
		# 	s.fix(psi,s.time.size-1,psi[-1])



		## Solve ##

		s.options.CV_TYPE = 1 # squared error
		s.options.IMODE = 6 # dynamic control
		s.options.SOLVER = 3
		s.options.OTOL = 1000
		s.options.RTOL = 1E-6
		s.solve(disp=True)

		import code; code.interact(local=locals())






if __name__ == '__main__':
	
	planner = Planner()
	planner.targetpos = np.array([2,0,0])
	planner.vel = np.array([0,1,0])
	planner.mpc()




