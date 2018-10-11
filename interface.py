import pcl
import numpy as np



class Map:

	def __init__(self):

		self.terrain = pcl.PointCloud_PointXYZ() # a point cloud of the terrain in the world frame
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


	# The position of the object t seconds in the future
	def pos(self,t=0):

		if self.type == 0:
			return self.extrapolate(t)
		elif self.type == 1:
			return self.pathplan(t)


	# Predicts the position of the object given its initial pos, vel, accel
	def extrapolate(self,t):

		return self.pos + self.vel * t + self.accel * 0.5 * t**2


	def pathplan(self,t):
		
