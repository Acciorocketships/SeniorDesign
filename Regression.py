from GaussND import *
import numpy as np

class Regressor:

	def __init__(self):
		self.restop = 10
		self.resstep = 0.1
		self.levels = 4
		self.map = [None, {}]


	def set(self,x,y,radius):
		gaussian = GaussND()
		mapcurr = self.map
		for level in range(self.levels):
			xdisc = self.tohash(self.discretize(x,self.stepsize(level)))
			

	def add(self,gaussian,radius):
		pos = gaussian.mean
		for corner in self.corners(pos,radius):
			box = self.discretize(corner, self.stepsize(0))
			if self.dist(pos,box) < (self.stepsize(0)*sqrt(len(pos)) + radius):
				self.map[self.tohash(box)][0] += gaussian
				# Handle if gaussian is None
				# do multiple levels


	def dist(self,pos1,pos2):
		return np.linalg.norm(pos1-pos2)
		

	# Finds the corners that circumscribe a position at a given level
	# Given a corner, this will return the centers of the adjacent cells
	def corners(self,middle,h):
		return self.cornershelper(middle, h, 0)

	def cornershelper(self,pos,h,i):
		if i == range(len(pos)):
			return [pos]
		pos1 = pos; pos1[i] += h
		pos2 = pos; pos2[i] -= h
		return self.cornershelper(pos1, h, i+1) + self.cornershelper(pos2, h, i+1)


	def stepsize(self,level):
		return self.restop * (self.resstep ** level)


	def discretize(self,val,step):
		if isinstance(val,np.ndarray):
			val = np.round(val / step) * step
		else:
			val = round(val / step) * step
		return val


	def tohash(self,val):
		if isinstance(val,np.ndarray):
			return tuple(np.reshape(arr,(-1,)))
		else:
			return val