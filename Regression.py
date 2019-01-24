from GaussND import *
import numpy as np
from math import log

class Regressor:

	def __init__(self):
		self.restop = 100
		self.resstep = 0.1
		self.levels = 5
		self.map = {}
		self.dim = 3


	def set(self,pos,val,radius=1):
		self.dim = pos.size
		currval = self.eval(pos)
		cov = radius * np.eye(pos.size)
		gaussian = GaussND(numN=(pos,cov))
		newval = gaussian[pos]
		gaussian *= (val-currval) / newval
		self.add(gaussian)
			

	def add(self,gaussian):
		pos = gaussian.numN[0].mean
		radius = gaussian.numN[0].cov[0][0]
		deepestlevel = min(round(log(1/radius * self.restop) / log(1/self.resstep))-1, self.levels-1)
		# Go to the largest level at which the gaussian spills into surrounding boxes
		curr = self.map
		for level in range(0,deepestlevel):
			key = self.tohash(self.discretize(pos, self.stepsize(level)))
			if key not in curr:
				curr[key] = ([],{})
			curr = curr[key][1]
		# Add the gaussian to the middle and surrounding boxes
		discpos = self.discretize(pos, self.stepsize(deepestlevel))
		for corner in self.corners(discpos, self.stepsize(deepestlevel)):
			key = self.tohash(corner)
			if key not in curr:
				curr[key] = ([],{})
			curr[key][0].append(gaussian)


	def eval(self,pos):
		if not isinstance(pos,np.ndarray):
			pos = np.array(pos)
		if len(pos.shape) == 2:
			outputs = []
			for i in range(pos.shape[1]):
				outputs.append(self.eval(pos[:,i]))
			outputs = np.array(outputs)
			return outputs
		gaussians = []
		curr = self.map
		for level in range(0,self.levels):
			key = self.tohash(self.discretize(pos, self.stepsize(level)))
			if key not in curr:
				break
			gaussians = gaussians + curr[key][0]
			curr = curr[key][1]
		return sum(map(lambda f: f[pos], gaussians))


	# Finds the corners that circumscribe a position at a given level
	# Given a corner, this will return the centers of the adjacent cells
	def corners(self,middle,h):
		return self.cornershelper(middle, h, 0)
	def cornershelper(self,pos,h,i):
		if i == len(pos):
			return [pos]
		pos0 = pos.copy()
		pos1 = pos.copy()
		pos2 = pos.copy()
		pos1[i] += h
		pos2[i] -= h
		return self.cornershelper(pos0, h, i+1) + self.cornershelper(pos1, h, i+1) + self.cornershelper(pos2, h, i+1)


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
			return tuple(np.reshape(val,(-1,)))
		else:
			return val


	def __getitem__(self,pos):
		return self.eval(pos)


	def __add__(self,term):	
		pos = term[0]
		val = term[1]
		currval = self.eval(pos)
		if len(term) > 2:
			self.set(pos,currval+val,radius=term[2])
		else:
			self.set(pos,currval+val)
		return self


	def __radd__(self,term):
		return self.__add__(term)


	def plot(self,lim=[[-5,5],[-5,5],[-5,5]]):
		if self.dim == 3:
			from mayavi import mlab
			# Evaluate
			xi,yi,zi = np.mgrid[lim[0][0]:lim[0][1]:50j, lim[1][0]:lim[1][1]:50j, lim[2][0]:lim[2][1]:50j]
			coords = np.vstack([item.ravel() for item in [xi, yi, zi]])
			density = self.eval(coords).reshape(xi.shape)
			# Plot scatter with mayavi
			figure = mlab.figure('DensityPlot',fgcolor=(0.0,0.0,0.0),bgcolor=(0.85,0.85,0.85),size=(600, 480))
			grid = mlab.pipeline.scalar_field(xi, yi, zi, density)
			minval = 0
			maxval = density.max()
			mlab.pipeline.volume(grid, vmin=minval, vmax=minval + .5*(maxval-minval))
			mlab.axes(xlabel="x1",ylabel="x2",zlabel="x3")
			mlab.show()
		elif self.dim == 2:
			from matplotlib import cm
			import matplotlib.pyplot as plt
			from mpl_toolkits.mplot3d import Axes3D
			fig = plt.figure()
			ax = fig.add_axes([0,0,1,1], projection='3d')
			# Evaluate
			xi,yi = np.mgrid[lim[0][0]:lim[0][1]:100j, lim[1][0]:lim[1][1]:100j]
			coords = np.vstack([item.ravel() for item in [xi, yi]])
			density = self.eval(coords).reshape(xi.shape)
			# Plot surface with matplotlib
			surf = ax.plot_surface(xi, yi, density, cmap=cm.coolwarm, linewidth=0, antialiased=False, rcount=100, ccount=100)
			ax.view_init(90, -90)
			plt.xlabel("$x_1$")
			plt.ylabel("$x_2$")
			fig.colorbar(surf, shrink=0.2, aspect=5)
			plt.show()
		elif self.dim == 1:
			import matplotlib.pyplot as plt
			fig = plt.figure()
			# Evaluate
			xi = np.mgrid[lim[0][0]:lim[0][1]:100j]
			density = self.eval(np.array([xi])).reshape(xi.shape)
			# Plot surface with matplotlib
			func = plt.plot(xi, density)
			plt.xlabel("$x$")
			plt.show()


def main():
	r = Regressor()
	p1 = np.array([2,0,0])
	p2 = np.array([-3,0,0])
	r = r + (p1,1,1)
	r = r + (p2,2,0.5)
	r.plot()


if __name__ == '__main__':
	main()
