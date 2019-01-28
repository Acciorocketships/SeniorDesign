from gekko import GEKKO as solver
import numpy as np


class Planner:


	def __init__(self):

		self.targetpos = np.array([0,0,0])
		self.targetvel = np.array([0,0,0])

		self.pos = np.array([0,0,0])
		self.vel = np.array([0,0,0])
		self.rot = np.array([0,0,0])

		self.path = []


	def pathplan(self):
		
		pass