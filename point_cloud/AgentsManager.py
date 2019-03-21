import sys
import pcl
import math
import argparse
from agent import Agent


class AgentManager:

	def __init__(self):
		self.agents = []

	def check_points(self, prev, after):
		pass

	def add_agent(self, agent):
		self.agents.append(agent)

	def remove_agent(self, agent):
		self.agents.remove(agent)

	def return_agents(self):
		result = []
		for a in self.agents:
			result.append(a.return_agent())
		return result

	def return_agents_in_r(self, pt, r):
		result = []
		for a in self.agents:
			dist = math.sqrt(pow(pt[0] - a.center[0], 2), pow(pt[1] - a.center[1], 2), pow(pt[2] - a.center[2], 2))
			if r + a.radius > dist:
				result.append(a)
		return result