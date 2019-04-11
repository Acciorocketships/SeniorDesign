import sys
sys.path.append('/usr/local/Cellar/')
import pcl
import math
from AgentManager import AgentManager
import tools
from random import randint

ag = AgentManager()

def test_add():
	for x in range(5):
		cl = pcl.PointCloud(tools.get_random_cloud())
		ag.add_agent(cl)
		temp = ag.get_agent_by_index(0)

def test_delete_by_index():
	for x in range(5):
		cl = pcl.PointCloud(tools.get_random_cloud())
		ag.add_agent(cl)

	print(len(ag.agents))

	for x in [4, 3, 2, 1, 0]:
		ag.remove_agent_by_index(x)

	print(len(ag.agents))

def test_delete_by_point():
	for x in range(5):
		cl = pcl.PointCloud(tools.get_random_cloud())
		ag.add_agent(cl)

	print(len(ag.agents))
	print(ag.agents)

	for x in [4, 3]:
		point = ag.cloud.to_list()[x]
		ag.remove_agent_by_point(point)

	print(len(ag.agents))
	print(ag.agents)

def test_update_by_index():
	for x in range(3):
		cl = pcl.PointCloud(tools.get_random_cloud())
		ag.add_agent(cl)
	print(ag.cloud.to_list())
	print("DO UPDATE")
	ag.update_agent_by_index(0, pcl.PointCloud(tools.get_random_cloud()))
	print(ag.cloud.to_list())

def test_update_by_point():
	for x in range(3):
		cl = pcl.PointCloud(tools.get_random_cloud())
		ag.add_agent(cl)
	print(ag.cloud.to_list())
	print("DO UPDATE")

	point = ag.cloud.to_list()[0]
	ag.update_agent_by_point(point, pcl.PointCloud(tools.get_random_cloud()))
	print(ag.cloud.to_list())

def test_concat_by_index():
	cl = pcl.PointCloud(tools.get_random_cloud())
	print(ag.agents)
	ag.add_agent(cl)
	ag.concat_agent_by_index(0, pcl.PointCloud(tools.get_random_cloud()))
	print(ag.agents)

def test_rad():
	for x in range(20):
		cl = pcl.PointCloud(tools.get_random_cloud(num=10))
		ag.add_agent(cl)

	result = ag.radius_search_points(ag.agents.keys()[10], 100, 5)
	print("ALL")
	print(ag.agents.keys())
	print("FOUND")
	print(result)

def test_drop():
	for x in range(20):
		cl = pcl.PointCloud(tools.get_random_cloud(num=10))
		ag.add_agent(cl)

	save = []
	for x in range(5):
		save.append(randint(0, 20))
	result = []
	ls = ag.agents.keys()
	print(ls)
	for x in save:
		print("X " + str(x))
		result.append(ls[x])

	print("TO DROP")
	print(result)
	print("BEFORE")
	print(ag.agents.keys())
	ag.drop_not_listed(set(result))
	print("AFTER")
	print(ag.agents.keys())






#test_add()
#test_delete_by_index()
#test_delete_by_point()
#test_update_by_point()
#test_concat_by_index()
#test_rad()
test_drop()






