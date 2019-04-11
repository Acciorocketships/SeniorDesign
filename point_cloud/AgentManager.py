import sys
sys.path.append('/usr/local/Cellar/')
import pcl
import tools
import datetime, time

class AgentManager:

    def __init__(self):
        self.cloud = pcl.PointCloud()
        self.agents = dict()
        self.vectors = dict()
        self.timestamp = dict()

    def add_agent(self, agent_cloud):
        tup = tools.round_coords(tools.find_bounding_box_center(agent_cloud.to_list()))
        self.agents[tuple(tup)] = agent_cloud
        self.octree = pcl.OctreePointCloudSearch(0.1)
        self.timestamp[tuple(tup)] = datetime.datetime.now()
        new_ls = self.cloud.to_list()
        new_ls.append(tup)
        self.cloud = pcl.PointCloud(new_ls)

    def remove_agent_by_index(self, point_index):
        lst = self.cloud.to_list()
        point = tuple(lst[point_index])
        del lst[point_index]
        self.remove_helper(point, lst)
        self.cloud = pcl.PointCloud(lst)

    def remove_agent_by_point(self, point):
        lst = self.cloud.to_list()
        point = tools.round_coords(point)
        lst.remove(point)
        point = tuple(point)
        self.remove_helper(point, lst)
        self.cloud = pcl.PointCloud(lst)

    def drop_not_listed(self, save_set):
        big_set = set(self.agents.keys())
        big_set = big_set - save_set
        for p in big_set:
            self.remove_helper(p)
        self.cloud = pcl.PointCloud(list(save_set))

    def remove_helper(self, point):
        self.agents.pop(point, None)
        self.timestamp.pop(point, None)
        self.vectors.pop(point, None)

    def get_agent_by_index(self, point_index):
        lst = self.cloud.to_list()
        point = tools.round_coords(tuple(lst[point_index]))
        return self.agents[point]

    def get_agent_by_point(self, point):
        return self.agents[tuple(point)]

    def compare_agents(self, new_cloud, point):
        point = tools.round_coords(point)
        agent_cloud = self.agents[point]
        '''
        TODO use fancy statistics
        .
        .
        .
        self.vectors[point] = v
        '''
        pass

    def predict_agent(self, point):
        point = tools.round_coords(point)
        if point in self.vectors.keys():
            agent_vector = self.vectors[point]
            agent_cloud = self.agents[point]
            #TODO
            
        else:
            return None

    def update_agent_by_index(self, point_index, new_cloud):
        lst = self.cloud.to_list()
        point = tuple(lst[point_index])
        self.update_helper(point, new_cloud)

    def update_agent_by_point(self, point, new_cloud):
        point = tuple(point)
        self.update_helper(point, new_cloud)

    def update_helper(self, point, new_cloud):
        new_point = tools.find_bounding_box_center(new_cloud.to_list())
        prev_time = self.timestamp.pop(point, None)
        new_time = datetime.datetime.now()
        diff = time.mktime(new_time.timetuple()) - time.mktime(prev_time.timetuple())
        
        self.remove_agent_by_point(point)
        self.add_agent(new_cloud)

        self.timestamp[new_point] = new_time
        v = [new_point[0]-point[0], new_point[1]-point[1], new_point[2] - point[2]]
        if diff > 0:
            v0 = v[0]/diff
            v1 = v[1]/diff
            v2 = v[2]/diff
            self.vectors[new_point] = [v0, v1, v2]

    def concat_agent_by_index(self, point_index, new_cloud):
        lst = self.cloud.to_list()
        point = tuple(lst[point_index])
        lst = self.agents[point].to_list()
        lst = lst + new_cloud.to_list()
        self.agents[point] = pcl.PointCloud(lst)


    def concat_agent_by_point(self, point, new_cloud):
        lst = self.agents[point].to_list()
        lst = lst + new_cloud.to_list()
        self.agents[point] = pcl.PointCloud(lst)


    def radius_search_indices(self, pt_pos, r, num):
        self.octree.set_input_cloud(self.cloud)
        self.octree.add_points_from_input_cloud()
        indices, rads = self.octree.radius_search(pt_pos, r, num)
        return indices

    def radius_search_points(self, pt_pos, r, num):
        indices = self.radius_search_indices(pt_pos, r, num)
        l = self.cloud.to_list()
        coords = []
        for i in indices:
            coords.append(l[i])
        return coords

    def radius_search_agents(self, pt, r, num):
        ag_pts = self.radius_search_points(pt, r, num)
        cloud_list = []
        for a in ag_pts:
            cloud_list.append(self.agents[a])
        return cloud_list

    









    
