import sys
sys.path.append('/usr/local/Cellar/')
import pcl
import agent
import tools

class AgentManager:

    def __init__(self):
        self.cloud = pcl.PointCloud()
        self.agents = dict()
        self.vectors = dict()

    def add_agent(self, agent_cloud):
        tup = tools.find_bounding_box_center(agent_cloud)
        self.agents[tup] = agent_cloud
        self.cloud = self.cloud + pcl.PointCloud(tup)

    def remove_agent_by_index(self, point_index):
        lst = self.cloud.to_list()
        point = lst[point_index]
        del lst[point_index]
        self.agents.pop(point, None)
        self.cloud = pcl.PointCloud(lst)

    def remove_agent_by_point(self, point):
        lst = self.cloud.to_list()
        del lst[point_index]
        self.agents.pop(point, None)
        self.cloud = pcl.PointCloud(lst)

    def get_agent_cloud(self, point):
        return self.agents[point]

    def compare_agents(self, new_cloud, point):
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
        if point in self.vectors.keys():
            agent_vector = self.vectors[point]
            agent_cloud = self.agents[point]
            
            '''
            TODO fancy math
            '''
        else:
            return None

    def update_agent(self, point, new_cloud):
        new_point = tools.find_bounding_box_center(new_cloud)
        self.agents.pop(point, None)
        self.agents[new_point] = new_cloud

    def concat_agent(self, point, new_cloud):
        existing_cloud = self.agents[point]
        joint_cloud = existing_cloud + new_cloud
        self.update_agent(point, joint_cloud)

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

    def radius_search_agents(self, pt, r):
        ag_pts = self.radius_search_points(pt, r, 100)
        cloud_list = []
        for a in ag_pts:
            cloud_list.append(self.agents[a])
        return cloud_list

    

    
