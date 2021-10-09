import numpy

class RandomAssignment:

    def __init__(self, nr_clusters, nr_agents):
        self.nr_clusters = nr_clusters
        self.nr_agents = nr_agents
        self.labels_ = []

    def fit(self, agent_infos):
        self.labels_ = [numpy.random.randint(0, self.nr_clusters) for _ in agent_infos]