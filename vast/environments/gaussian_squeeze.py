from vast.environments.environment import Environment
from gym import spaces
import numpy

class GaussianSqueezeEnvironment(Environment):

    def __init__(self, params):
        self.nr_agents = params["nr_agents"]
        self.global_state_space = spaces.Box(\
            -numpy.inf, numpy.inf, shape=(1,self.nr_agents))
        self.local_observation_space = spaces.Box(\
            -numpy.inf, numpy.inf, shape=(1, self.nr_agents))
        params["state_dim"] = numpy.prod(self.global_state_space.shape)
        params["observation_dim"] = numpy.prod(self.local_observation_space.shape)
        super(GaussianSqueezeEnvironment, self).__init__(params)
        self.mu = params["mu"]
        self.sigma = params["sigma"]
        self.objective_value = 0

    def get_agent_infos(self):
        return [(i, 0) for i in range(self.nr_agents)]

    def domain_value(self):
        return self.objective_value

    def perform_step(self, joint_action):
        rewards = super(GaussianSqueezeEnvironment, self).perform_step(joint_action)
        x = sum(joint_action)
        assert x >= 0 and x <= self.nr_agents*(self.nr_actions - 1), "'x' was {}".format(x)
        delta = (x-self.mu)/self.sigma
        self.objective_value = x*numpy.exp(-delta*delta)
        rewards.fill_(self.objective_value)
        rewards /= self.nr_agents
        return rewards

    def local_observation(self, agent_id=0):
        original = super(GaussianSqueezeEnvironment, self).local_observation(agent_id) 
        observation = original.view(-1)
        assert observation.size(0) == self.nr_agents
        observation[agent_id] = 1
        return original

    def global_state(self):
        original = super(GaussianSqueezeEnvironment, self).global_state()
        return original

    def render(self, viewer):
        return viewer

def make(params):
    params["nr_actions"] = 10
    params["time_limit"] = 1
    params["gamma"] = 1
    domain_name = params["domain_name"]
    params["mu"] = 400
    params["sigma"] = 200
    if domain_name == "GaussianSqueeze-200":
        params["nr_agents"] = 200
    if domain_name == "GaussianSqueeze-400":
        params["nr_agents"] = 400
    if domain_name == "GaussianSqueeze-800":
        params["nr_agents"] = 800
    return GaussianSqueezeEnvironment(params)

