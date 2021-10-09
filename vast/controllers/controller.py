import numpy
import torch
from sklearn.cluster import KMeans
from os.path import join
from vast.utils import get_param_or_default
from vast.controllers.memory import EpisodeMemory
from vast.controllers.random_assignment import RandomAssignment
from torch.distributions import Categorical

"""
 Skeletal implementation of a (multi-agent) controller.
"""
class Controller:

    def __init__(self, params):
        self.max_time_steps = params["time_limit"]
        self.nr_agents = params["nr_agents"]
        self.clip_norm = get_param_or_default(params, "clip_norm", 1)
        self.nr_actions = params["nr_actions"]
        self.gamma = params["gamma"]
        self.learning_rate = params["learning_rate"]
        self.device = torch.device("cpu")
        self.global_state_space = params["global_state_space"]
        self.nr_subteams = get_param_or_default(params, "nr_subteams", 1)
        assert self.nr_subteams >= 1
        self.local_observation_space = params["local_observation_space"]
        self.assignment_strategy_type = params["assignment_strategy_type"]
        self.eps = numpy.finfo(numpy.float32).eps.item()
        self.actions = list(range(self.nr_actions))
        self.policy_net = None
        self.optimizer = None
        self.memory = EpisodeMemory(params, device=self.device)
        self.joint_action_buffer = torch.zeros((self.max_time_steps, self.nr_agents), dtype=torch.long)
        self.subteam_index_buffer = torch.zeros((self.max_time_steps+1, self.nr_agents), dtype=torch.long)
        self.workers = [self]
        self.agent_ids = list(range(self.nr_agents))
        self.assignment_strategy = None
        if self.assignment_strategy_type == "KMEANS":
            self.assignment_strategy = KMeans(n_clusters=params["nr_centroids"])
        if self.assignment_strategy_type == "RANDOM":
            self.assignment_strategy = RandomAssignment(self.nr_subteams, self.nr_agents)
        assert self.assignment_strategy is not None or self.assignment_strategy_type is None

    def group_agents(self, agent_infos, time_step):
        if self.assignment_strategy is None:
            for i in range(self.nr_agents):
                self.subteam_index_buffer[time_step][i] = i%self.nr_subteams
        else:
            indices = torch.tensor([info is not None for info in agent_infos], dtype=torch.bool)
            infos = [info for info in agent_infos if info is not None]
            if len(infos) > 0:
                max_nr_subteams = min(self.nr_subteams, len(infos))
                if self.assignment_strategy_type == "KMEANS":
                    self.assignment_strategy = KMeans(n_clusters=max_nr_subteams)
                if self.assignment_strategy_type == "RANDOM":
                    self.assignment_strategy = RandomAssignment(max_nr_subteams, self.nr_agents)
                self.assignment_strategy.fit(infos)
                self.subteam_index_buffer[time_step][indices] = torch.tensor(self.assignment_strategy.labels_, dtype=torch.long)
        return self.subteam_index_buffer[time_step]

    def save_model_weights(self, path):
        if self.policy_net is not None:
            path = join(path, "policy_weights.pth")
            torch.save(self.policy_net.state_dict(), path)

    def load_model_weights(self, path):
        if self.policy_net is not None:
            path = join(path, "policy_weights.pth")
            self.policy_net.load_state_dict(torch.load(path, map_location='cpu'))
            self.policy_net.eval()
    
    def policy(self, observations, time_step, training_mode=True):
        assert len(observations) == self.nr_agents,\
            "Expected {}, got {}".format(len(observations), self.nr_agents)
        probs = self.local_probs(observations, training_mode)
        for i, P in enumerate(probs):
            self.joint_action_buffer[time_step][i] = Categorical(P).sample().item()
        return self.joint_action_buffer[time_step]

    def local_probs(self, observations, training_mode=True):
        length = len(observations.size())-1
        batch_size = numpy.prod(observations.size()[:length])
        probs = torch.ones((batch_size, self.nr_actions), dtype=torch.float32)/self.nr_actions
        return probs

    def update(self, state, observations, joint_action, rewards, next_state, next_observations, done, subteam_indices):
        transition = {
                "state": state,
                "observations": observations,
                "joint_action": joint_action,
                "reward": rewards,
                "subteam_indices": subteam_indices,
                "next_observations": next_observations,
                "next_state": next_state,
                "done": done}
        self.memory.save(transition)
        result = {"centralized_update":(None, None), "local_update":None}
        if self.memory.is_full():
            s, obs, ja, returns, rewards, dones, subteam_indices = self.memory.get_training_data()
            batch_size = s.size(0)
            op = self.local_probs(obs)
            op = op.view(batch_size, self.nr_agents, self.nr_actions).detach()
            result = {}
            result["centralized_update"] = self.centralized_update(s, ja, obs, op, dones, returns, subteam_indices)
            result["local_update"] = self.local_update(s, ja, obs, op, dones, returns)
            self.memory.clear()
        return result

    def centralized_update(self, states, joint_actions, observations, old_probs, dones, returns, subteam_indices):
        return True

    def local_update(self, states, joint_action, observations, old_probs, dones, returns, agent_ids=None):
        return True
