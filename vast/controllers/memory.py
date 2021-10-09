import torch
import numpy
from vast.utils import assertEquals

"""
 Episode Memory. For efficiency all memory objects are already
 preallocated and overwritten on demand.
"""
class EpisodeMemory:
    def __init__(self, params, device):
        self.device = device
        self.episode_buffer = []
        self.episode_count = 0
        self.current_index = 0
        self.nr_agents = params["nr_agents"]
        self.nr_actions = params["nr_actions"]
        self.time_limit = params["time_limit"]
        self.state_dim = numpy.prod(params["global_state_space"])
        self.observation_dim = numpy.prod(params["local_observation_space"])
        self.nr_episodes = params["episode_capacity"]
        self.episode_time_limit = params["time_limit"]
        self.gamma = params["gamma"]
        self.max_time_steps = int(self.nr_episodes*self.episode_time_limit)
        self.state_buffer = torch.zeros((self.max_time_steps, self.state_dim), dtype=torch.float32, device=device)
        self.observation_buffer = torch.zeros((self.max_time_steps, self.nr_agents, self.observation_dim), dtype=torch.float32, device=device)
        self.joint_action_buffer = torch.zeros((self.max_time_steps, self.nr_agents), dtype=torch.long, device=device)
        self.return_buffer = torch.zeros((self.max_time_steps, self.nr_agents), dtype=torch.float32, device=device)
        self.reward_buffer = torch.zeros((self.max_time_steps, self.nr_agents), dtype=torch.float32, device=device)
        self.subteam_index_buffer = torch.zeros((self.max_time_steps, self.nr_agents), dtype=torch.long, device=device)
        self.dummy_ones = torch.ones(self.max_time_steps)
        self.dones = self.allocate_false_tensor()
        self.indices = self.allocate_false_tensor()

    def allocate_false_tensor(self):
        return self.dummy_ones != self.dummy_ones

    def save(self, new_transition):
        self.episode_buffer.append(new_transition)
        assert len(self.episode_buffer) <= self.episode_time_limit
        if new_transition["done"]:
            self.episode_count += 1
            return_value = 0
            for transition in reversed(self.episode_buffer):
                return_value = transition["reward"] + self.gamma*return_value
                transition["return"] = return_value
                index = self.current_index
                self.state_buffer[index].copy_(transition["state"])
                self.observation_buffer[index].copy_(transition["observations"])
                self.joint_action_buffer[index].copy_(transition["joint_action"])
                self.return_buffer[index].copy_(transition["return"])
                self.reward_buffer[index].copy_(transition["reward"])
                self.subteam_index_buffer[index].copy_(transition["subteam_indices"])
                self.dones[index] = transition["done"]
                self.indices[index] = True
                self.current_index += 1
            self.current_index = int(self.episode_count*self.time_limit)
            self.episode_buffer.clear()

    def get_training_data(self):
        states = self.state_buffer[self.indices]
        observations = self.observation_buffer[self.indices]
        joint_actions = self.joint_action_buffer[self.indices]
        returns = self.return_buffer[self.indices]
        rewards = self.reward_buffer[self.indices]
        dones = self.dones[self.indices]
        subteam_indices = self.subteam_index_buffer[self.indices]
        assertEquals(states.size(0), self.size())
        assertEquals(states.size(0), observations.size(0))
        assertEquals(states.size(0), joint_actions.size(0))
        assertEquals(states.size(0), returns.size(0))
        assertEquals(states.size(0), rewards.size(0))
        return states, observations, joint_actions, returns, rewards, dones, subteam_indices

    def is_full(self):
        return self.episode_count >= self.nr_episodes

    def clear(self):
        self.episode_count = 0
        self.current_index = 0
        self.state_buffer.fill_(0.0)
        self.observation_buffer.fill_(0.0)
        self.joint_action_buffer.fill_(0)
        self.return_buffer.fill_(0.0)
        self.reward_buffer.fill_(0.0)
        self.subteam_index_buffer.fill_(0.0)
        self.dones.fill_(False)
        self.indices.fill_(False)

    def size(self):
        return len([i for i in self.indices if i])
    