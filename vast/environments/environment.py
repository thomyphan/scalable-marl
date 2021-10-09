from vast.utils import get_param_or_default
import torch

"""
 Generic interface of a Partially Observable Multi-Agent Environment.
"""
class Environment:

    """
     Setup the environment. Should be called by subclasses.
    """
    def __init__(self, params):
        self.state_dim = params["state_dim"]
        self.observation_dim = params["observation_dim"]
        self.nr_agents = params["nr_agents"]
        self.nr_actions = params["nr_actions"]
        self.time_limit = params["time_limit"]
        self.gamma = params["gamma"]
        self.cooperative = get_param_or_default(params, "cooperative", True)
        self.actions = list(range(self.nr_actions))
        self.time_step = 0
        self.agent_ids = list(range(self.nr_agents))
        self.discounted_returns = torch.zeros(self.nr_agents, dtype=torch.float32)
        self.undiscounted_returns = torch.zeros(self.nr_agents, dtype=torch.float32)
        self.reward_buffer = torch.zeros((self.time_limit+1, self.nr_agents), dtype=torch.float32)
        self.state_buffer = torch.zeros((self.time_limit+1, self.state_dim), dtype=torch.float32)
        self.observation_tensor = torch.zeros(self.observation_dim, dtype=torch.float32)
        self.observation_buffer = torch.zeros((self.time_limit+1, self.nr_agents, self.observation_dim), dtype=torch.float32)

    def get_agent_infos(self):
        return [None for _ in range(self.nr_agents)]

    """
     Performs the joint action in order to change the environment.
     Returns the reward for each agent in a list sorted by agent ID.
    """
    def perform_step(self, joint_action):
        assert not self.is_done(), "Episode terminated at time step {}. Please, reset before calling 'step'."\
            .format(self.time_step)
        return self.reward_buffer[self.time_step].detach()

    """
     Indicates if an episode is done and the environments needs
     to be reset.
    """
    def is_done(self):
        return self.time_step >= self.time_limit

    """
     Performs a joint action to change the state of the environment.
     Returns the joint observation, the joint reward, a done flag,
     and other optional information (e.g., logged data).
     Note: The joint action must be a list ordered according to the agent ID!.
    """
    def step(self, joint_action):
        assert len(joint_action) == self.nr_agents, "Length of 'joint_action' is {}, expected {}"\
            .format(len(joint_action), self.nr_agents)
        assert not self.is_done(), "Episode terminated at time step {}. Please, reset before calling 'step'."\
            .format(self.time_step)
        rewards = self.perform_step(joint_action)
        assert len(rewards) == self.nr_agents, "Length of 'rewards' is {}, expected {}"\
            .format(len(rewards), self.nr_agents)
        observations = self.joint_observation()
        assert len(observations) == self.nr_agents, "Length of 'observations' is {}, expected {}"\
            .format(len(observations), self.nr_agents)
        self.time_step += 1
        if self.cooperative:
            total_reward = rewards.sum()
            rewards.fill_(total_reward)
        self.undiscounted_returns += rewards
        self.discounted_returns += (self.gamma**self.time_step)*rewards
        return observations, rewards, self.is_done(), {}

    """
     The global state of the environment. Only visible in fully
     observable domains.
    """
    def global_state(self):
        return self.state_buffer[self.time_step].detach()

    """
     The local observation for a specific agent. Only visible for
     the corresponding agent and private to others.
    """
    def local_observation(self, agent_id=0):
        self.observation_tensor.fill_(0.0)
        return self.observation_tensor

    """
     Returns the observations of all agents in a listed sorted by agent ids.
    """
    def joint_observation(self):
        for i in range(self.nr_agents):
            self.observation_buffer[self.time_step][i].copy_(self.local_observation(i))
        return self.observation_buffer[self.time_step].detach()

    """
     Returns a high-level value which is domain-specific.
    """
    def domain_value(self):
        return 0

    def render(self, viewer):
        return viewer

    """
     Re-Setup of the environment for a new episode.
    """
    def reset(self):
        self.time_step = 0
        self.discounted_returns.fill_(0.0)
        self.undiscounted_returns.fill_(0.0)
        self.reward_buffer.fill_(0.0)
        self.state_buffer.fill_(0.0)
        self.observation_buffer.fill_(0.0)
        return self.joint_observation()

    """
     Returns current state information of this environment
     as dictionary.
    """
    def state_summary(self):
        summary = {
            "nr_agents": self.nr_agents,
            "nr_actions": self.nr_actions,
            "time_step": self.time_step,
            "global_discounted_return": self.discounted_returns,
            "global_undiscounted_return": self.undiscounted_returns,
            "time_limit": self.time_limit,
            "gamma": self.gamma
        }
        return summary
        