from vast.controllers.controller import Controller
from vast.utils import assertEquals, get_param_or_default
from torch.distributions import Categorical
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy

class ActorCriticNet(nn.Module):
    def __init__(self, input_shape, nr_actions, nr_hidden_units):
        super(ActorCriticNet, self).__init__()
        self.nr_input_features = numpy.prod(input_shape)
        self.nr_hidden_units = nr_hidden_units
        self.fc_net = nn.Sequential(
            nn.Linear(self.nr_input_features, self.nr_hidden_units),
            nn.ELU(),
            nn.Linear(self.nr_hidden_units, self.nr_hidden_units),
            nn.ELU()
        )
        self.action_head = nn.Linear(self.nr_hidden_units, nr_actions)
        self.value_head = nn.Linear(self.nr_hidden_units, 1)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.fc_net(x)
        return F.softmax(self.action_head(x), dim=-1), self.value_head(x)

class ActorCritic(Controller):

    def __init__(self, params):
        super(ActorCritic, self).__init__(params)
        self.no_ppo = get_param_or_default(params, "no_ppo", False)
        self.nr_update_iterations = get_param_or_default(params, "nr_update_iterations", 1) # Only required for PPO
        self.eps_clipping = get_param_or_default(params, "eps_clipping", 0.2) # Only required for PPO
        self.use_masking = get_param_or_default(params, "use_masking", True)
        self.critic_learner = params["critic_learner"]
        self.policy_net = ActorCriticNet(self.local_observation_space, self.nr_actions, params["actor_hidden_units"])
        self.parameters = self.policy_net.parameters()
        self.optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)

    def save_model_weights(self, path):
        super(ActorCritic, self).save_model_weights(path)
        if self.critic_learner is not None:
            self.critic_learner.save_model_weights(path)

    def load_model_weights(self, path):
        super(ActorCritic, self).load_model_weights(path)
        if self.critic_learner is not None:
            self.critic_learner.load_model_weights(path)
    
    def local_probs(self, observations, training_mode=True):
        observations = observations.view(-1, self.policy_net.nr_input_features)
        if self.use_masking:
            probs = super(ActorCritic, self).local_probs(observations, training_mode)
            indices = torch.tensor([o.sum() > self.eps for o in observations], dtype=torch.bool, device=self.device)
            probs[indices], _ = self.policy_net(observations[indices])
        else:
            probs, _ = self.policy_net(observations)
        return probs.detach()

    def centralized_update(self, states, joint_actions, observations, old_probs, dones, returns, subteam_indices):
        result = (0, 0)
        if self.critic_learner is not None:
            for _ in range(self.nr_update_iterations):
                result = self.critic_learner.update(states, joint_actions, observations, returns, dones, old_probs, subteam_indices)
        return result

    def get_values(self, states, observations, joint_actions):
        return self.critic_learner.get_global_values(states, observations, joint_actions)

    def policy_loss(self, advantage, probs, action, old_probs):
        m1 = Categorical(probs)
        if self.no_ppo:
            return -m1.log_prob(action)*advantage
        else:
            m2 = Categorical(old_probs.detach())
            ratio = torch.exp(m1.log_prob(action) - m2.log_prob(action))
            clipped_ratio = torch.clamp(ratio, 1-self.eps_clipping, 1+self.eps_clipping)
            surrogate_loss1 = ratio*advantage
            surrogate_loss2 = clipped_ratio*advantage
            return -torch.min(surrogate_loss1, surrogate_loss2)

    def action_probs_and_advantages(self, states, joint_actions, observations, returns):
        batch_size = states.size(0)
        new_batch_size = batch_size*self.nr_agents
        observations = observations.view(new_batch_size, -1)
        if self.use_masking:
            indices = torch.tensor([o.sum() > self.eps for o in observations], dtype=torch.bool, device=self.device)
            observations = observations[indices]
            action_probs = torch.zeros(new_batch_size, self.nr_actions)
            q_values = torch.zeros(new_batch_size, 1)
            action_probs[indices], q_values[indices] = self.policy_net(observations)
            if self.critic_learner is not None:
                q_values = torch.zeros(new_batch_size, self.nr_actions)
                q_values[indices] = self.critic_learner.get_local_values(states, observations, joint_actions)
        else:
            action_probs, q_values = self.policy_net(observations)
            if self.critic_learner is not None:
                q_values = self.critic_learner.get_local_values(states, observations, joint_actions)
        if self.critic_learner is not None:
            assertEquals(action_probs.size(), q_values.size())
            baselines_values = (action_probs.detach()*q_values.detach()).view(batch_size, self.nr_agents, self.nr_actions)
            baselines_values = baselines_values.sum(2)
            actions = joint_actions.view(new_batch_size, 1)
            q_values = q_values.gather(1, actions)
        else:
            baselines_values = q_values.detach()
        baselines_values = baselines_values.view(batch_size, self.nr_agents)
        assertEquals(baselines_values.size(), returns.size())
        advantages = returns - baselines_values
        return action_probs, advantages, q_values.squeeze()

    def local_update(self, states, joint_actions, observations, old_probs, dones, returns):
        batch_size = states.size(0)
        for _ in range(self.nr_update_iterations):
            action_probs, advantages, values = self.action_probs_and_advantages(\
                states, joint_actions, observations, returns)
            action_probs = action_probs.view(batch_size, self.nr_agents, self.nr_actions)
            advantages = advantages.view(batch_size, self.nr_agents, 1)
            values = values.view(batch_size, self.nr_agents)
            policy_losses = []
            value_losses = []
            for joint_action, old_joint_probs, joint_probs, joint_advantages, joint_R, joint_value in zip(\
                joint_actions, old_probs, action_probs, advantages, returns, values):
                for action, old_prob, probs, advantage, R, value in zip(\
                    joint_action, old_joint_probs, joint_probs, joint_advantages, joint_R, joint_value):
                    if probs.sum() > self.eps:
                        policy_losses.append(self.policy_loss(advantage.item(), probs, action, old_prob))
                        value_losses.append(F.mse_loss(R, value))
            loss = torch.stack(policy_losses).mean()
            if self.critic_learner is None:
                loss += torch.stack(value_losses).mean()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters, self.clip_norm)
            self.optimizer.step()
        return True
