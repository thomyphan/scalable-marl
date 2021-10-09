from vast.controllers.value_learner import VDN
from vast.utils import assertEquals
from torch.distributions import Categorical
from os.path import join
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy

class VASTNet(nn.Module):
    def __init__(self, nr_input_features, nr_subteams, nr_hidden_units = 128):
        super(VASTNet, self).__init__()
        self.nr_input_features = nr_input_features
        self.nr_hidden_units = nr_hidden_units
        self.fc_net = nn.Sequential(
            nn.Linear(self.nr_input_features, self.nr_hidden_units),
            nn.ELU(),
            nn.Linear(self.nr_hidden_units, self.nr_hidden_units),
            nn.ELU()
        )
        self.action_head = nn.Linear(self.nr_hidden_units, nr_subteams)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.fc_net(x)
        return F.softmax(self.action_head(x), dim=-1)

"""
 Value Learner for Variable Agent Sub-Teams (VAST)
"""
class VAST(VDN):

    def __init__(self, params):
        super(VAST, self).__init__(params)
        self.value_learner = params["value_learner"]
        self.Q_net = self.value_learner.Q_net
        assertEquals(self.nr_subteams, self.value_learner.nr_agents)
        self.use_subteam_indices = params["use_subteam_indices"]
        self.vast_input_space = numpy.prod(self.local_observation_space)+numpy.prod(self.global_state_space)
        self.assignment_learner = VASTNet(self.vast_input_space, self.nr_subteams)
        self.parameters = self.assignment_learner.parameters()
        self.assignment_optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        self.episodes_per_epoch = params["episodes_per_epoch"]

    def save_model_weights(self, path):
        self.value_learner.save_model_weights(path)
        path = join(path, "vast_network.pth")
        torch.save(self.assignment_learner.state_dict(), path)

    def load_model_weights(self, path):
        self.value_learner.load_model_weights(path)
        path = join(path, "vast_network.pth")
        self.assignment_learner.load_state_dict(torch.load(path, map_location='cpu'))
        self.assignment_learner.eval()

    def get_subteam_q_values(self, states, observations, joint_actions, subteam_indices, greedy=False):
        batch_size = states.size(0)
        new_batch_size = batch_size*self.nr_agents
        observations = observations.view(new_batch_size, -1)
        Q_local_values, representations = self.Q_net(observations)
        representations = representations.detach().view(batch_size, self.nr_agents, -1)
        vast_inputs = []
        for state, obs in zip(states, observations.view(batch_size, self.nr_agents, -1)):
            for observation in obs:
                vast_input = torch.cat([state.view(-1), observation.view(-1)], dim=-1)
                assertEquals(vast_input.size(0), self.vast_input_space)
                vast_inputs.append(vast_input)
        actions = joint_actions.view(new_batch_size, 1)
        if not self.use_subteam_indices: # Sample sub-team assignments
            assignments = self.assignment_learner(torch.stack(vast_inputs))
            assertEquals(self.nr_subteams, assignments.size(1))
            assignment_dist = Categorical(assignments)
            if greedy:
                subteam_ids = assignments.max(1)[1]
            else:
                subteam_ids = assignment_dist.sample().detach()
            assignment_log_probs = assignment_dist.log_prob(subteam_ids)
            subteam_ids = subteam_ids.view(batch_size, self.nr_agents)
            max_assignment_ids = assignments.max(1)[1].detach()
        else: # Otherwise use assignments of alternative assignment strategy
            assignment_log_probs = None
            subteam_ids = subteam_indices
            max_assignment_ids = subteam_indices
        Q_local_values_real = Q_local_values.gather(1, actions).squeeze()
        Q_local_values_max = Q_local_values.max(1)[0].squeeze()
        Q_local_values_real = Q_local_values_real.view(batch_size, self.nr_agents)
        Q_local_values_max = Q_local_values_max.view(batch_size, self.nr_agents)
        sub_Q_values = []
        sub_Qmax_values = []
        for Q, Qmax, s_indices in zip(Q_local_values_real, Q_local_values_max, subteam_ids): # Compute sub-team values
            sub_Q = []
            sub_Qmax = []
            for i in range(self.nr_subteams):
                subteam_member_indices = \
                    torch.tensor([i == index for index in s_indices], dtype=torch.bool, device=self.device)
                sub_Q.append(Q[subteam_member_indices].sum())
                sub_Qmax.append(Qmax[subteam_member_indices].sum())
            sub_Qmax_values.append(torch.stack(sub_Qmax))
            sub_Q_values.append(torch.stack(sub_Q))
        sub_Q_values = torch.stack(sub_Q_values).view(batch_size, self.nr_subteams)
        sub_Qmax_values = torch.stack(sub_Qmax_values).view(batch_size, self.nr_subteams)
        return sub_Q_values, sub_Qmax_values, assignment_log_probs, subteam_ids, max_assignment_ids

    def update(self, states, joint_actions, observations, returns, dones, old_probs, subteam_indices):
        batch_size = returns.size(0)
        new_batch_size = batch_size*self.nr_agents
        sub_Q_values, sub_Qmax_values, log_probs, sampled_ids, max_ids = \
            self.get_subteam_q_values(states, observations, joint_actions, subteam_indices)
        returns_shaped = torch.zeros(batch_size, self.nr_subteams)
        for R, R_shaped in zip(returns, returns_shaped):
            R_shaped.fill_(R.mean())
        self.value_learner.optimizer_update(sub_Q_values, sub_Qmax_values, states, returns_shaped, subteam_indices)
        if not self.use_subteam_indices:
            Q_values, _ = self.Q_net(observations.view(new_batch_size, -1))
            Q_values = Q_values.detach().view(new_batch_size, self.nr_actions)
            old_probs = old_probs.detach().view(new_batch_size, self.nr_actions)
            returns = returns.view(new_batch_size)
            advantages = [R - sum(Q*P).item() for Q, P, R in zip(Q_values, old_probs, returns)]
            advantages = torch.stack(advantages).view(new_batch_size)
            self.update_assignment_learner(advantages, log_probs)
        sampled_ids = sampled_ids.view(batch_size, self.nr_agents)
        max_ids = max_ids.view(batch_size, self.nr_agents)
        sampled_ids_unique = []
        max_ids_unique = []
        for sample_id, max_id in zip(sampled_ids, max_ids):
            sampled_ids_unique.append(sample_id.unique().detach().numpy().tolist())
            max_ids_unique.append(max_id.unique().detach().numpy().tolist())
        return sampled_ids_unique, max_ids_unique

    def update_assignment_learner(self, advantages, log_probs):
        policy_losses = [-log_prob*advantage for log_prob, advantage in zip(log_probs, advantages)]
        loss = torch.stack(policy_losses).mean()
        self.assignment_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters, self.clip_norm)
        self.assignment_optimizer.step()

    def global_value_of(self, Q_values, states):
        return self.value_learner.global_value_of(Q_values, states)
