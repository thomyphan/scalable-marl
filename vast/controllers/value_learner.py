from vast.controllers.controller import Controller
from vast.utils import assertEquals
from os.path import join
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy

class QNet(nn.Module):
    def __init__(self, input_shape, nr_actions, nr_hidden_units):
        super(QNet, self).__init__()
        self.nr_input_features = numpy.prod(input_shape)
        self.nr_hidden_units = nr_hidden_units
        self.layer_1 = nn.Linear(self.nr_input_features, self.nr_hidden_units)
        self.layer_2 = nn.Linear(self.nr_hidden_units, self.nr_hidden_units)
        self.Q_head = nn.Linear(self.nr_hidden_units, nr_actions)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = F.elu(self.layer_1(x))
        hidden_representation = self.layer_2(x)
        x = F.elu(hidden_representation)
        return self.Q_head(x), hidden_representation

class QValueLearner(Controller):

    def __init__(self, params):
        super(QValueLearner, self).__init__(params)
        self.Q_net = QNet(self.local_observation_space, self.nr_actions, params["actor_hidden_units"])
        self.parameters = list(self.Q_net.parameters())
        self.optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)

    def save_model_weights(self, path):
        path = join(path, "Q_net.pth")
        torch.save(self.Q_net.state_dict(), path)

    def load_model_weights(self, path):
        path = join(path, "Q_net.pth")
        self.Q_net.load_state_dict(torch.load(path, map_location='cpu'))
        self.Q_net.eval()

    def get_local_values(self, states, observations, joint_actions):
        values, _ = self.Q_net(observations)
        return values

    def update(self, states, joint_actions, observations, returns, dones, old_probs, subteam_indices):
        batch_size = returns.size(0)
        new_batch_size = batch_size*self.nr_agents
        observations = observations.view(new_batch_size, -1)
        actions = joint_actions.view(new_batch_size, 1)
        Q_values, _ = self.Q_net(observations)
        Q_values = Q_values.gather(1, actions).squeeze()
        Q_values = Q_values.view(batch_size, self.nr_agents)
        return self.optimizer_update(Q_values, None, states, returns, subteam_indices)

    def optimizer_update(self, Q_values, Qmax_values, states, returns, subteam_indices):
        assertEquals(Q_values.size(), returns.size())
        loss = F.mse_loss(returns, Q_values)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters, self.clip_norm)
        self.optimizer.step()
        return (0, 0)

    def global_value(self, Q_values, states):
        return Q_values.mean(1)

    def global_value_of(self, Q_values, states):
        return self.global_value(Q_values, states)

class VDN(QValueLearner):

    def __init__(self, params):
        super(VDN, self).__init__(params)
        self.optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)

    def optimizer_update(self, Q_values, Qmax_values, states, returns, subteam_indices):
        Q_values = self.global_value(Q_values, states)
        returns = returns.mean(1)
        assertEquals(Q_values.size(), returns.size())
        loss = F.mse_loss(Q_values, returns)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters, self.clip_norm)
        self.optimizer.step()
        return (0, 0)

    def global_value(self, Q_values, states):
        return Q_values.sum(1)

class QMIXNet(nn.Module):
    def __init__(self, input_shape, nr_agents, mixing_hidden_size):
        super(QMIXNet, self).__init__()
        self.nr_agents = nr_agents
        self.mixing_hidden_size = mixing_hidden_size
        self.state_shape = numpy.prod(input_shape)
        self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_shape, mixing_hidden_size),
                                           nn.ELU(),
                                           nn.Linear(mixing_hidden_size, mixing_hidden_size * self.nr_agents))
        self.hyper_w_final = nn.Sequential(nn.Linear(self.state_shape, mixing_hidden_size),
                                           nn.ELU(),
                                           nn.Linear(mixing_hidden_size, mixing_hidden_size))
        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_shape, mixing_hidden_size)
        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_shape, mixing_hidden_size),
                               nn.ELU(),
                               nn.Linear(mixing_hidden_size, 1))

    def forward(self, global_state, Q_values):
        global_state = global_state.view(global_state.size(0), -1)
        w1 = torch.abs(self.hyper_w_1(global_state))
        b1 = self.hyper_b_1(global_state)
        w1 = w1.view(-1, self.nr_agents, self.mixing_hidden_size)
        b1 = b1.view(-1, 1, self.mixing_hidden_size)
        hidden = F.elu(torch.bmm(Q_values, w1) + b1)
        # Second layer
        w_final = torch.abs(self.hyper_w_final(global_state))
        w_final = w_final.view(-1, self.mixing_hidden_size, 1)
        # State-dependent bias
        v = self.V(global_state).view(-1, 1, 1)
        # Compute final output
        y = torch.bmm(hidden, w_final) + v
        return y.view(Q_values.size(0), -1, 1)

class QMIX(VDN):

    def __init__(self, params):
        super(QMIX, self).__init__(params)
        self.central_value_network = QMIXNet(\
            self.global_state_space, self.nr_agents, params["critic_hidden_units"])
        self.parameters += list(self.central_value_network.parameters())
        self.optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)

    def global_value(self, Q_values, states):
        Q_values = Q_values.view(-1, 1, self.nr_agents)
        return self.central_value_network(states, Q_values).squeeze()

    def save_model_weights(self, path):
        super(QMIX, self).save_model_weights(path)
        path = join(path, "central_value_network.pth")
        torch.save(self.central_value_network.state_dict(), path)

    def load_model_weights(self, path):
        super(QMIX, self).load_model_weights(path)
        path = join(path, "central_value_network.pth")
        self.central_value_network.load_state_dict(torch.load(path, map_location='cpu'))
        self.central_value_network.eval()

class QTRANBaseNet(nn.Module):
    def __init__(self, input_shape, nr_agents, nr_hidden_units):
        super(QTRANBaseNet, self).__init__()
        self.nr_input_features = numpy.prod(input_shape) + nr_agents
        self.nr_hidden_units = nr_hidden_units
        self.nr_agents = nr_agents
        self.fc_net = nn.Sequential(
            nn.Linear(self.nr_input_features, self.nr_hidden_units),
            nn.ELU(),
            nn.Linear(self.nr_hidden_units, self.nr_hidden_units),
            nn.ELU()
        )
        self.joint_action_head = nn.Linear(self.nr_hidden_units, 1)
        self.state_value_head = nn.Linear(self.nr_hidden_units, 1)

    def forward(self, global_states, Q_values):
        batch_size = global_states.size(0)
        global_states = global_states.view(batch_size, -1)
        Q_values = Q_values.view(batch_size, self.nr_agents)
        x = torch.cat([global_states, Q_values], dim=-1)
        x = self.fc_net(x)
        return self.joint_action_head(x), self.state_value_head(x)

"""
 Value Learner for Variable Agent Sub-Teams (VAST)
"""
class QTRAN(VDN):

    def __init__(self, params):
        super(QTRAN, self).__init__(params)
        self.central_value_network = QTRANBaseNet(\
            self.global_state_space, self.nr_agents, params["critic_hidden_units"])
        self.parameters += list(self.central_value_network.parameters())
        self.optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)

    def optimizer_update(self, Q_local_values_real, Q_local_values_max, states, returns, subteam_indices):
        batch_size = Q_local_values_real.size(0)
        dummy = torch.zeros(3*batch_size)
        real_detached_indices = dummy != dummy
        real_undetached_indices = dummy != dummy
        max_undetached_indices = dummy != dummy
        for counter in range(batch_size):
            real_detached_indices[counter] = True
            real_undetached_indices[counter+batch_size] = True
            max_undetached_indices[counter+2*batch_size] = True
        concat_states = torch.cat([states, states, states], dim=0)
        assertEquals(3*batch_size, concat_states.size(0))
        concat_Q_values = torch.cat([Q_local_values_real.detach(), Q_local_values_real, Q_local_values_max], dim=0)
        assertEquals(3*batch_size, concat_Q_values.size(0))
        Q_total, V_total = self.global_value(concat_Q_values, concat_states)
        Q = Q_total[real_detached_indices].squeeze()
        V = V_total[real_undetached_indices].squeeze()
        Q_max = Q_total[max_undetached_indices].detach().squeeze()
        V_max = V_total[max_undetached_indices].squeeze()
        Q_transformed_real = Q_local_values_real.sum(1)
        Q_transformed_max = Q_local_values_max.sum(1)
        returns = returns.sum(1)
        assertEquals(Q.size(), returns.size())
        assertEquals(V.size(), returns.size()) 
        assertEquals(Q_max.size(), returns.size())
        assertEquals(V_max.size(), returns.size()) 
        loss_value = (Q-returns)**2
        loss_constraint1 = (Q_transformed_max-Q_max+V_max**2)
        constraint2_term = (Q_transformed_real - Q.detach() + V).unsqueeze(1)
        assertEquals(batch_size, constraint2_term.size(0))
        assertEquals(1, constraint2_term.size(1))
        concat = torch.cat([constraint2_term, torch.zeros_like(constraint2_term)], dim=-1)
        assertEquals(batch_size, concat.size(0))
        assertEquals(2, concat.size(1))
        loss_constraint2 = (concat.min(1)[0])**2
        assertEquals(batch_size, loss_value.size(0))
        assertEquals(loss_value.size(), loss_constraint1.size())
        assertEquals(loss_value.size(), loss_constraint2.size())
        loss = (loss_value + loss_constraint1 + loss_constraint2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters, self.clip_norm)
        self.optimizer.step()
        return (0, 0)

    def update(self, states, joint_actions, observations, returns, dones, old_probs, subteam_indices):
        batch_size = returns.size(0)
        new_batch_size = batch_size*self.nr_agents
        observations = observations.view(new_batch_size, -1)
        actions = joint_actions.view(new_batch_size, 1)
        Q_local_values, _ = self.Q_net(observations)
        Q_local_values_real = Q_local_values.gather(1, actions).squeeze()
        Q_local_values_max = Q_local_values.max(1)[0].squeeze()
        Q_local_values_real = Q_local_values_real.view(batch_size, self.nr_agents)
        Q_local_values_max = Q_local_values_max.view(batch_size, self.nr_agents)
        return self.optimizer_update(Q_local_values_real, Q_local_values_max, states, returns, subteam_indices)

    def global_value(self, Q_values, states):
        Q_values = Q_values.view(-1, 1, self.nr_subteams)
        return self.central_value_network(states, Q_values)

    def global_value_of(self, Q_values, states):
        return self.global_value(Q_values, states)[0]

    def save_model_weights(self, path):
        super(QTRAN, self).save_model_weights(path)
        path = join(path, "central_value_network.pth")
        torch.save(self.central_value_network.state_dict(), path)

    def load_model_weights(self, path):
        super(QTRAN, self).load_model_weights(path)
        path = join(path, "central_value_network.pth")
        self.central_value_network.load_state_dict(torch.load(path, map_location='cpu'))
        self.central_value_network.eval()
