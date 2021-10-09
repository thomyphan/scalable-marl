from vast.environments.environment import Environment
from gym import spaces
import vast.environments.battle_rendering as battle_rendering
import numpy
import random

AGENT_CHANNEL = 0
OPPONENT_CHANNEL = 1
OBSTACLE_CHANNEL = 2
CHANNELS = [AGENT_CHANNEL, OPPONENT_CHANNEL, OBSTACLE_CHANNEL]

NOOP = 0
MOVE_NORTH = 1
MOVE_SOUTH = 2
MOVE_WEST = 3
MOVE_EAST = 4
ATTACK = 5

BATTLE_ACTIONS = [NOOP, MOVE_NORTH, MOVE_SOUTH, MOVE_WEST, MOVE_EAST, ATTACK]


class Agent:

    def __init__(self, i, env, max_health_points, initial_positions):
        self.initial_positions = initial_positions
        self.id = i
        self.max_health_points = max_health_points
        self.health_points = self.max_health_points
        self.position = None
        self.env = env
        self.step_recovery = 0.1
        self.reset()

    def is_alive(self):
        assert self.health_points >= 0
        return self.health_points > 0

    def is_dead(self):
        return not self.is_alive()

    def take_hit(self):
        self.health_points = max(0, self.health_points - 1)

    def recover(self):
        self.health_points = min(self.max_health_points, self.health_points+self.step_recovery)

    def attack(self, opponent_map):
        if self.position in opponent_map:
            opponents = [o for o in opponent_map[self.position] if o.is_alive()]
            if len(opponents) > 0:
                opponent = random.choice(opponents)
                opponent.take_hit()
                return 1, opponent.is_dead()
            else:
                return 0, False
        return -0.1, False

    def reset(self):
        positions = [pos for pos in self.initial_positions]
        self.position = random.choice(positions)
        self.health_points = self.max_health_points

    def visible_positions(self):
        x0, y0 = self.position
        x_center = int(self.env.view_range/2)
        y_center = int(self.env.view_range/2)
        positions = [(x,y) for x in range(-x_center+x0, x_center+1+x0)\
            for y in range(-y_center+y0, y_center+1+y0)]
        return positions

    def act(self, action, opponent_map):
        x, y = self.position
        new_position = (x, y)
        if action == MOVE_NORTH and y + 1 < self.env.height:
            new_position = (x, y + 1)
        if action == MOVE_SOUTH and y - 1 >= 0:
            new_position = (x, y - 1)
        if action == MOVE_WEST and x - 1 >= 0:
            new_position = (x - 1, y)
        if action == MOVE_EAST and x + 1 < self.env.width:
            new_position = (x + 1, y)
        self.position = new_position
        if action == ATTACK:
            return self.attack(opponent_map)
        return 0, False
    
    def relative_position(self, other_position):
        x_0, y_0 = self.position
        x, y = other_position
        dx = x - x_0
        dy = y - y_0
        return (dx, dy)

    def state_summary(self):
        return {
            "id": self.id,
            "position_x": self.position[0],
            "position_y": self.position[1]
        }

class BattleEnvironment(Environment):

    def __init__(self, params):
        self.width = params["width"]
        self.height = params["height"]
        self.view_range = params["view_range"]
        global_channels = len(CHANNELS) - 1
        local_channels = len(CHANNELS)+1
        self.global_state_space = spaces.Box(\
            -numpy.inf, numpy.inf, shape=(global_channels, self.width, self.height))
        self.local_observation_space = spaces.Box(\
            -numpy.inf, numpy.inf, shape=(local_channels, self.view_range, self.view_range))
        params["state_dim"] = numpy.prod(self.global_state_space.shape)
        params["observation_dim"] = numpy.prod(self.local_observation_space.shape)
        super(BattleEnvironment, self).__init__(params)
        self.agent_kills = 0
        self.opponent_kills = 0
        self.max_health_points = params["max_health_points"]
        self.agents = [Agent(i, self, self.max_health_points, [pos])\
            for i, pos in enumerate(params["agent_initial_positions"])]
        self.opponents = [Agent(i, self, params["max_health_points"], [pos])\
            for i, pos in enumerate(params["opponent_initial_positions"])]
        assert len(self.agents) == self.nr_agents

    def get_agent_infos(self):
        infos = []
        for agent in self.agents:
            if agent.is_dead():
                infos.append(None)
            else:
                infos.append(agent.position)
        return infos

    def get_non_agent_infos(self):
        infos = []
        for agent in self.opponents:
            if agent.is_dead():
                infos.append(None)
            else:
                infos.append(agent.position)
        return infos

    def survival_rate(self):
        return self.agent_kills*1.0/self.nr_agents

    def kill_rate(self):
        return self.opponent_kills*1.0/self.nr_agents

    def domain_value(self):
        return self.opponent_kills - self.agent_kills

    def is_done(self):
        time_limit_reached = super(BattleEnvironment, self).is_done()
        return time_limit_reached or self.opponent_kills >= self.nr_agents or self.agent_kills >= self.nr_agents

    def create_map(self, agents):
        agent_map = {}
        for agent in agents:
            if agent.is_alive():
                if agent.position not in agent_map:
                    agent_map[agent.position] = []
                agent_map[agent.position].append(agent)
        return agent_map

    def perform_step(self, joint_action):
        assert len(joint_action) == len(self.agents)
        rewards = super(BattleEnvironment, self).perform_step(joint_action)
        action_agents = list(zip(joint_action, self.agents))
        random.shuffle(action_agents)
        total_reward = 0
        opponent_map = self.create_map(self.opponents)
        for action, agent in action_agents:
            if agent.is_alive():
                agent.recover()
                reward, killed = agent.act(action, opponent_map)
                if killed:
                    self.opponent_kills += 1
                total_reward += reward
        agent_map = self.create_map(self.agents)
        for opponent in self.opponents:
            if opponent.is_alive():
                opponent.recover()
                if opponent.position in agent_map:
                    reward, agent_kill = opponent.act(ATTACK, agent_map)
                else:
                    if opponent.position[0] < self.width/2:
                        action_candidates = []
                    else:
                        action_candidates = [MOVE_WEST]
                    visible_positions = opponent.visible_positions()
                    for pos in visible_positions:
                        if pos in agent_map:
                            dx, dy = opponent.relative_position(pos)
                            if dx > 0:
                                action_candidates.append(MOVE_EAST)
                            if dx < 0:
                                action_candidates.append(MOVE_WEST)
                            if dy > 0:
                                action_candidates.append(MOVE_NORTH)
                            if dy < 0:
                                action_candidates.append(MOVE_SOUTH)
                    if len(action_candidates) == 0:
                        action_candidates += BATTLE_ACTIONS
                    reward, agent_kill = opponent.act(random.choice(action_candidates), agent_map)
                reward = max(0, reward)
                if reward > 0:
                    total_reward -= 0.5
                if agent_kill:
                    self.agent_kills += 1
        rewards.fill_(total_reward)
        rewards /= self.nr_agents
        return rewards

    def local_observation(self, agent_id=0):
        original = super(BattleEnvironment, self).local_observation(agent_id) 
        observation = original.view(self.local_observation_space.shape)
        agent = self.agents[agent_id]
        if agent.is_dead():
            return original
        x_center = int(self.view_range/2)
        y_center = int(self.view_range/2)
        observation[0][x_center][y_center] = agent.health_points*1.0/self.max_health_points
        visible_positions = agent.visible_positions()
        agent_map = self.create_map(self.agents)
        opponent_map = self.create_map(self.opponents)
        for visible_position in visible_positions:
            x, y = visible_position
            out_of_bounds = x < 0 or y < 0 or x >= self.width or y >= self.height
            if out_of_bounds:
                dx, dy = agent.relative_position(visible_position)
                observation[OBSTACLE_CHANNEL+1][x_center+dx][y_center+dy] += 1
            if visible_position in agent_map:
                other_agents = agent_map[visible_position]
                nr_other_agents = len(other_agents)
                for other_agent in other_agents:
                    assert nr_other_agents > 0
                    if other_agent.id != agent.id and other_agent.is_alive():
                        dx, dy = agent.relative_position(other_agent.position)
                        observation[AGENT_CHANNEL+1][x_center+dx][y_center+dy] += other_agent.health_points*1.0/self.max_health_points
            if visible_position in opponent_map:
                opponents = opponent_map[visible_position]
                nr_opponents = len(opponents)
                for opponent in opponents:
                    assert nr_opponents > 0
                    if opponent.is_alive():
                        dx, dy = agent.relative_position(opponent.position)
                        observation[OPPONENT_CHANNEL+1][x_center+dx][y_center+dy] += opponent.health_points*1.0/self.max_health_points
        return original

    def global_state(self):
        original = super(BattleEnvironment, self).global_state()
        state = original.view(self.global_state_space.shape)
        for agent in self.agents:
            if agent.is_alive():
                x, y = agent.position
                state[AGENT_CHANNEL][x][y] += agent.health_points*1.0/self.max_health_points
        for opponent in self.opponents:
            if opponent.is_alive():
                x, y = opponent.position
                state[OPPONENT_CHANNEL][x][y] += opponent.health_points*1.0/self.max_health_points
        return original

    def reset(self):
        self.agent_kills = 0
        self.opponent_kills = 0
        for agent in self.agents:
            agent.reset()
        for opponent in self.opponents:
            opponent.reset()
        return super(BattleEnvironment, self).reset()

    def state_summary(self):
        summary = super(BattleEnvironment, self).state_summary()
        summary["agents"] = [agent.state_summary() for agent in self.agents]
        summary["opponent"] = [opponent.state_summary() for opponent in self.opponents]
        return summary

    def render(self, viewer):
        viewer = battle_rendering.render(self, viewer)
        return viewer

BATTLE_LAYOUTS = {
    "Battle-20": """
        . . . . . . . . . .
        . . . . . . . . . .
        . A A . . . . O O .
        A A A A . . O O O O
        A A A A . . O O O O
        A A A A . . O O O O
        A A A A . . O O O O
        . A A . . . . O O .
        . . . . . . . . . .
        . . . . . . . . . .
    """,
    "Battle-40": """
        . . . . . . . . . . . . . .
        . . . . . . . . . . . . . .
        . . . . . . . . . . . . . .
        . A A A A . . . . O O O O .
        . A A A A . . . . O O O O .
        A A A A A A . . O O O O O O
        A A A A A A . . O O O O O O
        A A A A A A . . O O O O O O
        A A A A A A . . O O O O O O
        . A A A A . . . . O O O O .
        . A A A A . . . . O O O O .
        . . . . . . . . . . . . . .
        . . . . . . . . . . . . . .
        . . . . . . . . . . . . . .
    """,
    "Battle-80": """
        . . . . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . . . . .
        . . A A A A . . . . . . O O O O .
        . A A A A A A . . . . O O O O O O .
        . A A A A A A . . . . O O O O O O .
        A A A A A A A A . . O O O O O O O O
        A A A A A A A A . . O O O O O O O O
        A A A A A A A A . . O O O O O O O O
        A A A A A A A A . . O O O O O O O O
        A A A A A A A A . . O O O O O O O O
        A A A A A A A A . . O O O O O O O O
        . A A A A A A . . . . O O O O O O .
        . A A A A A A . . . . O O O O O O .
        . . A A A A . . . . . . O O O O . .
        . . . . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . . . . .
    """
}

def make(params):
    domain_name = params["domain_name"]
    params["nr_actions"] = len(BATTLE_ACTIONS)
    layout = BATTLE_LAYOUTS[domain_name]
    params["height"] = 0
    params["width"] = 0
    params["time_limit"] = 100
    params["gamma"] = 0.99
    params["max_health_points"] = 3
    params["obstacles"] = []
    params["view_range"] = 7
    params["agent_initial_positions"] = []
    params["opponent_initial_positions"] = []
    for _,line in enumerate(layout.splitlines()):
        splitted_line = line.strip().split()
        if splitted_line:
            for x,cell in enumerate(splitted_line):
                position = (x,params["height"])
                if cell == '#':
                    params["obstacles"].append(position)
                if cell == 'A':
                    params["agent_initial_positions"].append(position)
                if cell == 'O':
                    params["opponent_initial_positions"].append(position)
                params["width"] = x
            params["height"] += 1
            params["width"] += 1
    params["nr_agents"] = len(params["agent_initial_positions"])
    return BattleEnvironment(params)

