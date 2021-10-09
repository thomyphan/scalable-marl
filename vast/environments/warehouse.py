from vast.environments.environment import Environment
from vast.utils import get_param_or_default
from gym import spaces
import numpy
import random
import vast.environments.task_gen as task_gen
import vast.environments.warehouse_rendering as warehouse_rendering

NOOP = 0
MOVE_NORTH = 1
MOVE_SOUTH = 2
MOVE_WEST = 3
MOVE_EAST = 4
DROPOFF = 5

# Used to encode explicit graph edges, restricting possible moves
GRAPH_MAP = {
    "^": MOVE_NORTH,
    "v": MOVE_SOUTH,
    "<": MOVE_WEST,
    ">": MOVE_EAST
}

WAREHOUSE_ACTIONS = [NOOP, MOVE_NORTH, MOVE_SOUTH, MOVE_WEST, MOVE_EAST, DROPOFF]
MOVE_SUCCESSFUL = 0
MOVE_INTO_ROBOT = 1
MOVE_OUT_OF_BOUNDS = 2

class Robot:

    def __init__(self, taskgen_generate, id, env, initial_positions):
        self.id = id
        self.initial_positions = initial_positions
        self.position = random.choice(initial_positions)
        self.env = env
        if taskgen_generate is None:
            raise ValueError("No task generator provided!")
        self.taskgen_generate = taskgen_generate
        self.tasks = self.taskgen_generate()
        self.done = False

    def reset(self, reset_position):
        if reset_position:
            self.position = random.choice(self.initial_positions)
        self.tasks = self.taskgen_generate()
        self.done = False

    def current_tasks(self):
        if not self.done:
            return self.tasks[0]
        return []

    def move(self, action):
        if action == NOOP:
            return MOVE_SUCCESSFUL
        x, y = self.position
        new_position = (x, y)
        if not self.env.graph_yx or self.env.has_edge(action, x, y):
            if action == MOVE_NORTH and y + 1 < self.env.height:
                new_position = (x, y + 1)
            if action == MOVE_SOUTH and y - 1 >= 0:
                new_position = (x, y - 1)
            if action == MOVE_WEST and x - 1 >= 0:
                new_position = (x - 1, y)
            if action == MOVE_EAST and x + 1 < self.env.width:
                new_position = (x + 1, y)
        if new_position == self.position:
            return MOVE_OUT_OF_BOUNDS
        return self.set_position(new_position)

    def set_position(self, new_position):
        if self.already_occupied(new_position):
            return MOVE_INTO_ROBOT
        self.position = new_position
        return MOVE_SUCCESSFUL

    def relative_position(self, other_position):
        x_0, y_0 = self.position
        x, y = other_position
        dx = x - x_0
        dy = y - y_0
        return (dx, dy)

    def already_occupied(self, new_position):
        other_positions = [agent.position for agent in self.env.agents\
            if self.id != agent.id and not agent.done]
        potential_collision = new_position in other_positions
        unavailable_location = new_position not in self.env.locations
        return potential_collision or unavailable_location

    def dropoff(self, dropoff_type):
        if dropoff_type in self.current_tasks():
            self.tasks[0].remove(dropoff_type)
            if not self.tasks[0]:
                self.tasks.pop(0)
                if not self.tasks:
                    self.done = True
            return True
        else:
            return False

    def visible_positions(self):
        x0, y0 = self.position
        x_center = int(self.env.view_range/2)
        y_center = int(self.env.view_range/2)
        positions = [(x,y) for x in range(-x_center+x0, x_center+1+x0)\
            for y in range(-y_center+y0, y_center+1+y0)]
        return positions

    def nr_process_requests(self):
        return sum([len(bucket) for bucket in self.tasks])

class Location:

    def __init__(self, id, position, dropoff_type):
        self.id = id
        self.position = position
        self.dropoff_type = dropoff_type

class WarehouseEnvironment(Environment):

    def __init__(self, params):
        self.env_name = params["domain_name"]
        self.width = params["width"]
        self.height = params["height"]
        self.location_grid = params["location_grid"]
        self.graph_yx = params["graph_grid_yx"]
        self.view_range = params["view_range"]
        self.initial_positions = params["initial_positions"]
        self.nr_location_types = len(set(numpy.ndarray.flatten(self.location_grid)))
        dropoff_types_set = set(numpy.ndarray.flatten(self.location_grid))
        self.nr_location_types = len(dropoff_types_set)
        if -1 in dropoff_types_set:
            self.nr_location_types -= 1
        nr_channels_global = 5 * self.nr_location_types
        nr_channels_local = 5
        params["nr_actions"] = len(WAREHOUSE_ACTIONS)
        self.locations = {}
        self.location_types_to_positions = {}
        for location_type in range(self.nr_location_types):
            self.location_types_to_positions[location_type] = []
        location_count = 0
        for x, line in enumerate(self.location_grid):
            for y, location_type in enumerate(line):
                location_type = int(location_type)
                if location_type >= 0 and location_type < self.nr_location_types:
                    location_pos = (x, y)
                    location = Location(location_count, location_pos, location_type)
                    self.locations[location_pos] = location
                    location_count += 1
                    self.location_types_to_positions[location_type].append(location_pos)
        self.robot_reset = get_param_or_default(params, "robot_reset", True)
        self.nr_agents = params["nr_agents"]
        self.agents = [Robot(params["taskgen_generate"], i, self, self.initial_positions)\
            for i in range(self.nr_agents)]
        params["nr_agents"] = len(self.agents)
        self.global_state_space = spaces.Box(\
            -numpy.inf, numpy.inf, shape=(nr_channels_global, self.width, self.height))
        self.local_observation_space = spaces.Box(\
            -numpy.inf, numpy.inf, shape=(nr_channels_local, self.view_range, self.view_range))
        params["state_dim"] = numpy.prod(self.global_state_space.shape)
        params["observation_dim"] = numpy.prod(self.local_observation_space.shape)
        self.completion_count = 0
        super(WarehouseEnvironment, self).__init__(params)

    def has_edge(self, movement, x, y):
        edges_str = self.graph_yx[y][x]
        edges = [GRAPH_MAP[char] for char in edges_str]  # convert from string to movement int
        return movement in edges

    def get_agent_infos(self):
        return [agent.position for agent in self.agents]

    def get_non_agent_infos(self):
        return [location.position for location in self.locations.values()]

    def global_failure_rate(self):
        counts = sum([machine.get_attempts() for machine in self.agents])
        failures = sum([machine.get_failures() for machine in self.agents])
        if counts < 1:
            return 0
        return failures/counts

    def perform_step(self, joint_action):
        rewards = super(WarehouseEnvironment, self).perform_step(joint_action)
        agents_and_actions = [(action, agent) for action, agent in zip(joint_action, self.agents) if not agent.done]
        random.shuffle(agents_and_actions)
        for action, agent in agents_and_actions:
            if not agent.done:
                if action == DROPOFF:
                    if agent.position in self.locations:
                        dropoff_type = self.locations[agent.position].dropoff_type
                        successful = agent.dropoff(dropoff_type)
                        if successful:
                            rewards[agent.id] += 1
                else:
                    result = agent.move(action)
                    if result == MOVE_INTO_ROBOT:
                        rewards[agent.id] -= 0.5
            if agent.done:
                self.completion_count += 1
                if self.robot_reset:
                    agent.reset(False)
            else:
                rewards[agent.id] -= 0.01
        return rewards

    def is_done(self):
        time_limit_reached = super(WarehouseEnvironment, self).is_done()
        if self.robot_reset:
            return time_limit_reached
        all_agents_done = not [agent for agent in self.agents if not agent.done]
        return time_limit_reached or all_agents_done

    def local_observation(self, agent_id=0):
        original = super(WarehouseEnvironment, self).local_observation(agent_id)
        agent = self.agents[agent_id]
        if agent.done:
            return original
        observation = original.view(self.local_observation_space.shape)
        x0, y0 = agent.position
        x_center = int(self.view_range/2)
        y_center = int(self.view_range/2)
        visible_positions = agent.visible_positions()
        observation[0][x_center][y_center] = 1
        for visible_position in visible_positions:
            if visible_position not in self.locations:
                dx, dy = agent.relative_position(visible_position)
                observation[1][x_center+dx][y_center+dy] += 1
        for other_agent in self.agents:
            if other_agent.position in visible_positions and other_agent.id != agent.id:
                dx, dy = agent.relative_position(other_agent.position)
                observation[2][x_center+dx][y_center+dy] += 1
        for i, bucket in enumerate(agent.tasks):
            for task in bucket:
                positions = self.location_types_to_positions[task]
                for position in positions:
                    if position in visible_positions:
                        dx, dy = agent.relative_position(position)
                        observation[i+3][x_center+dx][y_center+dy] += 1
        return original

    def global_state(self):
        original = super(WarehouseEnvironment, self).global_state()
        observation = original.view(self.global_state_space.shape)
        observation[0, :, :] = -1
        for m in self.locations.keys():
            x_m, y_m = m
            observation[0][x_m][y_m] = self.locations[m].dropoff_type
        for agent in [a for a in self.agents if not a.done]:
            x, y = agent.position
            dropoff_type = self.locations[agent.position].dropoff_type
            required_location = dropoff_type in agent.current_tasks()
            base_index = 1
            if required_location:
                observation[base_index][x][y] += 1
            else:
                observation[base_index+1][x][y] += 1
            current_buckets = len(agent.tasks)
            base_index += 2
            completed_buckets = 2 - current_buckets
            for i in range(current_buckets):
                for p in range(self.nr_location_types):
                    if p in agent.tasks[i]:
                        index = base_index+p+self.nr_location_types*i
                        offset = completed_buckets*self.nr_location_types
                        observation[index+offset][x][y] += 1  # Having task assigned
        return original

    def domain_value(self):
        return self.completion_count

    def reset(self):
        self.completion_count = 0
        self.max_step_count = 0
        for agent in self.agents:
            agent.reset(True)
        return super(WarehouseEnvironment, self).reset()

    def state_summary(self):
        summary = super(WarehouseEnvironment, self).state_summary()
        return summary

    def render(self, render_stub, metadata=None):
        warehouse_rendering.render(self, render_stub)
        return render_stub

WAREHOUSE_LAYOUTS = {
    "Warehouse-4": (4, [(1,0), (1,4)], numpy.swapaxes(numpy.array(
            [[-1, +0, -1],
             [ 1, +2,  1],
             [+4, -1, +5],
             [ 1, +3,  1],
             [-1, +0, -1]]),
            0, 1),
            [
                [   "",   "v",   ""],
                [ "v>", "<^>", "<v"],
                [ "^v",    "", "^v"],
                [ "^>", "<v>", "<^"],
                [   "",   "^",   ""]
            ][::-1]),
    "Warehouse-8": (8, [(1,0), (1,4), (3,0), (3,4)], numpy.swapaxes(numpy.array(
            [[ 1, +0,  1, +0,  1],
             [ 1, +2,  1, +3,  1],
             [+4, -1, +5, -1, +4],
             [ 1, +3,  1, +2,  1],
             [ 1, +0,  1, +0,  1]]),
            0, 1),
            [
                [ "v>", "<v>", "<v>", "<v>", "<v"],
                ["^v>", "<^>","<v^>", "<^>","<^v"],
                [ "^v",    "",  "v^",    "", "^v"],
                ["^v>", "<v>","<v^>", "<^>","<^v"],
                [ "^>", "<^>", "<^>", "<^>", "<^"]
            ][::-1]),
    "Warehouse-16": (16, [(0,3), (0,5), (12,3), (12,5), (3, 0), (5, 0), (7, 0), (9, 0), (3, 8), (5, 8), (7, 8), (9, 8)], numpy.swapaxes(numpy.array(
            [[-1, -1, -1, +0, -1, +0, -1, +0, -1, +0, -1, -1, -1],
             [-1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1],
             [-1,  1,  1, +2,  1, +3,  1, +2,  1, +3,  1,  1, -1],
             [+0,  1, +4, -1, +5, -1, +4, -1, +5, -1, +4,  1, +0],
             [-1,  1,  1, +3,  1, +2,  1, +3,  1, +2,  1,  1, -1],
             [+0,  1, +4, -1, +5, -1, +4, -1, +5, -1, +4,  1, +0],
             [-1,  1,  1, +3,  1, +2,  1, +3,  1, +2,  1,  1, -1],
             [-1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1],
             [-1, -1, -1, +0, -1, +0, -1, +0, -1, +0, -1, -1, -1]]),
            0, 1),
            [
               [   "",    "",    "",   "v",    "",   "v",    "",   "v",    "",   "v",    "",    "",    ""],
               [   "",  "v>", "<v>","<^v>", "<v>","<^v>", "<v>","<^v>", "<v>","<^v>", "<v>",  "<v",    ""],
               [   "", "v^>","<^v>", "<^>","<^v>", "<^>","<^v>", "<^>","<^v>", "<^>","<^v>", "<v^",    ""],
               [  ">","<^v>", "<^v",    "",  "^v",    "",  "^v",    "",  "^v",    "", "^v>","<v^>",   "<"],
               [   "", "v^>","<^v>",  "<>","<^v>",  "<>","<^v>",  "<>","<^v>",  "<>","<^v>", "<v^",    ""],
               [  ">","<^v>", "<^v",    "",  "^v",    "",  "^v",    "",  "^v",    "", "^v>","<v^>",   "<"],
               [   "", "v^>","<^v>", "<v>","<^v>", "<v>","<^v>", "<v>","<^v>", "<v>","<^v>", "<v^",    ""],
               [   "",  "^>", "<^>","<^v>", "<^>","<^v>", "<^>","<^v>", "<^>","<^v>", "<^>",  "<^",    ""],
               [   "",    "",    "",   "^",    "",   "^",    "",   "^",    "",   "^",    "",    "",    ""] 
            ][::-1])
}

def make(params):
    domain_name = params["domain_name"]
    params["nr_agents"], params["initial_positions"],params["location_grid"], \
        params["graph_grid_yx"] = WAREHOUSE_LAYOUTS[domain_name]
    params["view_range"] = 5
    params["time_limit"] = get_param_or_default(params, "time_limit", 50)
    params["gamma"] = 0.95
    params["bucket_size"] = 5
    params["width"], params["height"] = params["location_grid"].shape
    params["taskgen_generate"] = task_gen.TaskgenRangeFixedLasttask(2,
            bucket_size=params["bucket_size"],  # only for random tasks - last bucket has only one task
            process_type_low=2,
            process_type_high=numpy.max(params["location_grid"]),
            process_type_last=0  # End task: 0
            ).generate
    return WarehouseEnvironment(params)

