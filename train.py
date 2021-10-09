import vast.algorithm as algorithm
import vast.domain as domain
import vast.experiments as experiments
import vast.data as data
import numpy
import sys
from settings import params

params["algorithm_name"] = sys.argv[1]
params["domain_name"] = sys.argv[2]
eta = numpy.finfo(numpy.float32).eps.item()
if len(sys.argv) > 3:
    eta = float(sys.argv[3])
env = domain.make(params)
params["nr_subteams"] = int(numpy.ceil(params["nr_agents"]*eta))
params["global_state_space"] = env.global_state_space.shape
params["local_observation_space"] = env.local_observation_space.shape

params["directory"] = "{}/{}-agents_domain-{}_subteams-{}_{}".\
    format(params["output_folder"],\
        params["nr_agents"],\
        params["domain_name"],\
        params["nr_subteams"],\
        params["algorithm_name"])
params["directory"] = data.mkdir_with_timestap(params["directory"])

controller = algorithm.make(params)
result = experiments.run_training(env, controller, params)
