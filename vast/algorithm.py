import vast.controllers.controller as controller
import vast.controllers.actor_critic as actor_critic
import vast.controllers.value_learner as value_learner
import vast.controllers.vast as vast

def make_controller(params):
    algorithm_name = params["algorithm_name"]
    if algorithm_name == "IL":
        params["nr_subteams"] = 1
        params["critic_learner"] = None
        return actor_critic.ActorCritic(params)
    if algorithm_name == "QMIX":
        params["nr_subteams"] = 1
        params["critic_learner"] = value_learner.QMIX(params)
        return actor_critic.ActorCritic(params)
    if algorithm_name == "QTRAN":
        params["nr_subteams"] = 1
        params["critic_learner"] = value_learner.QTRAN(params)
        return actor_critic.ActorCritic(params)
    if algorithm_name.startswith("VAST"):
        assert "nr_subteams" in params, "{} requires 'nr_subteams'".format(algorithm_name)
        params["use_subteam_indices"] = False
        params["assignment_strategy_type"] = None
        if algorithm_name.endswith("-FIXED"):
            params["use_subteam_indices"] = True
            algorithm_name = algorithm_name.replace("-FIXED", "")
        if algorithm_name.endswith("-SPATIAL"):
            params["use_subteam_indices"] = True
            params["assignment_strategy_type"] = "KMEANS"
            params["nr_centroids"] = int(params["nr_subteams"]/2)
            algorithm_name = algorithm_name.replace("-SPATIAL", "")
        if algorithm_name.endswith("-RANDOM"):
            params["use_subteam_indices"] = True
            params["assignment_strategy_type"] = "RANDOM"
            algorithm_name = algorithm_name.replace("-RANDOM", "")
        nr_agents = params["nr_agents"]
        params["nr_agents"] = params["nr_subteams"]
        if algorithm_name.endswith("-VDN"):
            params["value_learner"] = value_learner.VDN(params)
        elif algorithm_name.endswith("-QMIX"):
            params["value_learner"] = value_learner.QMIX(params)
        elif algorithm_name.endswith("-QTRAN"):
            params["value_learner"] = value_learner.QTRAN(params)
        elif algorithm_name.endswith("-IL"):
            params["value_learner"] = value_learner.QValueLearner(params)
        else:
            raise ValueError("Unknown 'value_learner' for {}'".format(algorithm_name))
        params["nr_agents"] = nr_agents
        params["critic_learner"] = vast.VAST(params)
        return actor_critic.ActorCritic(params)

"""
 Creates a new (multi-agent) controller.
"""
def make(params):
    params["assignment_strategy_type"] = None
    algorithm_name = params["algorithm_name"]
    if algorithm_name == "Random":
        return controller.Controller(params)
    if "PPO" in algorithm_name:
        params["no_ppo"] = False
        params["algorithm_name"] = params["algorithm_name"].replace("PPO-", "")
        return make_controller(params)
    else:
        params["no_ppo"] = True
        params["nr_update_iterations"] = 1
        return make_controller(params)