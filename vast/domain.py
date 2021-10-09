import vast.environments.battle as battle
import vast.environments.warehouse as warehouse
import vast.environments.gaussian_squeeze as gaussian_squeeze

"""
 Creates a new environment.
"""
def make(params):
    domain_name = params["domain_name"]
    if domain_name.startswith("Battle-"):
        params["nr_epochs"] = 2000
        return battle.make(params)
    if domain_name.startswith("Warehouse-"):
        params["nr_epochs"] = 3000
        return warehouse.make(params)
    if domain_name.startswith("GaussianSqueeze-"):
        params["nr_epochs"] = 10000
        return gaussian_squeeze.make(params)
    raise ValueError("Unknown domain '{}'".format(domain_name))