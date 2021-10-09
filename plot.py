import vast.plotting as plotting
import matplotlib.pyplot as plot
import numpy
import sys
from os.path import join
from settings import params

DOMAIN_DATA = {
    "Warehouse-4": (4, "completed orders", 50, 3000, "QTRAN"),
    "Warehouse-8": (8, "completed orders", 50, 3000, "QTRAN"),
    "Warehouse-16": (16, "completed orders", 50, 3000, "QMIX"),
    "Battle-20": (20, "kill count", 100, 2000, "QMIX"),
    "Battle-40": (40, "kill count", 100, 2000, "QMIX"),
    "Battle-80": (80, "kill count", 100, 2000, "QMIX"),
    "GaussianSqueeze-200": (200, "performance", 1, 10000, "QTRAN"),
    "GaussianSqueeze-400": (400, "performance", 1, 10000, "QTRAN"),
    "GaussianSqueeze-800": (800, "performance", 1, 10000, "QTRAN"),
}

PREFIX = {
    "A" : "assignment_strategies_",
    "F" : "factorization_operators_",
    "S" : "state_of_the_art_comparison_",
    "D" : "division_diversity_"
}

domain_name = sys.argv[1]
evaluation_type = sys.argv[2]
show_legend = len(sys.argv) > 3
if domain_name.startswith("Warehouse"):
    domain_short = "warehouse_"
if domain_name.startswith("Battle"):
    domain_short = "battle_"
if domain_name.startswith("GaussianSqueeze"):
    domain_short = "gaussiansqueeze_"
nr_agents = None
y_label = None
plot.figure(figsize=(4, 2.5))
ax = plot.gca()
if domain_name in DOMAIN_DATA:
    nr_agents, y_label, episode_length, data_length, best_baseline = DOMAIN_DATA[domain_name]
filename_prefix = PREFIX[evaluation_type] + domain_short + str(nr_agents) + "_agents"
if show_legend:
    filename_prefix += "_legend"
filename = filename_prefix+".svg"
filename_png = filename_prefix+".png"
filename_pdf = filename_prefix+".pdf"
path = ""
assert domain_name is not None, "Unknown domain '{}'".format(domain_name)
if evaluation_type in ["A", "C"]:
    eta = 0.5
    if evaluation_type == "C":
        y_min = None
        y_max = None

episodes_per_epoch = 10
params["filter_size"] = 10
params["directory"] = params["output_folder"]
params["stats_label"] = "domain_values"
params["x_label"] = "steps"
params["y_label"] = y_label
params["data_length"] = data_length
params["x_axis_values"] = [i*episodes_per_epoch*episode_length for i in range(params["data_length"])]

approaches = [("b", 0.25, "VAST-QTRAN_"),("darkblue", 0.5, "VAST-QTRAN_"), ("darkorange", None, "QMIX"),("r", None, "QTRAN"), ("purple", None, "IL")]
if evaluation_type == "F":
    approaches = [("magenta", 0.5, "VAST-IL_"), ("c", 0.5, "VAST-VDN_"), ("g", 0.5, "VAST-QMIX_"), ("darkblue", 0.5, "VAST-QTRAN_")]
if evaluation_type == "A":
    approaches = [("c", 0.25, "VAST-QTRAN-RANDOM_"), ("purple", 0.25, "VAST-QTRAN-FIXED_"), ("k", 0.25, "VAST-QTRAN-SPATIAL_"), ("b", 0.25, "VAST-QTRAN_"), ("r", None, best_baseline)]
if evaluation_type == "D":
    approaches = [("b", 0.25, "VAST-QTRAN_"),("darkblue", 0.5, "VAST-QTRAN_")]
    params["stats_label"] = "sampled_subteams"
    params["y_label"] = "division diversity"

ALGORITHM_LABELS = {
    "VAST-QTRAN_": "VAST",
    "QMIX": "QMIX",
    "QTRAN": "QTRAN",
    "IL": "IL"
}

if evaluation_type == "F":
    ALGORITHM_LABELS = {
        "VAST-QTRAN_": "QTRAN",
        "VAST-QMIX_": "QMIX",
        "VAST-IL_": "IL",
        "VAST-VDN_": "VDN",
    }

if evaluation_type == "A":
    ALGORITHM_LABELS = {
        "VAST-QTRAN_": "MetaGrad",
        "VAST-QTRAN-FIXED_": "Fixed",
        "VAST-QTRAN-RANDOM_": "Random",
        "VAST-QTRAN-SPATIAL_": "Spatial",
        "QMIX": "Best Baseline",
        "QTRAN": "Best Baseline"
    }
plot_handles = []
for color, eta, algorithm in approaches:
    if eta is None:
        nr_subteams_ = 1
        params["label"] = ALGORITHM_LABELS[algorithm]
    else:
        nr_subteams_ = max(1,int(eta*nr_agents))
        if evaluation_type == "F":
            params["label"] = r"VAST $(\Psi_{"+ALGORITHM_LABELS[algorithm]+"})$"
        elif evaluation_type == "A":
            params["label"] = r"VAST $(\mathcal{X}_{"+ALGORITHM_LABELS[algorithm]+"})$"
        else:
            denominator = int(numpy.round(1.0/eta))
            params["label"] = r"VAST $(\eta=\frac{1}{"+str(denominator)+"})$"
    if domain_name == "GaussianSqueeze-200-400-200" and algorithm == "IA2C":
        params["nr_runs"] = 30
    params["data_prefix_pattern"] = "{}-agents_domain-{}_subteams-{}_{}".format(nr_agents, domain_name, nr_subteams_, algorithm)
    params["color"] = color
    handle = plotting.plot_runs(params)
    if handle is not None:
        plot_handles.append(handle[0])
evaluation_type_label = "State-of-the-Art Comparion"
if evaluation_type == "F":
    evaluation_type_label = "VFF Operators"
if evaluation_type == "A":
    evaluation_type_label = "Sub-Team Assignment Strategies"
if evaluation_type == "D":
    evaluation_type_label = "Division Diversity"
plot.grid()
max_steps = params["x_axis_values"][-1]
from decimal import Decimal
x_ticks = [0, max_steps/2, max_steps]
x_ticks_labels = [0, '%.2E' % Decimal(str(max_steps/2)), '%.2E' % Decimal(str(max_steps))]
plot.xticks(x_ticks, x_ticks_labels)

if show_legend:
    legend = plot.legend(facecolor="white", framealpha=0.8)
plot.tight_layout()
ax.grid(which='both', linestyle='--')
plotting.show()
plot.savefig(join(path, filename), bbox_inches='tight')
plot.savefig(join(path, filename_png), bbox_inches='tight')
plot.savefig(join(path, filename_pdf), bbox_inches='tight')