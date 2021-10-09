from scipy import stats
from vast.utils import get_param_or_default
from vast.data import list_directories, list_files, load_json
import matplotlib.pyplot as plot
import numpy

def bootstrap(data, n_boot=10000, ci=95):
    boot_dist = []
    for _ in range(int(n_boot)):
        resampler = numpy.random.randint(0, data.shape[0], data.shape[0])
        sample = data.take(resampler, axis=0)
        boot_dist.append(numpy.mean(sample, axis=0))
    b = numpy.array(boot_dist)
    s1 = numpy.apply_along_axis(stats.scoreatpercentile, 0, b, 50.-ci/2.)
    s2 = numpy.apply_along_axis(stats.scoreatpercentile, 0, b, 50.+ci/2.)
    return (s1,s2)

def tsplot(data, params, alpha=0.12, **kw):
    data = numpy.array(data)
    default_x = list(range(data.shape[1]))
    x = get_param_or_default(params, "x_axis_values", default_x)[:len(default_x)]
    assert len(x) == len(default_x), "Expected x-axis length of {} but got {}".format(len(default_x), len(x))
    est = numpy.mean(data, axis=0)
    ci = get_param_or_default(params, "ci", 95)
    cis = bootstrap(data, ci=ci)
    color = get_param_or_default(params, "color", None)
    label = params["label"]
    x_label = params["x_label"]
    y_label = params["y_label"]
    plot.title(get_param_or_default(params, "title"))
    if color is not None:
        plot.fill_between(x,cis[0],cis[1],alpha=alpha, color=color, **kw)
        handle = plot.plot(x,est,label=label,color=color,**kw)
    else:
        plot.fill_between(x,cis[0],cis[1],alpha=alpha, **kw)
        handle = plot.plot(x,est,label=label, **kw)
    plot.xlabel(x_label)
    plot.ylabel(y_label)
    plot.margins(x=0)
    return handle

def plot_runs(params):
    data = []
    directory_count = 0
    label = params["label"]
    filter_size = params["filter_size"]
    path = params["directory"]
    filename = get_param_or_default(params, "filename", "results.json")
    data_prefix_pattern = params["data_prefix_pattern"]
    stats_label = params["stats_label"]
    data_length = get_param_or_default(params, "data_length", None)
    nr_runs = get_param_or_default(params, "nr_runs", None)
    for directory in list_directories(path, lambda x: x.startswith(data_prefix_pattern)):
        if nr_runs is None or directory_count < nr_runs:
            for json_file in list_files(directory, lambda x: x == filename):
                json_data = load_json(json_file)
                if stats_label in ["domain_values"]:
                    return_values = json_data[stats_label]
                elif stats_label == "sampled_subteams":
                    subteams_progress = json_data[stats_label]
                    return_values = []
                    for subteams in subteams_progress:
                        subteams_dict = {}
                        step_size = 10
                        if "GaussianSqueeze" in data_prefix_pattern:
                            step_size = 1 # Take all episodes of epoch, since GSD is only single-step
                        subteams = [subteams[i] for i in range(0, len(subteams), step_size)]
                        for assignments in subteams:
                            assignments.sort()
                            subteams_dict[tuple(assignments)] = None
                        return_values.append(len(subteams_dict))
                else:
                    return_values = numpy.mean(json_data[stats_label], axis=0)
                if data_length is not None:
                    return_values = return_values[:data_length]
                kernel = numpy.ones(filter_size)/filter_size
                return_values = numpy.convolve(return_values, kernel, mode='valid')
                data.append(return_values)
                directory_count += 1
    if len(data) > 0:
        return tsplot(data, params)
    return None

def show(showgrid=True):
    if showgrid:
        plot.grid()
    plot.show()
