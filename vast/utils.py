import numpy
import random

def assertEquals(first, second):
    assert first == second, "Expected {}, got {}".format(first, second)

def get_param_or_default(params, label, default_value=None):
    if label in params:
        return params[label]
    else:
        return default_value

def argmax(values):
    max_value = max(values)
    default_index = numpy.argmax(values)
    candidate_indices = []
    for i,value in enumerate(values):
        if value >= max_value:
            candidate_indices.append(i)
    if not candidate_indices:
        return default_index
    return random.choice(candidate_indices)