import numpy as np


def create_poisson_histogram(poisson_lambda, n):
    poisson_sample = np.random.poisson(lam=poisson_lambda, size=n)
    degree_count = {}
    for degree in poisson_sample:
        if degree not in degree_count:
            degree_count[degree] = 0
        degree_count[degree] += 1
    degree_count = dict(sorted(degree_count.items()))
    return sorted(poisson_sample, reverse=True), degree_count
