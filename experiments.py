import numpy as np

from collections import Counter

import create_time_series_data
import create_visibility_graph

from scipy.stats import kstest, poisson
import matplotlib.pyplot as plt

import networkx_util
from probability_util import create_poisson_histogram


def run_single_experiment(n=1000, mode='manual', plot=None, title=None, xlabel=None, ylabel=None,
                          rolling_average_window=0, poisson_lambda=5):
    df = create_time_series_data.create_random_time_series(series_length=n,
                                                           rolling_average_window=rolling_average_window)
    if mode == 'manual':
        vertices_list, edges_list = create_visibility_graph.time_series_to_visibility_graph(df)
    else:
        vertices_list, edges_list = create_visibility_graph.time_series_to_visibility_graph_with_tsvg(df)

    graph = networkx_util.create_networkx_graph(vertices_list, edges_list)
    degree_sequence, degree_count = networkx_util.create_degree_distribution(graph, plot=plot, scale='linear',
                                                                             title=title, xlabel=xlabel, ylabel=ylabel,
                                                                             plot_poisson_lambda=5)
    if plot:
        networkx_util.create_degree_distribution(graph, plot=plot, scale='log', title=title, xlabel=xlabel,
                                                 ylabel=ylabel, plot_poisson_lambda=poisson_lambda)
    return graph, degree_sequence, degree_count

    # average_degree = 2 * len(edges_list) / len(vertices_list)
    # poisson_sequence, poisson_count = create_poisson_histogram(average_degree, n)


def baseline_experiment():
    run_single_experiment(n=10000, mode='auto', plot='matplotlib', title='Baseline Experiment, n=10000')


def compare_degree_distributions(n=1000, mode='manual', plot_statistic_values=False):
    """
    Experiment 1 - search the best parameter for the Poisson distribution to minimize difference from VG degree distribution

    """
    df = create_time_series_data.create_random_time_series(series_length=n)
    if mode == 'manual':
        vertices_list, edges_list = create_visibility_graph.time_series_to_visibility_graph(df)
    else:
        vertices_list, edges_list = create_visibility_graph.time_series_to_visibility_graph_with_tsvg(df)

    graph = networkx_util.create_networkx_graph(vertices_list, edges_list)
    degree_sequence, degree_count = networkx_util.create_degree_distribution(graph, plot=False, scale='linear')
    average_degree = 2 * len(graph.edges) / len(graph.nodes)
    output_ks_test_results = []
    for k in range(1, 20):
        poisson_sequence, poisson_count = create_poisson_histogram(k, n)
        ks_test_results = kstest(rvs=degree_sequence, cdf=poisson_sequence)
        test_statistic = ks_test_results[0]
        output_ks_test_results.append(test_statistic)
    best_parameter_fit = output_ks_test_results.index(min(output_ks_test_results)) + 1
    if plot_statistic_values:
        import matplotlib.pyplot as plt
        plt.scatter(x=range(1, 20), y=output_ks_test_results)
        plt.xlabel('k')
        plt.ylabel('KS Test Statistic')
        plt.title('KS Test Statistic vs. Poisson Parameter k')
        plt.show()
    return average_degree, best_parameter_fit


def first_experiment_poissonian_parameter_estimation():
    output_best_parameter = []
    output_average_degrees = []
    for experiment in range(250):
        average_degree, best_parameter = compare_degree_distributions(n=10000, mode='auto', plot_statistic_values=True)
        output_best_parameter.append(best_parameter)
        output_average_degrees.append(average_degree)
    average_degree = sum(output_average_degrees) / len(output_average_degrees)
    average_degree_std = sum([(x - average_degree) ** 2 for x in output_average_degrees]) / len(output_average_degrees)
    x = Counter(output_best_parameter)
    print(average_degree)
    print(average_degree_std)
    print(x)


def calculate_poissonian_fit_vs_sample_percentage(n=1000, mode='manual'):
    """
    Experiment 2 - calculate the Poissonian fit for fraction of sample. In each experiment we remove top p% value of
    degrees from both VG and Poissonian distribution and calculate the KS test statistic for the remaining values.
    """
    df = create_time_series_data.create_random_time_series(series_length=n)
    if mode == 'manual':
        vertices_list, edges_list = create_visibility_graph.time_series_to_visibility_graph(df)
    else:
        vertices_list, edges_list = create_visibility_graph.time_series_to_visibility_graph_with_tsvg(df)
    graph = networkx_util.create_networkx_graph(vertices_list, edges_list)
    degree_sequence, degree_count = networkx_util.create_degree_distribution(graph, plot=False, scale='linear')
    poisson_sequence, poisson_count = create_poisson_histogram(5, n)
    output_ks_test_results = []
    for percentage_to_drop in range(0, 50, 5):
        num_of_records_to_drop = int(len(degree_sequence) * percentage_to_drop / 100)
        degree_sequence_after_drop = degree_sequence[num_of_records_to_drop:]
        poisson_sequence_after_drop = poisson_sequence[num_of_records_to_drop:]
        new_degree_sequence = np.random.choice(degree_sequence_after_drop, size=int(0.5 * len(degree_sequence)),
                                               replace=False)
        new_poisson_sequence = np.random.choice(poisson_sequence_after_drop, size=int(0.5 * len(poisson_sequence)),
                                                replace=False)
        ks_test_results = kstest(rvs=new_degree_sequence, cdf=new_poisson_sequence)
        test_statistic = ks_test_results[0]
        output_ks_test_results.append(test_statistic)
    plt.scatter(x=range(100, 50, -5), y=output_ks_test_results)
    plt.xlabel('Percentage of Records')
    plt.ylabel('KS Test Statistic')
    plt.title('KS Test Statistic vs. Percentage of Records')
    plt.show()
    print(output_ks_test_results)


def analyze_degree_distribution_for_rolling_average_series(n=1000, mode='manual', rolling_average_window=10,
                                                           plot_statistic_values=False):
    """
    Experiment 3 - analyze the degree distribution for rolling average series
    """
    df = create_time_series_data.create_random_time_series(series_length=10000,
                                                           rolling_average_window=rolling_average_window)
    if mode == 'manual':
        vertices_list, edges_list = create_visibility_graph.time_series_to_visibility_graph(df)
    else:
        vertices_list, edges_list = create_visibility_graph.time_series_to_visibility_graph_with_tsvg(df)
    graph = networkx_util.create_networkx_graph(vertices_list, edges_list)
    degree_sequence, degree_count = networkx_util.create_degree_distribution(graph, plot=False, scale='linear')
    average_degree = 2 * len(graph.edges) / len(graph.nodes)
    output_ks_test_results = []
    for k in range(1, 20):
        poisson_sequence, poisson_count = create_poisson_histogram(k, n)
        ks_test_results = kstest(rvs=degree_sequence, cdf=poisson_sequence)
        test_statistic = ks_test_results[0]
        output_ks_test_results.append(test_statistic)
    best_parameter_fit = output_ks_test_results.index(min(output_ks_test_results)) + 1
    if plot_statistic_values:
        import matplotlib.pyplot as plt
        plt.scatter(x=range(1, 20), y=output_ks_test_results)
        plt.xlabel('k')
        plt.ylabel('KS Test Statistic')
        plt.title('KS Test Statistic vs. Poisson Parameter k (Rolling Average 5)')
        plt.show()
    return average_degree, best_parameter_fit


if __name__ == '__main__':
    analyze_degree_distribution_for_rolling_average_series(n=10000, mode='auto', rolling_average_window=5,
                                                           plot_statistic_values=True)
    run_single_experiment(n=10000, mode='auto', plot='matplotlib', rolling_average_window=2, poisson_lambda=6,
                          title="Degree Distribution vs. Poissonian(6) PMF\n Rolling Average 2")
