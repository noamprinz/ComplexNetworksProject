import numpy as np

from collections import Counter

from scipy.stats import poisson

import networkx as nx


def create_networkx_graph(vertices_list, edges_list):
    G = nx.Graph()
    G.add_nodes_from(vertices_list)
    G.add_edges_from(edges_list)
    return G


def create_degree_distribution(networkx_graph, plot='seaborn', scale='linear', title=None, xlabel=None, ylabel=None,
                               plot_poisson_lambda=None):
    degree_sequence = sorted([d for n, d in networkx_graph.degree()], reverse=True)
    degree_count = {}
    for degree in degree_sequence:
        if degree not in degree_count:
            degree_count[degree] = 0
        degree_count[degree] += 1
    degree_count = dict(sorted(degree_count.items()))
    if plot == 'seaborn':
        plot_degree_distribution(degree_sequence, scale=scale, mode='seaborn', title=title, xlabel=xlabel,
                                 ylabel=ylabel, plot_poisson_lambda=plot_poisson_lambda)
    elif plot == 'matplotlib':
        plot_degree_distribution(degree_sequence, scale=scale, mode='matplotlib', title=title, xlabel=xlabel,
                                 ylabel=ylabel, plot_poisson_lambda=plot_poisson_lambda)
    return degree_sequence, degree_count


def plot_degree_distribution(degree_sequence, scale='linear', mode='matplotlib', title=None, xlabel=None, ylabel=None,
                             plot_poisson_lambda=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    if mode == 'matplotlib':
        histogram = Counter(degree_sequence)
        x, y = zip(*histogram.items())
        plt.scatter(x, y)
    else:
        import seaborn as sns
        sns.set_style('whitegrid')
        sns.histplot(x=degree_sequence, ax=ax)
    title = title if title is not None else 'Degree distribution'
    xlabel = xlabel if xlabel is not None else 'Degree'
    ylabel = ylabel if ylabel is not None else 'Number of nodes'

    if scale == 'log':
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(xlabel + ' - logarithmic scale')
        plt.ylabel(ylabel + ' - logarithmic scale')
        plt.title(title + ' - logarithmic scale')
    else:
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
    if plot_poisson_lambda is not None:
        k = np.arange(0, 100)
        pmf = poisson.pmf(k, mu=plot_poisson_lambda)
        pmf = np.round(pmf, 5) * len(degree_sequence)
        plt.plot(k, pmf, 'o-', color='red', label='poisson pmf')
        print('')

    plt.show()
    plt.close()


if __name__ == '__main__':
    vertices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    edges = [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10)]
    vertices, distribution = create_degree_distribution(vertices, edges)
