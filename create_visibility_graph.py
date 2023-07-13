import os

import numpy as np
import pandas as pd

from ts2vg import NaturalVG


def time_series_to_visibility_graph(time_series: pd.Series):
    """
    Creates a visibility graph from a time series
    :param time_series: a time series
    :return: a visibility graph
    """
    vertices_list = list(time_series.index)
    edges_list = []
    for vertex in vertices_list:
        potential_visible_vertices = vertices_list[vertices_list.index(vertex) + 1:]
        vertex_value = time_series[vertex]
        for candidate in potential_visible_vertices:
            candidate_value = time_series[candidate]
            coefficients = np.polyfit([vertex,candidate], [vertex_value,candidate_value], 1)
            visibility_line = np.poly1d(coefficients)
            neighbours_to_check = vertices_list[vertices_list.index(vertex) + 1:vertices_list.index(candidate)]
            neighbours_values = time_series[neighbours_to_check]
            neighbours_visibility = visibility_line(neighbours_to_check)
            if np.all(neighbours_visibility > neighbours_values):
                edges_list.append((vertex, candidate))
                print(f'edge {(vertex,candidate)} added')
    return vertices_list, edges_list


def time_series_to_visibility_graph_with_tsvg(time_series):
    time_series_values = list(time_series.values)
    vg = NaturalVG()
    vg.build(time_series_values)
    vertices_list = list(time_series.index)
    edges_list = sorted(list(vg.edges))
    return vertices_list, edges_list



if __name__ == '__main__':
    from create_time_series_data import create_random_time_series
    df = create_random_time_series(series_length=10000)
    sec_vertices_list, sec_edges_list = time_series_to_visibility_graph_with_tsvg(df)
    print(len(sec_edges_list))



