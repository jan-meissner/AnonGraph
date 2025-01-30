from __future__ import annotations

import networkx as nx
import numpy as np

from anonymigraph.generation.role_infused_partition import (
    _generate_adjacency_and_features,
    _sample_graph,
    role_infused_partition,
)


def test_generate_adjacency_and_features_basic():
    c, n, p, k = 7, 11, 0.1, 8
    omega_role = np.random.rand(k, k)
    omega_role = (omega_role + omega_role.T) / 2

    adjacency_matrix, feature_matrix = _generate_adjacency_and_features(c, n, p, omega_role)

    assert adjacency_matrix.shape == (c * n * k, c * n * k), "Adjacency matrix shape is incorrect"
    assert feature_matrix.shape == (c * n * k, 2), "Feature matrix shape is incorrect"

    assert np.array_equal(adjacency_matrix, adjacency_matrix.T), "Adjacency matrix is not symmetric"

    for i in range(c * n * k):
        community, role = (i // n // k), i // n % k
        assert feature_matrix[i, 0] == community, f"Community assignment for node {i} is incorrect"
        assert feature_matrix[i, 1] == role, f"Role assignment for node {i} is incorrect"


def test_sample_graph_basic():
    n = 5
    prob_matrix = np.random.rand(n, n)
    prob_matrix = (prob_matrix + prob_matrix.T) / 2
    np.fill_diagonal(prob_matrix, 0)

    _sample_graph(prob_matrix, random_seed=0)

    assert np.all(np.isin(prob_matrix, [0, 1])), "Matrix contains non-binary values"
    assert np.array_equal(prob_matrix, prob_matrix.T), "Matrix is not symmetric after sampling"


def test_role_infused_partition_basic():
    c, n, p = 2, 5, 0.1
    omega_role = np.array([[0.3, 0.2], [0.2, 0.4]])

    G = role_infused_partition(c, n, p, omega_role)

    assert isinstance(G, nx.Graph), "Returned object is not a NetworkX graph"
    assert G.number_of_nodes() == c * n * 2, "Number of nodes in the graph is incorrect"


def test_role_infused_partition():
    c, n, p = 7, 10, 0.5
    omega_role = np.array([[0.8, 0.2], [0.2, 0.8]])

    graph = role_infused_partition(c, n, p, omega_role, random_seed=42)

    assert isinstance(graph, nx.Graph)
    assert len(graph.nodes) == c * n * 2
    assert len(graph.edges) > 0

    for node_id in graph.nodes:
        community_label, role_label = graph.nodes[node_id]["features"]
        assert 0 <= community_label < c
        assert 0 <= role_label < 2
