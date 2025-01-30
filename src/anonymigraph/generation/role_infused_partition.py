import networkx as nx
import numpy as np
from numba import njit, prange


@njit(parallel=True)
def _generate_adjacency_and_features(c: int, n: int, p: float, omega_role: np.ndarray):
    """generates Role-Infused Partition Model's probabilistic adjacency matrix and features"""
    k = omega_role.shape[0]
    total_nodes = c * n * k
    adjacency_matrix = np.zeros((total_nodes, total_nodes))
    feature_matrix = np.zeros((total_nodes, 2))

    for i in prange(total_nodes):
        community_i, role_i = (i // (n * k), (i % (n * k)) // n)
        feature_matrix[i, 0] = community_i
        feature_matrix[i, 1] = role_i

        for j in range(i + 1, total_nodes):
            community_j, role_j = (j // (n * k), (j % (n * k)) // n)
            prob = omega_role[role_i, role_j] if community_i == community_j else p
            adjacency_matrix[i, j] = adjacency_matrix[j, i] = prob

    return adjacency_matrix, feature_matrix


@njit(parallel=True)
def _sample_graph(prob_adj_matrix: np.ndarray, random_seed: int):
    """Samples a adjacency matrix from a probabilistic adjacency matrix inplace."""
    n = prob_adj_matrix.shape[0]

    for i in prange(n):
        if random_seed is not None:
            np.random.seed(i + random_seed)
        for j in range(i + 1, n):
            if np.random.rand() <= prob_adj_matrix[i, j]:
                val = 1
            else:
                val = 0
            prob_adj_matrix[i, j] = prob_adj_matrix[j, i] = val


def role_infused_partition(
    c: int, n: int, p: float, omega_role: np.ndarray, random_seed: int = None, return_networkx_graph: bool = True
):
    """
    Generates a random role-infused partition graph as a NetworkX graph.
    Each node has two integer labels that indicate it's community and role.

    Implements role infused partition model as defined in:
    Scholkemper, M., & Schaub, M. T. (2023). An Optimization-based Approach To Node Role Discovery in Networks:
    Approximating Equitable Partitions. arXiv preprint arXiv:2305.19087.

    Args:
        c (int): Number of communities.
        n (int): Number of nodes per role.
        p (float): Probability of inter-community connections.
        omega_role (numpy.ndarray): Connection probabilities between roles within the same community.
        random_seed (int, optional): Random seed for reproducibility.
        return_networkx_graph (bool, optiona): If true returns a networkx graph, else the adjacency and feature matrix.

    Returns:
        numpy.ndarray: Sampled adjacency matrix of the graph.
        numpy.ndarray: Feature matrix.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    adj_matrix, feature_matrix = _generate_adjacency_and_features(c, n, p, omega_role)
    _sample_graph(adj_matrix, random_seed)

    if return_networkx_graph:
        G = nx.from_numpy_array(adj_matrix)
        for node_id, features in enumerate(feature_matrix):
            G.nodes[node_id]["features"] = features
        return G
    else:
        return adj_matrix, feature_matrix
