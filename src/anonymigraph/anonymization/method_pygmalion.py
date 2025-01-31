import random

import networkx as nx
import numpy as np
from scipy import sparse
from sklearn.isotonic import IsotonicRegression

from .abstract_anonymizer import AbstractAnonymizer


class PygmalionModelAnonymizer(AbstractAnonymizer):
    """
    Applies edge-differential privacy to the joint degree matrix of a graph.

    The method implements differential privacy as described by Sala et al. [1],
    adding noise to the graphs joint degree matrix, then generating a synthetic
    graph using the sampling method from Mahadevan et al. [2].

    Args:
        eps (float): Epsilon parameter for controlling the level of privacy.

    References:
        [1] Sala et al. (2011). "Sharing graphs using differentially private graph models."
        [2] Mahadevan et al. (2006). "Systematic topology analysis and generation using degree correlations."
    """

    def __init__(self, eps: float):
        self.eps = eps

    def anonymize(self, graph: nx.Graph, random_seed=None) -> nx.Graph:
        if random_seed is not None:
            random.seed(random_seed)

        if nx.is_empty(graph):
            return graph

        if graph.is_directed():
            raise nx.NetworkXNotImplemented("Not implemented for directed graphs.")

        jdm_G = get_joint_degree_matrix(graph)
        jdm_G_priv = eps_diff_privacy_pygmalion(jdm_G, eps=self.eps)
        G_multi = sample_pseudograph_2k_graph(jdm_G_priv)
        G = nx.Graph(G_multi)
        return nx.convert_node_labels_to_integers(G)


def get_joint_degree_matrix(G):
    # gives distribution of p(k1, k2) (but not probabilities but in frequencies)
    A = nx.adjacency_matrix(G)
    degree_vec = A.sum(axis=1)

    # only upper triangle to avoid double counts
    A_upper = sparse.triu(A, k=1).tocoo()
    source_nodes = A_upper.row
    dest_nodes = A_upper.col

    degrees_source = degree_vec[source_nodes]
    degrees_dest = degree_vec[dest_nodes]
    # ensure that degrees_source < degrees_dest by computing element-wise min and max
    degrees_min = np.minimum(degrees_source, degrees_dest)
    degrees_max = np.maximum(degrees_source, degrees_dest)

    # Get maximum degree
    d_max = degree_vec.max()
    matrix_size = d_max + 1
    dk_matrix = sparse.coo_matrix(
        (np.ones_like(degrees_min), (degrees_min, degrees_max)), shape=(matrix_size, matrix_size)
    )

    # count by summing duplicates
    dk_matrix.sum_duplicates()
    dk_matrix.tocsr().tocoo()  # ensure ordering of row/col/data

    return dk_matrix.row, dk_matrix.col, dk_matrix.data  # return k1, k2 and count


def sample_pseudograph_2k_graph(jdm, seed=None):
    """
    Modified version to work with joint degree frequencies that are NOT realizable.
    Generate a pseudograph using the pseudograph approach for a given 2K joint degree distribution.

    Mahadevan, Priya, et al. "Systematic topology analysis and generation using degree correlations."
    ACM SIGCOMM Computer Communication Review 36.4 (2006): 135-146.
    """

    if seed is not None:
        random.seed(seed)

    stubs_per_degree = {}  # key: degree value: list of stubs connected to that degree
    edge_id = 0

    edge_ids = []
    for k1, k2, count in zip(*jdm):
        for _ in range(count):
            edge_id += 1
            edge_ids.append(edge_id)

            if k1 not in stubs_per_degree:
                stubs_per_degree[k1] = []
            if k2 not in stubs_per_degree:
                stubs_per_degree[k2] = []

            # edges have two stubs one side connects to k1 degree nodes other to k2
            stubs_per_degree[k1].append((edge_id, "k1"))
            stubs_per_degree[k2].append((edge_id, "k2"))

    node_id = 0
    stub_to_node = {}  # save to which node stubs connect to
    nodes = set()

    for k, stubs in stubs_per_degree.items():
        random.shuffle(stubs)

        while stubs:
            node_stubs = []
            for _ in range(min(k, len(stubs))):
                node_stubs.append(stubs.pop())

            # Normally we would ensure that all nodes are degree k but this is not always the case as
            # our dk-2-series is not necessarily realizable
            # if len(node_stubs) == k:
            node_id += 1
            nodes.add(node_id)

            for stub in node_stubs:
                stub_to_node[stub] = node_id

    G = nx.MultiGraph()
    for node in nodes:
        G.add_node(node)
    for edge_id in edge_ids:
        stub1 = (edge_id, "k1")
        stub2 = (edge_id, "k2")
        node1 = stub_to_node[stub1]
        node2 = stub_to_node[stub2]
        G.add_edge(node1, node2)

    return G


def eps_diff_privacy_pygmalion(jdm, eps=100):
    k1_u, k2_v, count = jdm

    # Sort for isontonic regression
    sorted_indices = np.argsort(count)
    k1_u = k1_u[sorted_indices]
    k2_v = k2_v[sorted_indices]
    count = count[sorted_indices]

    # corresponds to finest ∂ ordering on dK-2
    # Get beta subseries Sensitivity for minimal cluster size
    max_degree = np.maximum(k1_u, k2_v)
    beta_subseries_sensitivity = 4 * (max_degree - 1) + 1

    # ------------------------
    # e - diff private dk-2-series:
    noise = np.random.laplace(loc=0.0, scale=beta_subseries_sensitivity, size=count.shape) / eps
    count = count + noise

    # ------------------------
    # Isotonic smoothing as postprocessing
    ir = IsotonicRegression(out_of_bounds="clip")
    isotonic_count = ir.fit_transform(np.arange(len(count)), count)

    return k1_u, k2_v, np.round(isotonic_count).astype(int)
