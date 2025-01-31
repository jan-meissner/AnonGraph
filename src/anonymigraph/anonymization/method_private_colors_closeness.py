import logging

import igraph as ig
import networkx as nx
import numpy as np
from scipy.sparse import coo_matrix

from anonymigraph.anonymization._external.nest_model._rewire import _rewire
from anonymigraph.anonymization._method_private_colors.losses import undirected_p_loss
from anonymigraph.utils import _validate_input_graph

logger = logging.getLogger(__name__)


class PrivateClosenessColors:
    def __init__(self, G, w, closeness_samples=100, r=10):
        """
        Implements the anonymizer based on our technique when using the Closeness Centrality utility loss and using an agglomerative clustering optimization approach.
        Slow. Proof-of-concept.

        Args:
            w (float): The privacy-utility trade-off parameter.
            r (int): Multiplier for the number of edge swap attempts. Total swap attempts are r * num_edges.
            closeness_samples (int): The number of samples generated to estimate the closeness centrality based utility loss.
        """

        self.w = w
        self.r = r
        self.closeness_samples = closeness_samples

    def anonymize(self, G, random_seed):
        np.random.seed(random_seed)

        G_ig = ig.Graph.from_networkx(G)
        c_g = np.array(G_ig.closeness())
        c_g = np.where(np.isnan(c_g), 0, c_g)

        A_coo = nx.adjacency_matrix(G).tocoo().astype(np.float64)
        mapped_get_objectives = lambda colors: _get_objectives(G, A_coo, c_g, colors, self.r, self.closeness_samples)

        initial_colors = {i: i for i in G.nodes()}
        colors, _ = _agglomerative_clustering(initial_colors, self.w, mapped_get_objectives)

        # Create networkx anonyimzed graph from A_prime (using same node order as G)
        A_prime = _FastCCMSampler(G, colors).sample(random_seed=random_seed)

        Ga = nx.from_numpy_array(A_prime)
        order = list(G.nodes())
        relabel_map = {i: order[i] for i in range(len(order))}
        G_new = nx.relabel_nodes(Ga, relabel_map)
        return G_new


class _FastCCMSampler:
    def __init__(self, G, colors, r=10):
        _validate_input_graph(G)

        self.r = r
        self.color_arr = np.array([colors[node] for node in list(G.nodes())], dtype=np.uint32)
        _, self.color_arr = np.unique(self.color_arr, return_inverse=True)

        self.edges = np.array(G.edges(), dtype=np.uint32)

    def sample(self, random_seed: int = None) -> nx.Graph:
        edges_rewired = _rewire(
            self.edges, self.color_arr.reshape(1, -1), r=self.r, parallel=False, random_seed=random_seed
        )

        I = edges_rewired[:, 0]
        J = edges_rewired[:, 1]
        I_sym = np.hstack([I, J])
        J_sym = np.hstack([J, I])
        A_csr_sample = coo_matrix((np.ones_like(I_sym), (I_sym, J_sym))).toarray().astype(np.float64)

        return A_csr_sample


def _get_objectives(G, A_coo, c_g, colors, r, closeness_samples):
    sampler = _FastCCMSampler(G, colors, r=r)
    loss_samples = []

    random_base_seed = np.random.randint(0, 10_000_000)
    for i in range(closeness_samples):
        G_a = ig.Graph.Adjacency(sampler.sample(random_seed=random_base_seed + i), mode="undirected")
        c_ga = np.array(G_a.closeness())
        c_ga = np.where(np.isnan(c_ga), 0, c_ga)

        loss = np.linalg.norm(c_ga - c_g)
        loss_samples.append(loss)

    u_loss = np.mean(loss_samples)
    p_loss, _ = undirected_p_loss(A_coo, np.array([colors[node] for node in list(G.nodes())], dtype=np.uint32))

    std_on_u = np.std(loss_samples, ddof=1)

    return u_loss, p_loss, std_on_u


def _agglomerative_clustering(initial_colors, w, get_objectives):
    colors = initial_colors

    u_loss, p_loss, _ = get_objectives(colors)
    current_obj = u_loss + w * p_loss

    improvement = True

    while improvement:
        improvement = False
        best_merge_obj = current_obj
        best_u = None
        best_p = None
        best_merge_pair = (None, None)

        unique_colors = list(set(colors.values()))

        for i in range(len(unique_colors)):
            for j in range(i + 1, len(unique_colors)):
                c1 = unique_colors[i]
                c2 = unique_colors[j]

                merged_c1_c2 = dict(colors)
                for node in merged_c1_c2:
                    if merged_c1_c2[node] == c2:
                        merged_c1_c2[node] = c1

                temp_u_loss, temp_p_loss, temp_u_std = get_objectives(merged_c1_c2)
                temp_obj = temp_u_loss + w * temp_p_loss

                if temp_obj < best_merge_obj:
                    best_u = temp_u_loss
                    best_p = temp_p_loss
                    best_merge_obj = temp_obj
                    best_merge_pair = (c1, c2)

        logger.info(f"New Iter: Merge {best_merge_pair}, {best_merge_obj}, {best_u}, {best_p}")

        if best_merge_pair != (None, None):
            c1, c2 = best_merge_pair

            for node in colors:
                if colors[node] == c2:
                    colors[node] = c1

            current_obj = best_merge_obj
            improvement = True
        else:
            improvement = False

    return colors, current_obj
