import logging

import networkx as nx
import numpy as np

from anonymigraph.anonymization._method_private_colors.losses import (
    undirected_p_loss,
    undirected_u_katz_loss,
    undirected_u_short_term,
)
from anonymigraph.utils import calculate_katz

logger = logging.getLogger(__name__)


class SamplingFreeEvaluator:
    def __init__(
        self,
        G,
        colors,
        w=1,
        use_katz_utility=True,
        alpha=0.01,
        beta=1,
    ):
        """
        Helper class for evaluation of coloring found by the optimization algorithms.
        """
        self.G = G
        self.colors = colors
        self.colors_arr = np.array([self.colors[u] for u in self.G.nodes()], dtype=np.int64)
        self.w = w
        self.unique_colors = np.unique(self.colors_arr)

        self.use_katz_utility = use_katz_utility
        self.alpha = alpha
        self.beta = beta

        self.A_scipy = nx.adjacency_matrix(self.G, nodelist=self.G.nodes()).astype(np.float64)
        self.katz_centrality = calculate_katz(self.A_scipy, alpha=self.alpha, beta=self.beta)

        if self.use_katz_utility:
            self.loss_u = undirected_u_katz_loss(self.katz_centrality, self.colors_arr)
        else:
            self.loss_u = undirected_u_short_term(self.A_scipy, np.array(self.colors_arr, dtype=np.int64))
        self.loss_p, _ = undirected_p_loss(self.A_scipy, np.array(self.colors_arr, dtype=np.int64))

        self.global_min_loss_p, _ = undirected_p_loss(self.A_scipy, np.ones((len(self.colors_arr),)))
        # self.global_min_loss_p = 0.0
        self.global_min_loss_u = 0.0

        self.total_loss = self.loss_u + self.w * self.loss_p

    def get_losses(self):
        return float(self.total_loss), float(self.loss_u), float(self.loss_p)

    def get_suboptimality_gap(self):
        """Difference between total_loss and lower bound on lower bound of global minimum of total loss."""
        return float(self.total_loss - (self.global_min_loss_u + self.w * self.global_min_loss_p))

    def get_num_cluster(self):
        return len(self.unique_colors)

    def get_results(self):
        results = dict(zip(["total_loss", "loss_U", "loss_P"], self.get_losses()))
        results["num_clusters"] = self.get_num_cluster()
        results["sub_opt_gap"] = self.get_suboptimality_gap()
        results["colors"] = self.colors
        return results


def _get_unique_neighbor_colored_statistics(G, colors):
    """
    Helper function for experiment 3.
    """
    node_equiv_class_counts = {}
    node_to_equiv_class = {}

    for node in G.nodes():
        neighbor_colors = [colors[neighbor] for neighbor in G.neighbors(node)]
        color_degree_tuple = tuple(sorted(neighbor_colors))

        node_equiv_class = (colors[node], color_degree_tuple)

        node_to_equiv_class[node] = node_equiv_class

        if node_equiv_class in node_equiv_class_counts:
            node_equiv_class_counts[node_equiv_class] += 1
        else:
            node_equiv_class_counts[node_equiv_class] = 1

    counts = np.array([node_equiv_class_counts[color_degree] for color_degree in node_to_equiv_class.values()])

    # Added here temporarily
    logger.critical(
        f"Equiv. class: Mean Prob. Guess: {np.mean(1/counts)} Mean Prob. Guess No 1s: {np.mean(1/counts[counts > 1])} Median: {np.median(counts)}, Mean: {np.mean(counts)}, Nodes with Unique Color Neighbourhood: {np.sum(counts == 1)}, Number of Nodes: {len(counts)}, Unique Colors: {len(set(colors.values()))}"
    )
