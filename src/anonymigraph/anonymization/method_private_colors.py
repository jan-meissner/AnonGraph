import logging

import networkx as nx

from anonymigraph.anonymization._method_private_colors.ccm_sampler import CCMSampler
from anonymigraph.anonymization._method_private_colors.helpers import (
    _get_unique_neighbor_colored_statistics,
)
from anonymigraph.anonymization._method_private_colors.optimizers import (
    LocalSearchColorOptimizer,
    Optimal1dColorOptimizer,
)

from .abstract_anonymizer import AbstractAnonymizer

logger = logging.getLogger(__name__)


class PrivateColorAnonymizer(AbstractAnonymizer):
    """
    Implements the anonymizer based on our technique when using a Katz-based utility loss.
    """

    def __init__(self, w, alpha, is_eager, use_optimal1d, beta=1):
        self.w = w
        self.alpha = alpha
        self.beta = beta
        self.is_eager = is_eager
        self.use_optimal1d = use_optimal1d

    def anonymize(self, G: nx.Graph, random_seed=None) -> nx.Graph:
        if self.use_optimal1d:
            color_optimizer = Optimal1dColorOptimizer(G, w=self.w, alpha=self.alpha, beta=self.beta)
        else:
            color_optimizer = LocalSearchColorOptimizer(
                G, w=self.w, alpha=self.alpha, beta=self.beta, is_eager=self.is_eager
            )
        color_optimizer.fit(seed=random_seed)

        _get_unique_neighbor_colored_statistics(G, color_optimizer.colors)

        # Sample a new graph
        A, A_prime = CCMSampler(G, color_optimizer.colors, use_pseudograph_to_sample=False).sample(seed=random_seed)

        # Create networkx anonymized graph from A_prime (using same node order as G)
        Ga = nx.from_scipy_sparse_array(A_prime)
        order = list(G.nodes())
        relabel_map = {i: order[i] for i in range(len(order))}
        G_new = nx.relabel_nodes(Ga, relabel_map)
        return G_new
