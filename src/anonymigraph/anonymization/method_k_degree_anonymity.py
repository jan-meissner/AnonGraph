import networkx as nx
import numpy as np

from anonymigraph.utils import _validate_input_graph

from ._external.k_degree_anonymity import k_degree_anonymity as _k_deg
from .abstract_anonymizer import AbstractAnonymizer


class KDegreeAnonymizer(AbstractAnonymizer):
    """
    K-degree anonymity on a given graph G.

    Implementation from Liu & Terzi's paper on k-degree anonymity (see reference [1]).

    Args:
        k (int): The k-anonymity degree parameter, each node's degree is shared by at least k-1 other nodes.
        noise (int, optional): The level of noise to add after failed anonymization attempts.
        with_deletions (bool, optional): If True, allows the deletion of edges to achieve k-degree anonymity.

    References:
        Liu, K., & Terzi, E. (2008). "Towards identity anonymization on graphs." In ACM SIGMOD International Conference
        on Management of Data. https://dl.acm.org/doi/10.1145/1376616.1376629

    Implementation Notes:
        Implementation from:
        https://github.com/blextar/graph-k-degree-anonymity/tree/master
    """

    def __init__(self, k: int, noise: int = 10, with_deletions: bool = True):
        self.k = k
        self.noise = noise
        self.with_deletions = with_deletions

    def anonymize(self, G: nx.Graph, random_seed=None) -> nx.Graph:
        _validate_input_graph(G)

        if random_seed is not None:
            np.random.seed(random_seed)

        Ga = _k_deg(G, self.k, noise=min(self.noise, G.number_of_nodes()), with_deletions=self.with_deletions)

        for node, data in G.nodes(data=True):
            Ga.nodes[node].update(data)

        return Ga
