from collections import Counter

import networkx as nx

from anonymigraph.metrics.abstract_metric import AbstractMetric
from anonymigraph.metrics.utility.structural.abstract_graph_metric import (
    AbstractGraphMetric,
)


class EdgeJaccardMetric(AbstractMetric):
    """Compute the Jaccard Index of the original and anonymized edge sets"""

    def evaluate(self, G: nx.Graph, Ga: nx.Graph):
        edges_G = set(G.edges())
        edges_Ga = set(Ga.edges())

        intersection = edges_G.intersection(edges_Ga)
        union = edges_G.union(edges_Ga)

        return len(intersection) / len(union)


class PercentageKDegreeAnonMetric(AbstractGraphMetric):
    """Compute the fraction of nodes that are k degree anonymous."""

    def __init__(self, k):
        super().__init__(pass_graph_as_igraph=False)

        self.k = k

    def compute_scalar(self, G: nx.Graph):
        counts = Counter(list(dict(G.degree()).values()))
        return sum(count for count in counts.values() if count >= self.k) / G.number_of_nodes()
