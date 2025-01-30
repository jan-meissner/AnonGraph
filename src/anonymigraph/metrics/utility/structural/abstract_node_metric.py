import networkx as nx
from numpy.typing import ArrayLike
from scipy.stats import wasserstein_distance

from anonymigraph.metrics.abstract_metric import AbstractMetric


class AbstractNodeMetric(AbstractMetric):
    """
    Base class for computing and calculating node-level graph metrics.
    """

    def compute_node_distribution(self, G: nx.Graph) -> ArrayLike:
        """
        Computes the distribution of a node-level metric for a given graph.

        Args:
            G (nx.Graph): The graph to compute the node metric distribution for.

        Returns:
            ArrayLike: An array-like object representing the distribution of the node metric.
        """
        raise NotImplementedError("Subclass implements this method.")

    def evaluate(self, G: nx.Graph, Ga: nx.Graph):
        """
        Default evaluation method for node-level graph metrics.

        Returns:
            float: The calculated distance between the two distributions.
        """
        centrality_G = self.compute_node_distribution(G)
        centrality_Ga = self.compute_node_distribution(Ga)
        return self.distribution_distance_func(centrality_G, centrality_Ga)

    def distribution_distance_func(self, valsP: ArrayLike, valsQ: ArrayLike):
        """
        Computes the distance between two distributions using the 1-Wasserstein distance.

        This method can be overridden by subclasses to use different measures of distribution distances.
        Such as p-Wasserstein distance.

        Args:
            valsP (ArrayLike): The distribution of the node-level metric for the original graph.
            valsQ (ArrayLike): The distribution of the node-level metric for the anonymized graph.

        Returns:
            float: The calculated distance between the two distributions.
        """
        # from collections import Counter
        # print(dict(sorted(Counter(valsP).items(), key=lambda item: item[0], reverse=True)))
        # print(dict(sorted(Counter(valsQ).items(), key=lambda item: item[0], reverse=True)))
        return wasserstein_distance(valsP, valsQ)
