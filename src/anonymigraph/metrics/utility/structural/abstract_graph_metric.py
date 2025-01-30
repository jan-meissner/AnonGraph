from typing import Any

import networkx as nx

from anonymigraph.metrics.abstract_metric import AbstractMetric


class AbstractGraphMetric(AbstractMetric):
    """
    Base class for scalar graph metrics computation and comparison.
    """

    def compute_scalar(self, G: nx.Graph) -> Any:
        """
        Computes a scalar value metric for a given graph.

        Args:
            G (nx.Graph): The graph to compute the scalar metric for.

        Returns:
            Any: The scalar value representing the computed metric for the graph.
        """
        raise NotImplementedError("Subclass implements this method")

    def evaluate(self, G: nx.Graph, Ga: nx.Graph):
        """
        Default evaluation method for scalar graph metrics.

        Returns:
            dict: Scalar metric for both G and Ga as a dict.
        """
        num_G = self.compute_scalar(G)
        num_Ga = self.compute_scalar(Ga)

        return {"G": num_G, "Ga": num_Ga}
