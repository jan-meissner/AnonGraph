from abc import ABC, abstractmethod

import networkx as nx


class AbstractAnonymizer(ABC):
    """
    Abstract base class for anonymization techniques.
    """

    @abstractmethod
    def anonymize(self, graph: nx.Graph, random_seed=None) -> nx.Graph:
        """
        Anonymize and return the anonymized graph.

        Args:
            graph (nx.Graph): The original graph that is to be anonymized.
            random_seed (int, optional): Seed for the random number generator.

        Returns:
            nx.Graph: The anonymized graph.
        """
