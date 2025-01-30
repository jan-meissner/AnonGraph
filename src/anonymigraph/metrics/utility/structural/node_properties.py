import networkx as nx
import numpy as np

from anonymigraph.metrics.utility.structural.abstract_node_metric import (
    AbstractNodeMetric,
)


class DegreeCentralityMetric(AbstractNodeMetric):
    """
    This class calculates and compares the degree centrality of two graphs.
    The comparison is done using the default distribution distance function used by AbstractNodeMetric.
    Uses GraphBLAS to accelerate the calculation.
    """

    def __init__(self):
        super().__init__()  # graphblas = True causes floating point errors

    def compute_node_distribution(self, G: nx.Graph):
        return list(nx.degree_centrality(G).values())


class EigenvectorMetric(AbstractNodeMetric):
    """
    This class calculates and compares the eigenvector centralities of two graphs.
    The comparison is done using the default distribution distance function used by AbstractNodeMetric.
    Uses GraphBLAS to accelerate the calculation.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the EigenvectorMetric.

        Args:
            *args: Additional positional arguments to be passed to nx.eigenvector_centrality.
            **kwargs: Additional keyword arguments to be passed to nx.eigenvector_centrality.
        """
        super().__init__(pass_graph_as_igraph=False)
        self.args = args
        self.kwargs = kwargs

        if "max_iter" not in self.kwargs:
            self.kwargs["max_iter"] = 1000

    def compute_node_distribution(self, G: nx.Graph):
        return list(nx.eigenvector_centrality(G, *self.args, **self.kwargs).values())


class LocalClusteringCoefficientMetric(AbstractNodeMetric):
    """
    This class calculates and compares the local clustering coefficients of two graphs.
    The comparison is done using the default distribution distance function used by AbstractNodeMetric.
    """

    def __init__(self):
        super().__init__(pass_graph_as_igraph=False)

    def compute_node_distribution(self, G: nx.Graph):
        return list(nx.clustering(G).values())


class KatzCentralityMetric(AbstractNodeMetric):
    """
    This class calculates and compares the Katz centralities of two graphs.
    The comparison is done using the default distribution distance function used by AbstractNodeMetric.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the KatzCentralityMetric.

        Args:
            *args: Additional positional arguments to be passed to nx.katz_centrality.
            **kwargs: Additional keyword arguments to be passed to nx.katz_centrality.
        """
        super().__init__(pass_graph_as_igraph=False)
        self.args = args
        self.kwargs = kwargs

    def compute_node_distribution(self, G: nx.Graph):
        return list(nx.katz_centrality(G, *self.args, **self.kwargs).values())


class PageRankMetric(AbstractNodeMetric):
    """
    This class calculates and compares the PageRank centralities of two graphs.
    The comparison is done using the default distribution distance function used by AbstractNodeMetric.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the PageRankMetric.

        Args:
            *args: Additional positional arguments to be passed to nx.pagerank.
            **kwargs: Additional keyword arguments to be passed to nx.pagerank.
        """
        super().__init__(pass_graph_as_igraph=False)
        # Store args and kwargs to be used in compute_node_distribution
        self.args = args
        self.kwargs = kwargs

    def compute_node_distribution(self, G: nx.Graph):
        return list(nx.pagerank(G, *self.args, **self.kwargs).values())


class ClosenessCentralityMetric(AbstractNodeMetric):
    """
    This class calculates and compares the closeness centralities of two graphs.
    The comparison is done using the default distribution distance function used by AbstractNodeMetric.
    """

    def __init__(self):
        super().__init__(pass_graph_as_igraph=True)

    def compute_node_distribution(self, G: nx.Graph):
        # Isolates are treated assigned nan in igraph - instead use 0 as done by networkx too
        return [0 if np.isnan(x) else x for x in G.closeness()]
