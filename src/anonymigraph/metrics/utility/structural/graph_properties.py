import networkx as nx
import numpy as np

from .abstract_graph_metric import AbstractGraphMetric


class ConnectedComponentsMetric(AbstractGraphMetric):
    """Compute the number of connected components in a graph."""

    def compute_scalar(self, G: nx.Graph):
        return nx.number_connected_components(G)


class NumberOfEdgesMetric(AbstractGraphMetric):
    """Calculate the total number of edges in a graph."""

    def compute_scalar(self, G: nx.Graph):
        return G.number_of_edges()


class NumberOfNodesMetric(AbstractGraphMetric):
    """Determine the total number of nodes in a graph."""

    def compute_scalar(self, G: nx.Graph):
        return G.number_of_nodes()


class MaxDegreeMetric(AbstractGraphMetric):
    """Determine the maximum degree of nodes in a graph."""

    def compute_scalar(self, G: nx.Graph):
        return max(dict(G.degree()).values())


class MedianDegreeMetric(AbstractGraphMetric):
    """Determine the median degree of nodes in a graph."""

    def compute_scalar(self, G: nx.Graph):
        degrees = list(dict(G.degree()).values())
        return np.median(degrees)


class MeanDegreeMetric(AbstractGraphMetric):
    """Determine the mean degree of nodes in a graph."""

    def compute_scalar(self, G: nx.Graph):
        degrees = list(dict(G.degree()).values())
        return np.mean(degrees)


class NumberOfTrianglesMetric(AbstractGraphMetric):
    """Count the total number of triangles in a graph."""

    def __init__(self):
        super().__init__(pass_graph_as_igraph=False)

    def compute_scalar(self, G: nx.Graph):
        return sum(nx.triangles(G).values()) / 3


class AverageClusteringCoefficientMetric(AbstractGraphMetric):
    """Calculate the average clustering coefficient of a graph."""

    def __init__(self):
        super().__init__(pass_graph_as_igraph=False)

    def compute_scalar(self, G: nx.Graph):
        return nx.average_clustering(G)


class TransitivityMetric(AbstractGraphMetric):
    """Compute the transitivity of a graph. Uses GraphBLAS to accelerate the calculation."""

    def __init__(self):
        super().__init__(pass_graph_as_igraph=False)

    def compute_scalar(self, G: nx.Graph):
        return nx.transitivity(G)
