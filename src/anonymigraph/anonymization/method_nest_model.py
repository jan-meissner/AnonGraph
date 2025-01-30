import networkx as nx
import numpy as np

from anonymigraph.anonymization._external.nest_model._rewire import _rewire
from anonymigraph.utils import _validate_input_graph

from ._external.nest_model.fast_wl import WL_fast
from .abstract_anonymizer import AbstractAnonymizer


class NestModelAnonymizer(AbstractAnonymizer):
    """
    Generates a synthetic graph that preserves the d-hop neighborhood structure of a given graph.
    Defined in Stamm et al. (2023).

    Args:
        depth (int): The 'hop' depth in the graph to be preserved in the anonymized version.
                     For example a depth of 1 preserves the 1-hop structure i.e. the degree.
        r (int, optional): Multiplier for the number of edge swap attempts. Higher values ensure Markov Chain is converged.
        parallel (bool, optional): Parallelization.

    References:
        Stamm, F. I., Scholkemper, M., Strohmaier, M., & Schaub, M. T. (2023).
        Neighborhood Structure Configuration Models. Proceedings of the ACM Web Conference 2023.
        [URL: http://arxiv.org/abs/2210.06843]

    Implementation based on:
        https://github.com/Feelx234/nestmodel
    """

    def __init__(self, depth, r=1, parallel=False):
        self.depth = depth
        self.r = r
        self.parallel = parallel

        if self.depth == 0:
            raise ValueError("Algorithm undefined for d=0, please choose d>0.")

    def anonymize(self, G: nx.Graph, random_seed: int = None, initial_colors=None) -> nx.Graph:
        """
        Anonymize and return the anonymized graph.

        Args:
            graph (nx.Graph): The original graph to be anonymized.
            random_seed (int, optional): Random Seed.
            initial_colors (np.ndarray, optional): Initial color labels for the Weisfeiler-Lehman algorithm.

        Returns:
            nx.Graph: The anonymized graph.
        """
        _validate_input_graph(G)

        edges = np.array(G.edges(), dtype=np.uint32)
        bidirectional_edges = np.row_stack((edges, edges[:, [1, 0]]))

        all_depth_colors = WL_fast(bidirectional_edges, labels=initial_colors, max_iter=self.depth)

        edges_rewired = _rewire(
            edges, all_depth_colors[-1].reshape(1, -1), r=self.r, parallel=self.parallel, random_seed=random_seed
        )

        Ga = nx.Graph()
        Ga.add_nodes_from(G.nodes(data=True))
        Ga.add_edges_from(edges_rewired)

        return Ga


def _validate_nest(G: nx.Graph, G_out: nx.Graph, depth: int, initial_colors: np.ndarray = None):
    """
    Validate the Nest Model. Checks if all

    Args:
        G: Original graph.
        G_out: Output graph from nest_model.
        depth: Same parameter as in nest_model.
        initial_colors  Same parameter as in nest_model.
    """
    edges = np.array(G.edges, dtype=np.uint32)
    edges_rewired = np.array(G_out.edges, dtype=np.uint32)

    bidirectional_edges = np.row_stack((edges, edges[:, [1, 0]]))
    bidirectional_edges_rewired = np.row_stack((edges_rewired, edges_rewired[:, [1, 0]]))

    all_depth_colors = WL_fast(bidirectional_edges, num_nodes=None, labels=initial_colors, max_iter=depth + 1 + 100)
    all_depth_colors_rewired = WL_fast(
        bidirectional_edges_rewired, num_nodes=None, labels=initial_colors, max_iter=depth + 1 + 100
    )

    for i in range(min(depth + 1, len(all_depth_colors))):
        # print(i, all_depth_colors[i])
        # print(i, all_depth_colors_rewired[i])
        if not np.all(all_depth_colors[i] == all_depth_colors_rewired[i]):
            raise AssertionError(f"Validation failed at depth {i}")
