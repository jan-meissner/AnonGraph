import networkx as nx
import numpy as np
from numba import njit

from anonymigraph.utils import _validate_input_graph

from .abstract_anonymizer import AbstractAnonymizer


class RandomEdgeAddDelAnonymizer(AbstractAnonymizer):
    """
    Implements the graph anonymization method described by Hay et al. (2007) for anonymizing social networks.
    Delete m edges sampled uniformly at random, and then add m edges sampled uniformly at random.

    Args:
        m (int): The number of edges to delete and then insert.

    References:
        Hay, M., Miklau, G., Jensen, D., Weis, P., & Srivastava, S. (2007). Anonymizing social networks. Computer
        Science Department Faculty Publication Series. https://scholarworks.umass.edu/cs_faculty_pubs/180
    """

    def __init__(self, m: int):
        self.m = m

    def anonymize(self, G: nx.Graph, random_seed=None) -> nx.Graph:
        if random_seed:
            np.random.seed(random_seed)

        _validate_input_graph(G)

        edges = np.array(G.edges(), dtype=np.uint32).reshape(-1, 2)

        num_edges_to_remove = min(self.m, len(edges))
        indices_to_remove = np.random.choice(len(edges), size=num_edges_to_remove, replace=False)
        edges_remaining = np.delete(edges, indices_to_remove, axis=0)

        new_edges = add_new_edges(
            edges_remaining, self.m, G.number_of_nodes(), random_seed=random_seed if random_seed else -1
        )

        Ga = nx.Graph()
        Ga.add_nodes_from(G.nodes(data=True))
        Ga.add_edges_from(new_edges)

        return Ga


@njit(cache=True)
def add_new_edges(edges: np.ndarray, m: int, num_nodes: int, random_seed: int = -1):
    """
    Adds m random edges that don't exist to the graph defined through the argument edges.
    Assumes that edges are labels from 0 to num_nodes-1.
    """
    if random_seed != -1:
        np.random.seed(random_seed + 1)

    edge_set = set()
    for i in range(len(edges)):
        edge_set.add((edges[i, 0], edges[i, 1]))
        edge_set.add((edges[i, 1], edges[i, 0]))

    num_max_possible_edges = num_nodes * (num_nodes - 1) / 2
    num_current_edges = len(edges)
    num_edges_to_add = min(m, num_max_possible_edges - num_current_edges)

    new_edges = []
    while len(new_edges) < num_edges_to_add:
        n1, n2 = np.random.randint(0, num_nodes, size=2)
        if n1 != n2 and (n1, n2) not in edge_set:
            new_edges.append([n1, n2])

            edge_set.add((n1, n2))
            edge_set.add((n2, n1))

    new_edges_array = np.array(new_edges, dtype=np.uint32)
    updated_edges = np.row_stack((edges, new_edges_array))

    return updated_edges
