import json
import os

import networkx as nx

from anonymigraph.anonymization import KDegreeAnonymizer
from anonymigraph.utils import relabel_graph

# def write_graph(G, fname):
#    """
#    Use this function to save graphs used for the test case.
#    """
#    # write_graph(G, "graph.json")
#    # write_graph(Ga,"graph_anon.json")
#    with open(fname, "w") as file:
#       json.dump(list(G.edges()), file)


def load_graph(fname):
    with open(fname, "r") as file:
        edges = json.load(file)

    G = nx.Graph()
    max_node = max(edge[0] for edge in edges)  # undirected graph so we can just look at the source node
    G.add_nodes_from(range(max_node + 1))
    G.add_edges_from(edges)
    return G


def test_ensuring_equivalence_with_original():
    """
    Generated testdata via:

    G = nx.erdos_renyi_graph(1000, 50/1000)

    # Anonymize the Graph G
    k = 100
    Ga = k_degree_anonymity(G, k, noise=10, with_deletions=True, random_seed=31221)

    write_graph(G, "graph.json")
    write_graph(Ga,"graph_anon.json")
    """

    current_dir = os.path.dirname(__file__)
    graph_path = os.path.join(current_dir, "testdata/graph.json")
    graph_anon_path = os.path.join(current_dir, "testdata/graph_anon.json")

    G = load_graph(graph_path)
    expected_Ga = load_graph(graph_anon_path)

    G = relabel_graph(G)
    expected_Ga = relabel_graph(expected_Ga)

    k = 100
    Ga = KDegreeAnonymizer(k, noise=10, with_deletions=True).anonymize(G, random_seed=31221)

    assert expected_Ga.edges() == Ga.edges()
