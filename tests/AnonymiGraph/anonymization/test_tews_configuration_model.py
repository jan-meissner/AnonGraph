import random
import string
from copy import deepcopy
from itertools import chain

import networkx as nx
import pytest

from anonymigraph.anonymization import ConfigurationModelAnonymizer


# Source: Tews, Mara. "Privacy-Aware Network Sharing with High Utility," 2023
def configuration_model_costum(G):
    """Returns a random graph with the given degree sequence, keeps original node labels."""
    G = deepcopy(G)
    deg_sequence = [d for v, d in G.degree()]
    random.shuffle(deg_sequence)
    node_labels = list(G.nodes)
    deg_sequence_dict = {k: v for (k, v) in zip(node_labels, deg_sequence)}
    node_dict = {i: G.nodes[i] for i in G.nodes}

    if G.is_directed():
        raise nx.NetworkXNotImplemented("not implemented for directed graphs")
    G_rand = _configuration_model_costum(deg_sequence_dict, G)

    G_rand = nx.Graph(G_rand)
    G_rand.remove_edges_from(nx.selfloop_edges(G_rand))
    nx.set_node_attributes(G_rand, node_dict)

    G_rand_labels = nx.Graph()
    G_rand_labels.add_nodes_from(sorted(G_rand.nodes(data=True)))
    G_rand_labels.add_edges_from(G_rand.edges(data=True))

    return G_rand_labels


# Source: Tews, Mara. "Privacy-Aware Network Sharing with High Utility," 2023
def _to_stublist(deg_sequence_dict):
    """Returns a list of degree-repeated node numbers."""
    return list(chain.from_iterable([n] * d for n, d in deg_sequence_dict.items()))


# Source: Tews, Mara. "Privacy-Aware Network Sharing with High Utility," 2023
def _configuration_model_costum(deg_sequence_dict, create_using):
    """Helper function for generating either undirected configuration model graphs."""

    n = len(deg_sequence_dict)
    G_temp = nx.empty_graph(deg_sequence_dict.keys(), create_using)

    if n == 0:
        return G_temp

    if nx.is_directed(G_temp):
        raise nx.NetworkXNotImplemented("not implemented for directed graphs")
    else:
        stublist = _to_stublist(deg_sequence_dict)
        n = len(stublist)
        half = n // 2
        random.shuffle(stublist)
        out_stublist, in_stublist = stublist[:half], stublist[half:]
        G_temp.add_edges_from(zip(out_stublist, in_stublist))
        return G_temp


def get_graph(n, m):
    """Generates a random Gnm Graph."""

    def get_random_string(length):
        return "".join(random.choice(string.ascii_letters) for _ in range(length))

    # Create a random graph
    G = nx.gnm_random_graph(n, m)

    # Add nodes and edges to the new graph with random names
    new_names = {n: get_random_string(5) for n in G.nodes}
    G = nx.relabel_nodes(G, new_names)

    # Assign random labels to each node
    for node in G.nodes():
        G.nodes[node]["label"] = f"Node {node}"

    for edge in G.edges():
        G.edges[edge]["weight"] = f"Edge {edge[0]}-{edge[1]}"

    return G


@pytest.mark.parametrize(
    "random_seed, n, m",
    [
        (1, 2, 3),
        (2, 5, 10),
        (3, 43, 22),
        (4, 0, 0),
        (5, 100, 1000),
    ],
)
def test_same_as_tews(random_seed, n, m):
    random.seed(random_seed)
    graph = get_graph(n, m)

    random.seed(random_seed)
    g_tews = configuration_model_costum(deepcopy(graph))

    random.seed(random_seed)
    g_anonymi = ConfigurationModelAnonymizer().anonymize(graph)

    assert nx.is_isomorphic(g_tews, g_anonymi)
