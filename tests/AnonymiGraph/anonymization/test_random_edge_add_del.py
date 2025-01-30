import networkx as nx
import pytest

from anonymigraph.anonymization import RandomEdgeAddDelAnonymizer


@pytest.fixture
def create_graph():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
    return G


def test_same_number_of_nodes(create_graph):
    anonymizer = RandomEdgeAddDelAnonymizer(m=2)
    Ga = anonymizer.anonymize(create_graph, random_seed=42)
    assert len(create_graph.nodes()) == len(Ga.nodes()), "Anonymized graph should have the same number of nodes"


def test_edge_modifications(create_graph):
    anonymizer = RandomEdgeAddDelAnonymizer(m=2)
    Ga = anonymizer.anonymize(create_graph, random_seed=42)
    assert (
        Ga.number_of_edges() == create_graph.number_of_edges()
    ), "Anonymized graph edges should be the same as long as the original graph has more edges than m"


def test_no_self_loops(create_graph):
    anonymizer = RandomEdgeAddDelAnonymizer(m=2)
    Ga = anonymizer.anonymize(create_graph, random_seed=42)
    for u, v in Ga.edges():
        assert u != v, "Anonymized graph should not have self-loops"


def test_reproducibility(create_graph):
    anonymizer = RandomEdgeAddDelAnonymizer(m=2)
    Ga1 = anonymizer.anonymize(create_graph, random_seed=42)
    Ga2 = anonymizer.anonymize(create_graph, random_seed=42)
    assert nx.is_isomorphic(Ga1, Ga2), "Anonymization should be reproducible with the same seed"


def test_add_remove_more_edges_than_possible(create_graph):
    anonymizer = RandomEdgeAddDelAnonymizer(m=100)
    Ga = anonymizer.anonymize(create_graph, random_seed=42)
    assert Ga.number_of_edges() == 6, "When m is too large it should return a complete graph."


def test_large_graph():
    m = 100
    G = nx.erdos_renyi_graph(3000, 10 / 3000, seed=42)
    anonymizer = RandomEdgeAddDelAnonymizer(m=m)
    Ga = anonymizer.anonymize(G, random_seed=42)

    G_edges = set(G.edges())
    Ga_edges = set(Ga.edges())

    edges_diff = G_edges.symmetric_difference(Ga_edges)

    assert len(edges_diff) <= 2 * m, "At most 2*m edges can differ between original and anonymized graph."
    assert Ga.number_of_edges() == G.number_of_edges()
