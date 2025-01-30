import networkx as nx

from anonymigraph.anonymization import ConfigurationModelAnonymizer


def test_empty_graph():
    anonymized_graph = ConfigurationModelAnonymizer().anonymize(nx.Graph())
    assert nx.is_empty(anonymized_graph), "The anonymized graph should be empty."


def test_graph_properties_preservation():
    graph = nx.erdos_renyi_graph(n=40, p=0.5)
    anonymized_graph = ConfigurationModelAnonymizer().anonymize(graph)
    assert graph.number_of_nodes() == anonymized_graph.number_of_nodes(), "Number of nodes should be preserved."
