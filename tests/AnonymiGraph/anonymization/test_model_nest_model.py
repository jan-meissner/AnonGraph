import networkx as nx

import anonymigraph.anonymization as anon


def test_nest_model():
    G = nx.erdos_renyi_graph(1000, 15 / 1000)
    G.add_nodes_from([1000])  # add isolate

    depth = 2
    G_nest = anon.NestModelAnonymizer(depth=depth, r=80).anonymize(G)
    anon.method_nest_model._validate_nest(G, G_nest, depth)


def test_nest_model_preserve_degree():
    G = nx.erdos_renyi_graph(1000, 15 / 1000)
    G.add_nodes_from([1000])  # add isolate

    depth = 1
    G_nest = anon.NestModelAnonymizer(depth=depth, r=80).anonymize(G)
    anon.method_nest_model._validate_nest(G, G_nest, depth)

    degrees_G = set(d for n, d in G.degree())
    degrees_G_nest = set(d for n, d in G_nest.degree())

    assert degrees_G == degrees_G_nest
