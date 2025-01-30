import networkx as nx

import anonymigraph.anonymization as anon


def test_variant_nest_model_runs():
    G = nx.erdos_renyi_graph(1000, 15 / 1000)
    G.add_nodes_from([1000])  # add isolate
    anon.VariantNestModelAnonymizer(k=2).anonymize(G)
