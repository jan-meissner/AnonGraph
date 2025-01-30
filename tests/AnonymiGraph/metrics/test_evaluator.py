import networkx as nx
import numpy as np

from anonymigraph.metrics.evaluator import Evaluator
from anonymigraph.metrics.utility.structural import (  # ClosenessCentralityMetric,
    AverageClusteringCoefficientMetric,
    ConnectedComponentsMetric,
    DegreeCentralityMetric,
    EigenvectorMetric,
    LocalClusteringCoefficientMetric,
    MaxDegreeMetric,
    MeanDegreeMetric,
    MedianDegreeMetric,
    NumberOfEdgesMetric,
    NumberOfNodesMetric,
    NumberOfTrianglesMetric,
    PageRankMetric,
    TransitivityMetric,
    WLColorMetric,
)
from anonymigraph.utils import relabel_graph


def test_utilites_identical():
    metrics = {
        "|V|": NumberOfNodesMetric(),
        "|E|": NumberOfEdgesMetric(),
        "|Δ|": NumberOfTrianglesMetric(),
        "|CC|": ConnectedComponentsMetric(),
        "Median Deg.": MedianDegreeMetric(),
        "Avg. Deg.": MeanDegreeMetric(),
        "Max Deg.": MaxDegreeMetric(),
        "Transitivity": TransitivityMetric(),
        "Avg. Clustering": AverageClusteringCoefficientMetric(),
        "Degree Centrality": DegreeCentralityMetric(),
        "Eigenvector Centrality": EigenvectorMetric(),
        "TVD WL Colors d=2": WLColorMetric(depth=2),
        "PageRank": PageRankMetric(),
        "Local Clustering Coefficient": LocalClusteringCoefficientMetric(),
    }

    evaluator = Evaluator(metrics, use_igraph=True)
    G = nx.gnp_random_graph(2000, 50 / 2000, seed=1233)

    data = {"no anonymization": evaluator.evaluate(G, G)}

    for method, metrics in data.items():
        for key, metric in metrics.items():
            if isinstance(metric, dict):
                assert np.allclose(metric["G"], metric["Ga"]), f"Method: {method}, Metric: {key}, Score: {metric}"
            else:
                assert np.allclose(metric, 0), f"Method: {method}, Metric: {key}, Score: {metric}"


def test_utilites_specific():
    metrics = {
        "|Nodes|": NumberOfNodesMetric(),
        "|Edges|": NumberOfEdgesMetric(),
        "|Triangles|": NumberOfTrianglesMetric(),
        "|Connected Components|": ConnectedComponentsMetric(),
        "Transitivity": TransitivityMetric(),
        "Average Clustering": AverageClusteringCoefficientMetric(),
        "Degree Centrality": DegreeCentralityMetric(),
        "Eigenvector Centrality": EigenvectorMetric(),
        "TVD WL Colors d=2": WLColorMetric(depth=2),
        "PageRank": PageRankMetric(),
        "Local Clustering Coefficien": LocalClusteringCoefficientMetric(),
    }
    evaluator = Evaluator(metrics, use_igraph=True)

    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (3, 1)])
    G.add_edges_from([(3, 4), (4, 5), (5, 3)])
    G.add_edge(1, 0)
    G.add_edge(2, 6)
    G.add_edge(3, 7)

    Ga = nx.Graph(G)
    Ga.remove_edge(5, 3)
    Ga.remove_edge(3, 2)
    Ga.add_edge(5, 2)

    eval = evaluator.evaluate(relabel_graph(G), relabel_graph(Ga))
    eval_expected = {
        "|Nodes|": {"G": 8, "Ga": 8},
        "|Edges|": {"G": 9, "Ga": 8},
        "|Triangles|": {"G": 2, "Ga": 0},
        "|Connected Components|": {"G": 1, "Ga": 1},
        "Transitivity": {"G": 3 * 2 / 18, "Ga": 0},
        "Average Clustering": {"G": 43 / 120, "Ga": 0},
        "Degree Centrality": 0.035714285714285705,
        "Eigenvector Centrality": 0.04159283405776115,
        "TVD WL Colors d=2": 0.75,
        "PageRank": 0.019091687199899968,
        "Local Clustering Coefficien": 0.35833333333333334,
    }

    assert set(eval.keys()) == set(eval_expected.keys())

    for key in eval_expected.keys():
        if isinstance(eval_expected[key], dict):
            assert np.allclose(
                eval_expected[key]["G"], eval[key]["G"]
            ), f"Assertion failed for key '{key}': Expected 'G' value {eval_expected[key]['G']} got {eval[key]['G']}"
            assert np.allclose(
                eval_expected[key]["Ga"], eval[key]["Ga"]
            ), f"Assertion failed for key '{key}': Expected 'Ga' value {eval_expected[key]['Ga']} got {eval[key]['Ga']}"
        else:
            assert np.allclose(
                eval_expected[key], eval[key]
            ), f"Assertion failed for key '{key}': Expected value {eval_expected[key]} got {eval[key]}"
