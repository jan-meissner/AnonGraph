import logging

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


def _validate_input_graph(G: nx.Graph):
    if not isinstance(G, nx.Graph):
        raise TypeError("The graph must be undirected. Please provide an undirected graph.")

    nodes = list(G.nodes())
    expected_labels = list(range(len(G)))

    if nodes != expected_labels:
        raise ValueError(
            "Graph nodes must be labeled with integers from 0 to G.number_of_nodes() - 1. Please relabel the graph"
            "accordingly or use `utils.convert_node_labels_to_integers`."
        )


def relabel_graph(G: nx.Graph) -> nx.Graph:
    """
    Returns a copy of the graph G with nodes relabeled to integers from 0 to G.number_of_nodes() - 1.

    Parameters:
    - G (nx.Graph): The graph to relabel.

    Returns:
    - nx.Graph: A copy of G with sequentially labeled nodes.
    """
    mapping = {node: i for i, node in enumerate(G.nodes())}
    return nx.relabel_nodes(G, mapping, copy=True)


def calculate_katz(A, alpha=0.1, beta=1.0, num_iters=10000, tol=None):
    """
    Calculate the Katz Centrality with numpy.
    """
    if num_iters is None:
        num_iters = float("inf")
    if tol is None:
        tol = 1e-12

    n = A.shape[0]
    vec = np.zeros(n)

    iter_count = 0
    while iter_count < num_iters:
        vec_next = alpha * A @ vec + beta

        if np.linalg.norm(vec_next - vec) < tol:
            logger.info(f"Katz converged after {iter_count} iterations.")
            break

        vec = vec_next

        if iter_count == 10000:
            logger.warning(
                "Iteration count exceeded 10,000. Consider increasing tolerance or checking the matrix for convergence properties."
            )
        if iter_count == 100000:
            logger.warning(
                "Iteration count exceeded 100,000. Consider increasing tolerance or checking the matrix for convergence properties."
            )
        if iter_count == 1000000:
            logger.warning(
                "Iteration count exceeded 1,000,000. Consider increasing tolerance or checking the matrix for convergence properties."
            )

        if iter_count >= num_iters:
            logger.warning("Number of iterations exceeded the maximum (num_iters) allowed iterations.")
            break

        iter_count += 1

    return vec
