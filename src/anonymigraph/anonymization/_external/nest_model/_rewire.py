from anonymigraph.anonymization._external.nest_model.fast_rewire import (
    get_block_indices,
    rewire_fast,
    sort_edges,
)


def _rewire(edges, colors, r=1, parallel=True, random_seed=None):
    """
    Rewires an undirected graph's subgraphs based on partition labels.

    Args:
      edges: np.ndarray (num_edges, 2), unique, arbitrarily oriented edges (each edge appears once)
      colors: np.ndarray (num_nodes,), color labels for nodes
      r: int, multiplier of edge swaps attempted (total swap attempts are r*num_edges)
      parallel: bool, enables parallelization
      random_seed: int, seed for random number generator. If set parallel also needs to be set to false.
    """
    edges_ordered, edges_classes, dead_arr, is_mono = sort_edges(edges, colors, is_directed=False)
    block_indices = get_block_indices(edges_classes, dead_arr)

    rewire_fast(
        edges_ordered,
        edges_classes[:, 0],
        is_mono[0],
        block_indices[0],
        is_directed=False,
        seed=random_seed,
        num_flip_attempts_in=r,
        parallel=parallel,
    )

    return edges_ordered
