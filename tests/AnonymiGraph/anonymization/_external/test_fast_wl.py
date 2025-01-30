import numpy as np
import pytest

from anonymigraph.anonymization._external.nest_model.fast_wl import (
    WL_fast,
    convert_labeling,
    my_bincount,
    to_in_neighbors,
)


def create_line_graph(n):
    edges = np.empty((2 * (n - 1), 2), dtype=np.uint64)
    edges[: n - 1, 0] = np.arange(n - 1)
    edges[: n - 1, 1] = np.arange(1, n)
    edges[n - 1 :, 1] = np.arange(n - 1)
    edges[n - 1 :, 0] = np.arange(1, n)
    return edges


def test_convert_labeling_1():
    arr = np.zeros(10, dtype=np.uint32)
    convert_labeling(arr)
    assert np.array_equal(arr, np.zeros(10, dtype=np.uint32))


def test_convert_labeling_2():
    arr = np.arange(11, dtype=np.uint32)
    convert_labeling(arr)
    assert np.array_equal(arr, np.arange(11, dtype=np.uint32))


def test_convert_labeling_3():
    arr = np.arange(11, dtype=np.uint32)
    arr[1] = 1000
    convert_labeling(arr)
    assert np.array_equal(arr, np.arange(11, dtype=np.uint32))


def test_to_in_neighbors():
    edges = np.array([[0, 1, 0], [1, 2, 2]], dtype=np.uint32).T
    arr1, arr2, _ = to_in_neighbors(edges)
    assert np.array_equal(arr1, [0, 0, 1, 3])
    assert np.array_equal(arr2, [0, 1, 0])


@pytest.mark.parametrize("n, expected_lengths", [(8, 4), (7, 4)])
def test_wl_line(n, expected_lengths):
    out = WL_fast(create_line_graph(n), n)
    assert isinstance(out, list)
    assert len(out) == expected_lengths


def test_wl_line_imperfection():
    n = 7
    starting_labels = np.array([0, 0, 0, 100, 0, 0, 0], dtype=np.uint32)
    out = WL_fast(create_line_graph(n), n, starting_labels)
    assert isinstance(out, list)
    assert len(out) == 2
    arr0, arr1 = out
    assert arr0.dtype == starting_labels.dtype
    assert arr1.dtype == starting_labels.dtype
    assert np.array_equal(arr0, [0, 0, 0, 1, 0, 0, 0])
    assert np.array_equal(arr1, [0, 1, 2, 3, 2, 1, 0])


def test_wl_4():
    edges = np.array(
        [
            [0, 3],
            [1, 2],
            [2, 4],
            [2, 5],
            [3, 6],
            [3, 7],
            [4, 8],
            [5, 8],
            [6, 7],
            [3, 0],
            [2, 1],
            [4, 2],
            [5, 2],
            [6, 3],
            [7, 3],
            [8, 4],
            [8, 5],
            [7, 6],
        ],
        dtype=np.uint32,
    )
    out = WL_fast(edges, 9)
    assert isinstance(out, list)
    assert len(out) == 6


def test_wl_isolated_nodes_with_last_node_id():
    edges = np.array(
        [
            [1, 2],
            [2, 4],
            [2, 5],
            [3, 6],
            [3, 7],
            [4, 8],
            [5, 8],
            [6, 7],
            [3, 0],
            [2, 1],
            [4, 2],
            [5, 2],
            [6, 3],
            [7, 3],
            [8, 4],
            [8, 5],
            [7, 6],
        ],
        dtype=np.uint32,
    )
    WL_fast(edges, 10)


def test_bincount():
    arr = np.array([2, 1, 0, 0], dtype=np.uint32)
    out = my_bincount(arr)
    assert np.array_equal(out, [2, 1, 1])
