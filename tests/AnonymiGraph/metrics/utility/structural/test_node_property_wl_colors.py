import numpy as np

from anonymigraph.metrics.utility.structural.node_property_wl_colors import (
    _labels_to_prob_dists,
)


def test_labels_to_prob_dists_basic_functionality():
    p_labels = np.array([1, 2, 2, 3])
    q_labels = np.array([2, 3, 4])
    p_dist, q_dist = _labels_to_prob_dists(p_labels, q_labels)
    assert np.allclose(p_dist, np.array([0.25, 0.5, 0.25, 0.0]))
    assert np.allclose(q_dist, np.array([0.0, 1 / 3, 1 / 3, 1 / 3]))


def test_labels_to_prob_dists_empty_input():
    p_labels = np.array([])
    q_labels = np.array([])
    p_dist, q_dist = _labels_to_prob_dists(p_labels, q_labels)
    assert p_dist.size == 0
    assert q_dist.size == 0


def test_labels_to_prob_dists_single_label():
    p_labels = np.array([1])
    q_labels = np.array([1])
    p_dist, q_dist = _labels_to_prob_dists(p_labels, q_labels)
    assert np.allclose(p_dist, [1.0])
    assert np.allclose(q_dist, [1.0])


def test_labels_to_prob_dists_identical_labels():
    p_labels = np.array([1, 1, 1])
    q_labels = np.array([1, 1])
    p_dist, q_dist = _labels_to_prob_dists(p_labels, q_labels)
    assert np.allclose(p_dist, [1.0])
    assert np.allclose(q_dist, [1.0])


def test_labels_to_prob_dists_disjoint_labels():
    p_labels = np.array([1, 2])
    q_labels = np.array([3, 4])
    p_dist, q_dist = _labels_to_prob_dists(p_labels, q_labels)
    assert np.allclose(p_dist, [0.5, 0.5, 0.0, 0.0])
    assert np.allclose(q_dist, [0.0, 0.0, 0.5, 0.5])


def test_labels_to_prob_dists_large_input():
    np.random.seed(10)
    p_labels = np.random.randint(0, 100, 1000)
    q_labels = np.random.randint(0, 100, 1000)
    p_dist, q_dist = _labels_to_prob_dists(p_labels, q_labels)
    assert len(p_dist) == 100  # Assuming 0-99 inclusive are all present
    assert len(q_dist) == 100
    assert np.allclose(np.sum(p_dist), 1)
    assert np.allclose(np.sum(q_dist), 1)
