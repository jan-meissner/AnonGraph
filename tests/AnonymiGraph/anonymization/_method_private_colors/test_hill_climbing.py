import networkx as nx
import numpy as np

from anonymigraph.anonymization._method_private_colors.hill_climbing import (
    get_privacy_loss_from_states,
    get_utility_loss_from_states,
    init_cluster_states,
    update_cluster_states,
)


def dense_to_neighborhood(A):
    n = A.shape[0]
    neighborhood_dict = {}

    for i in range(n):
        neighbors = set()
        for j in range(n):
            if A[i, j] != 0:
                neighbors.add(j)
        if neighbors:
            neighborhood_dict[i] = neighbors

    return neighborhood_dict


def calculate_privacy_loss_bruteforce(A, c):
    n = A.shape[0]
    k = np.max(c) + 1

    # calcualte assignment matrix
    H = np.zeros((n, k))
    for i in range(n):
        H[i, c[i]] = 1

    m = H.T @ A @ H  # calculate edge factor
    d = A @ H  # calcualte color degree

    # iterate through all possible color pairs
    metric = 0.0
    for c1 in range(k):
        for c2 in range(k):
            if c1 <= c2:
                if m[c1, c2] == 0:
                    continue

                # for each color pair calculate a sort of degree Assortativity
                sum_ij = 0.0
                for i in range(n):
                    for j in range(n):
                        if H[i, c1] == 1 and H[j, c2] == 1:  # iterate through all nodes in c1 and c2
                            sum_ij += A[i, j] * d[i, c2] * d[j, c1]  #

                metric += (0.5 if c1 == c2 else 1) * sum_ij / m[c1, c2]

    return metric


def calculate_utility_loss_bruteforce(A, x, c):
    n = A.shape[0]
    k = np.max(c) + 1

    # calcualte assignment matrix
    H = np.zeros((n, k))
    for i in range(n):
        H[i, c[i]] = 1

    Sx = H.T @ x
    N = H.T @ np.ones_like(x)

    utility = np.sum(x**2)

    for c in range(len(Sx)):
        if N[c] > 0:
            utility -= Sx[c] ** 2 / N[c]

    return utility


def test_losses_from_clusterstates():
    seed = int(np.random.randint(1, 100000))
    print("seed:", seed)
    n = 34
    p = 0.6
    G = nx.erdos_renyi_graph(n, p, seed=seed)

    k = 43
    np.random.seed(seed)
    color = np.random.randint(0, k, size=n)
    A = nx.adjacency_matrix(G).toarray()
    x = np.random.randn(*color.shape)

    assert np.sum(np.diag(A)) == 0

    true_p_loss = calculate_privacy_loss_bruteforce(A, color)
    true_u_loss = calculate_utility_loss_bruteforce(A, x, color)

    color = color.tolist()
    x = x.tolist()

    cluster_states = init_cluster_states(dense_to_neighborhood(A), x, color)

    p_loss = get_privacy_loss_from_states(cluster_states)
    u_loss = get_utility_loss_from_states(cluster_states, x)

    num_edges = G.number_of_edges()

    assert np.allclose(u_loss, true_u_loss)
    assert np.allclose(p_loss, true_p_loss)
    assert p_loss <= num_edges


def compare_states(true_states, states):
    assert set(true_states.keys()) == set(states.keys())

    for c1 in true_states.keys():
        assert true_states[c1]["n"] == states[c1]["n"]
        assert np.allclose(true_states[c1]["Sx"], states[c1]["Sx"])

    # Check secondary key on d,m,assort,color_adj
    for c1 in true_states.keys():
        assert set(true_states[c1]["d"].keys()) == set(states[c1]["d"].keys())
        assert set(true_states[c1]["m"].keys()) == set(states[c1]["m"].keys())
        assert set(true_states[c1]["assort"].keys()) == set(states[c1]["assort"].keys())
        assert set(true_states[c1]["color_adj"].keys()) == set(states[c1]["color_adj"].keys())
        assert true_states[c1]["cluster_idxs"] == states[c1]["cluster_idxs"]

    for c1 in true_states.keys():
        assert true_states[c1]["d"] == states[c1]["d"]

    for c1 in true_states.keys():
        for c2 in true_states[c1]["m"].keys():
            assert c1 <= c2
            assert true_states[c1]["m"][c2] == states[c1]["m"][c2]

    for c1 in true_states.keys():
        for c2 in true_states[c1]["assort"].keys():
            assert c1 <= c2
            assert true_states[c1]["assort"][c2] == states[c1]["assort"][c2]

    for c1 in true_states.keys():
        for c2 in true_states[c1]["color_adj"].keys():
            adj_true = true_states[c1]["color_adj"][c2]
            adj = states[c1]["color_adj"][c2]

            assert set(adj_true.keys()) == set(adj.keys())

            for i in adj_true.keys():
                assert adj_true[i] == adj[i]

    print("States are the same!")


def test_update_cluster_states():
    # Test update_cluster_states function 1000 times:
    for seed in range(1, 1000):
        n = 34
        p = 0.6
        G = nx.erdos_renyi_graph(n, p, seed=seed)

        k = 43
        np.random.seed(seed)
        color = np.random.randint(0, k, size=n)
        A = nx.adjacency_matrix(G).toarray()
        x = np.random.randn(*color.shape)

        assert np.sum(np.diag(A)) == 0

        color = color.tolist()
        x = x.tolist()

        c_changed = list(color)
        u = 0
        old_color = c_changed[0]
        new_color = 3
        c_changed[u] = new_color

        true_cluster_states = init_cluster_states(dense_to_neighborhood(A), x, c_changed)
        cluster_states = init_cluster_states(dense_to_neighborhood(A), x, color)

        original_u_loss = get_utility_loss_from_states(cluster_states, x)
        original_p_loss = get_privacy_loss_from_states(cluster_states)

        delta_u_loss, delta_p_loss = update_cluster_states(
            cluster_states, u, old_color, new_color, dense_to_neighborhood(A), x, color
        )

        u_loss = get_utility_loss_from_states(cluster_states, x)
        true_u_loss = get_utility_loss_from_states(true_cluster_states, x)

        p_loss = get_privacy_loss_from_states(cluster_states)
        true_p_loss = get_privacy_loss_from_states(true_cluster_states)

        assert np.allclose(u_loss, true_u_loss)
        assert np.allclose(p_loss, true_p_loss)

        assert np.allclose(delta_p_loss, p_loss - original_p_loss)
        assert np.allclose(delta_u_loss, u_loss - original_u_loss)

        compare_states(true_cluster_states, cluster_states)
