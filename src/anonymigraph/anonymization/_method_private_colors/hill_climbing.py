import logging

import numpy as np
from numba import njit, types
from numba.typed import Dict

logger = logging.getLogger(__name__)

sparse_int_vector_type = types.DictType(types.int64, types.int64)
int_list_type = types.ListType(types.int64)

MOVE_NONE, MOVE_SPLIT, MOVE_PAM_SWAP, MOVE_TWO_SWAP, MOVE_SPLIT_MERGE = -1, 0, 1, 2, 3
INTER_DELTA_OBJ, INTER_CLUSTER_UPDATE, INTER_MOVE_TYPE = 0, 1, 2
interaction_type = types.Tuple((types.float64, sparse_int_vector_type, types.int64))
int_two_tuple = types.Tuple((types.int64, types.int64))


@njit
def find_k_closest(sorted_array, target, k):
    """
    Find the k-closest indices to the target in a sorted list of indices.
    Returns a slice of the sorted_list containing the k-closest indices.
    """
    n = len(sorted_array)
    idx = np.searchsorted(sorted_array, target)

    left = idx - 1
    right = idx
    count = 0

    while count < k and (left >= 0 or right < n):
        if left >= 0 and right < n:
            if (target - sorted_array[left]) <= (sorted_array[right] - target):
                left -= 1
            else:
                right += 1
        elif left >= 0:
            left -= 1
        else:
            right += 1
        count += 1

    return sorted_array[left + 1 : right]


@njit
def build_exclusion_dict(colors):
    """
    For each color in colors finds the indices of elements that have a different color.
    """
    unique_clusters = np.unique(colors)

    exclusion_dict = Dict.empty(key_type=types.int64, value_type=types.int64[:])

    for color in unique_clusters:
        # Get the indices not in cluster c
        exclusion_indices = np.where(colors != color)[0]
        exclusion_dict[color] = exclusion_indices  # Indices are already sorted

    return exclusion_dict


@njit
def get_k_nearest_neighbors(colors, k):
    """
    Finds k-closest neighbours that are not of the same color and returns their index.
    Time complexity: O(n * num_colors + n log n + n k)
    """
    n = len(colors)
    if k == 0:
        return np.full((n, k), -1)

    exclusion_dict = build_exclusion_dict(colors)
    k_nearest_neighbors = np.full((n, k), -1, dtype=np.int64)

    for i in range(n):
        cluster_label = colors[i]
        exclusion_array = exclusion_dict[cluster_label]
        # Find the k closest indices
        k_nearest = find_k_closest(exclusion_array, i, k)
        # Assign the neighbors to the output array (with padding if needed)
        k_nearest_neighbors[i, : len(k_nearest)] = k_nearest

    return k_nearest_neighbors


@njit
def compute_min_distance(cluster_idxs, n, distance_row):
    j = 0  # Initialize j outside the loop
    num_clusters = len(cluster_idxs)

    for i in range(n):
        # Shift j such that cluster_idxs[j] <= i < cluster_idxs[j+1]
        while j < num_clusters - 2 and cluster_idxs[j + 1] <= i:
            j += 1

        # Calculate distances to the lower and upper cluster indices
        lower_dist = abs(i - cluster_idxs[j])
        upper_dist = abs(cluster_idxs[j + 1] - i) if (j + 1) < num_clusters else np.inf

        # Assign the minimal distance to the distance matrix
        distance_row[i] = min(lower_dist, upper_dist)


@njit
def find_top_k_nearest_clusters(colors, unique_colors, distance_matrix, k, representants):
    """
    Finds the top k nearest clusters for each point.
    """
    n = len(colors)
    k_nearest_neighs = np.full((n, k), -1)

    # For each point, find the top k nearest clusters
    for i in range(n):
        sorted_indices = np.argsort(distance_matrix[:, i])

        assert colors[i] == unique_colors[sorted_indices[0]]

        top_k = representants[sorted_indices[1 : (k + 1)]]
        k_nearest_neighs[i, : len(top_k)] = top_k

    return k_nearest_neighs


def find_nearest_clusters(colors, k):
    """
    Finds the k-nearest unique colors for each point in the colors array. Outputs array where out[i, l] is the l-th closest cluster to point i
    (representants by an cluster representants point, i.e. colors[out[i, l]] gives the actual color)

    Time Complexity: O(n* num_unique_colors)
    """
    n = len(colors)
    if k == 0:
        return np.full((n, k), -1)
    colors = np.array(colors)
    unique_colors = np.unique(colors)
    m = len(unique_colors)

    # Store minimal distance of cluster j to point i
    distance_matrix = np.full((m, n), np.inf)

    # Compute minimal distances in O(|V|*#num_colors)
    representants = np.empty(m, np.int64)
    for c_idx, color in enumerate(unique_colors):
        cluster_idxs = np.where(colors == color)[0]
        representants[c_idx] = cluster_idxs[0]
        # save any single idx (w.l.o.g we choose 0) as the cluster representant. This makes our output relabeling invariant
        compute_min_distance(cluster_idxs, n, distance_matrix[c_idx, :])

    return find_top_k_nearest_clusters(colors, unique_colors, distance_matrix, k, representants)


def get_pam_swaps_and_two_swaps(colors, num_nearest_clusters=1, num_k_nearest_neighbors=0):
    """
    Returns two arrays:
    - nearest_clusters (n, num_nearest_clusters): For each node i returns the num_nearest_clusters nearest clusters (modulo the cluster i is inside).
                                                  Returns a cluster representant to be invariant under relabeling. Gives all possible PAM-Swaps
                                                  i.e., colors[nearest_clusters[i, 0]] is the closest cluster to node i

    - num_k_nearest_neighbors (n, num_k_nearest_neighbors): For each node i returns the index of the the num_k_nearest_neighbors nearest points to i that are not in the same cluster as i.
                                                            Gives all allowed TWO-Swaps

    Time Complexity: O(n * num_unique_colors(colors) + n * num_k_nearest_neighbors)
    """
    colors = np.array(colors, dtype=np.int64)  # ensure right type for numba
    nearest_clusters = find_nearest_clusters(colors, k=num_nearest_clusters)
    k_nearest_neighbors = get_k_nearest_neighbors(colors, k=num_k_nearest_neighbors)
    return nearest_clusters, k_nearest_neighbors


def create_interaction_move(delta_obj, cluster_update, move_type):
    return (delta_obj, cluster_update, move_type)


def get_privacy_loss_from_states(cluster_states):
    loss = 0.0
    clusters = cluster_states.keys()
    for c in clusters:
        for c_prime in clusters:
            if c <= c_prime:
                m_cc_prime = cluster_states[c]["m"].get(c_prime, 0)
                if m_cc_prime == 0:
                    continue
                assort_cc_prime = cluster_states[c]["assort"][c_prime]
                weight = 0.5 if c == c_prime else 1.0
                loss += weight * assort_cc_prime / m_cc_prime
    return loss


def get_utility_loss_from_states(cluster_states, x):
    loss = 0.0
    for xi in x:
        loss += xi**2

    for c in cluster_states.keys():
        loss -= cluster_states[c]["Sx"] ** 2 / cluster_states[c]["n"]

    return loss


def init_cluster_states(A, x, c_assignment):
    n = len(c_assignment)
    clusters = set(c_assignment)

    # cluster_states
    cluster_states = {}
    for c in clusters:
        cluster_states[c] = {
            "n": None,
            "Sx": 0,
            "d": {},  # d_ic for nodes i
            "m": {},  # m_cc' values
            "assort": {},  # assort_cc' values
            "color_adj": {},  # Color adjacency
            "cluster_idxs": set(),
        }

    # cluster_idxs
    for i, c in enumerate(c_assignment):
        cluster_states[c]["cluster_idxs"].add(i)

    for c in clusters:
        cluster_states[c]["n"] = len(cluster_states[c]["cluster_idxs"])

    # Sx
    for c in clusters:
        for i in cluster_states[c]["cluster_idxs"]:
            cluster_states[c]["Sx"] += x[i]

    # color_adj
    for c in clusters:
        cluster_states[c]["color_adj"] = {}
        for c_prime in clusters:
            adj = {}
            nodes_c = cluster_states[c]["cluster_idxs"]
            nodes_c_prime_set = cluster_states[c_prime]["cluster_idxs"]
            for i in nodes_c:
                neighbors_in_c_prime = set(j for j in A.get(i, {}) if j in nodes_c_prime_set)
                if neighbors_in_c_prime:
                    adj[i] = neighbors_in_c_prime
            if adj:
                cluster_states[c]["color_adj"][c_prime] = adj

    # d
    for c in clusters:
        for i in range(n):
            c_prime = c_assignment[i]

            if c in cluster_states[c_prime]["color_adj"]:
                adj_c1_c2 = cluster_states[c_prime]["color_adj"][c]
                if i in adj_c1_c2:
                    cluster_states[c]["d"][i] = len(adj_c1_c2[i])

    # m and assort
    for c in clusters:
        for c_prime in clusters:
            if c <= c_prime:
                # Compute m_cc' as sum of degrees to c_prime for nodes in c
                m = sum(cluster_states[c_prime]["d"].get(i, 0) for i in cluster_states[c]["cluster_idxs"])
                if m != 0:
                    cluster_states[c]["m"][c_prime] = m

                    # Compute assort_cc' using color adjacency
                    adj = cluster_states[c]["color_adj"].get(c_prime, {})
                    assort_cc_prime = 0
                    for i in adj:
                        for j in adj[i]:
                            assort_cc_prime += cluster_states[c_prime]["d"][i] * cluster_states[c]["d"][j]
                    cluster_states[c]["assort"][c_prime] = assort_cc_prime

    return cluster_states


def update_cluster_states(cluster_states, u, c_old, c_new, A, x, c_assignment):
    """
    Updates the cluster_states when a node u changes its cluster from c_old to c_new. Also updates c_assignment.
    Both updates are INPLACE. Returns numeric change in privacy and utility loss.

    Implementation is << O(d^2) where d is the average degree.

    Parameters:
    - cluster_states: current cluster states
    - u: index of the node that changes its cluster
    - c_old: old cluster of u
    - c_new: new cluster
      of u
    - A: adjacency structure (dictionary mapping node to set of neighbors)
    - c_assignment: current cluster assignment list
    """
    if c_old == c_new:
        return 0.0, 0.0

    # Preparations before r and a:
    u_neighs = A.get(u, set())
    u_neighs_cols = set(c_assignment[v] for v in u_neighs)

    # Calculate loss deltas
    delta_p_loss = 0.0
    possible_changed_color_pairs = set([(c_old, c_old), (c_new, c_new)])
    possible_changed_color_pairs.add(tuple(sorted((c_old, c_new))))
    for c_neigh in u_neighs_cols:
        possible_changed_color_pairs.add(tuple(sorted((c_neigh, c_new))))
        possible_changed_color_pairs.add(tuple(sorted((c_neigh, c_old))))

    for c1, c2 in possible_changed_color_pairs:
        if c1 in cluster_states:
            m_c1c2 = cluster_states[c1]["m"].get(c2, 0)
            if m_c1c2 != 0:
                assort_c1c2 = cluster_states[c1]["assort"][c2]
                weight = 0.5 if c1 == c2 else 1.0
                delta_p_loss -= weight * assort_c1c2 / m_c1c2

    delta_u_loss = 0.0
    if c_new in cluster_states:
        delta_u_loss += cluster_states[c_new]["Sx"] ** 2 / cluster_states[c_new]["n"]
    if c_old in cluster_states:
        delta_u_loss += cluster_states[c_old]["Sx"] ** 2 / cluster_states[c_old]["n"]

    # 5.r Removing assort:
    for v in u_neighs:
        c_v = c_assignment[v]
        # Get the adjacency structure for c_v and c_old
        adj_v_to_c_old = cluster_states[c_v]["color_adj"][c_old][v]
        for v_neigh in adj_v_to_c_old:
            if v_neigh in u_neighs and c_v == c_old and v_neigh < v:
                continue  # avoid double counting edges that are between neighs of u

            c_min, c_max = min(c_v, c_old), max(c_v, c_old)

            remove_assort_contrib = cluster_states[c_v]["d"][v_neigh] * cluster_states[c_old]["d"][v]
            cluster_states[c_min]["assort"][c_max] -= remove_assort_contrib * (2 if c_v == c_old else 1)

        if c_new in cluster_states[c_v]["color_adj"]:
            if v in cluster_states[c_v]["color_adj"][c_new]:
                adj_v_to_c_new = cluster_states[c_v]["color_adj"][c_new][v]
                for v_neigh in adj_v_to_c_new:
                    if v_neigh in u_neighs and (c_v == c_old or (c_v == c_new and v_neigh < v)):
                        continue  # avoid double counting edges that are between neighs of u

                    c_min, c_max = min(c_v, c_new), max(c_v, c_new)
                    remove_assort_contrib = cluster_states[c_v]["d"][v_neigh] * cluster_states[c_new]["d"][v]
                    cluster_states[c_min]["assort"][c_max] -= remove_assort_contrib * (2 if c_v == c_new else 1)

    # 4.r Update m: subtract d_uc' from m_cc'
    for c_v in u_neighs_cols:
        c_min, c_max = min(c_old, c_v), max(c_old, c_v)

        # factor 2 for undirected color graphs
        cluster_states[c_min]["m"][c_max] = (
            cluster_states[c_min]["m"][c_max] - (2 if c_v == c_old else 1) * cluster_states[c_v]["d"][u]
        )

        if cluster_states[c_min]["m"][c_max] == 0:
            del cluster_states[c_min]["m"][c_max]
            # Remove assort_cc' if m_cc' is 0
            if c_max in cluster_states[c_min]["assort"]:
                del cluster_states[c_min]["assort"][c_max]

    # 3.r Remove degrees
    # For every neighbor v of u, remove 1 from the color degree to c_old
    for v in u_neighs:
        # Decrease degree of v to c_old
        if v in cluster_states[c_old]["d"]:
            cluster_states[c_old]["d"][v] -= 1
            if cluster_states[c_old]["d"][v] == 0:
                del cluster_states[c_old]["d"][v]

    # 2.r Update adjacency structure
    for v in u_neighs:
        c_v = c_assignment[v]

        # removing edge (u, v) which is colored as (c_old, c_v)
        c_old_c_v_u = cluster_states[c_old]["color_adj"][c_v][u]
        c_old_c_v_u.remove(v)

        c_v_c_old_v = cluster_states[c_v]["color_adj"][c_old][v]
        c_v_c_old_v.remove(u)

        if not c_old_c_v_u:
            del cluster_states[c_old]["color_adj"][c_v][u]

        if not c_v_c_old_v:
            del cluster_states[c_v]["color_adj"][c_old][v]

        if not cluster_states[c_old]["color_adj"][c_v]:
            del cluster_states[c_old]["color_adj"][c_v]

        if c_v != c_old and not cluster_states[c_v]["color_adj"][c_old]:
            del cluster_states[c_v]["color_adj"][c_old]

    # 1.r Update cluster size, cluster idx, Sx and cluster_indices
    cluster_states[c_old]["n"] -= 1
    cluster_states[c_old]["Sx"] -= x[u]
    cluster_states[c_old]["cluster_idxs"].remove(u)
    if cluster_states[c_old]["n"] == 0:
        del cluster_states[c_old]

    # 1.a If the new color is a new cluster, add it
    if c_new not in cluster_states:
        cluster_states[c_new] = {
            "n": 0,
            "Sx": 0,
            "d": {},
            "m": {},
            "assort": {},
            "color_adj": {},
            "cluster_idxs": set(),
        }

    cluster_states[c_new]["n"] += 1
    cluster_states[c_new]["Sx"] += x[u]
    cluster_states[c_new]["cluster_idxs"].add(u)

    # 2.a Update color_adj
    for v in u_neighs:
        c_v = c_assignment[v]

        # adding edge (u, v) which is colored as (c_new, c_v)
        if c_v not in cluster_states[c_new]["color_adj"]:
            cluster_states[c_new]["color_adj"][c_v] = {}
        if u not in cluster_states[c_new]["color_adj"][c_v]:
            cluster_states[c_new]["color_adj"][c_v][u] = set()
        cluster_states[c_new]["color_adj"][c_v][u].add(v)

        # Also update color_adj for c_v and c_new
        if c_new not in cluster_states[c_v]["color_adj"]:
            cluster_states[c_v]["color_adj"][c_new] = {}
        if v not in cluster_states[c_v]["color_adj"][c_new]:
            cluster_states[c_v]["color_adj"][c_new][v] = set()
        cluster_states[c_v]["color_adj"][c_new][v].add(u)

    # 3.a Update degrees
    for v in u_neighs:
        cluster_states[c_new]["d"][v] = cluster_states[c_new]["d"].get(v, 0) + 1

    # 4.a Update m
    for c_v in u_neighs_cols:
        c_min, c_max = min(c_new, c_v), max(c_new, c_v)

        # factor 2 for undirected color graphs
        cluster_states[c_min]["m"][c_max] = (
            cluster_states[c_min]["m"].get(c_max, 0) + (2 if c_v == c_new else 1) * cluster_states[c_v]["d"][u]
        )

    # 5.a Update assort
    for v in u_neighs:
        c_v = c_assignment[v]
        # Get the adjacency structure for c_v and c_old
        if c_old in cluster_states[c_v]["color_adj"]:
            if v in cluster_states[c_v]["color_adj"][c_old]:
                adj_v_to_c_old = cluster_states[c_v]["color_adj"][c_old][v]
                for v_neigh in adj_v_to_c_old:
                    if v_neigh in u_neighs and c_v == c_old and v_neigh < v:
                        continue  # avoid double counting edges that are between neighs of u

                    c_min, c_max = min(c_v, c_old), max(c_v, c_old)

                    remove_assort_contrib = cluster_states[c_v]["d"][v_neigh] * cluster_states[c_old]["d"][v]
                    cluster_states[c_min]["assort"][c_max] += remove_assort_contrib * (2 if c_v == c_old else 1)

        adj_v_to_c_new = cluster_states[c_v]["color_adj"][c_new][v]
        for v_neigh in adj_v_to_c_new:
            if v_neigh in u_neighs and (c_v == c_old or (c_v == c_new and v_neigh < v)):
                continue  # avoid double counting edges that are between neighs of u

            c_min, c_max = min(c_v, c_new), max(c_v, c_new)
            remove_assort_contrib = cluster_states[c_v]["d"][v_neigh] * cluster_states[c_new]["d"][v]
            cluster_states[c_min]["assort"][c_max] = cluster_states[c_min]["assort"].get(
                c_max, 0
            ) + remove_assort_contrib * (2 if c_v == c_new else 1)

    # Calculate loss deltas
    for c1, c2 in possible_changed_color_pairs:
        if c1 in cluster_states:
            m_c1c2 = cluster_states[c1]["m"].get(c2, 0)
            if m_c1c2 != 0:
                assort_c1c2 = cluster_states[c1]["assort"][c2]
                weight = 0.5 if c1 == c2 else 1.0
                delta_p_loss += weight * assort_c1c2 / m_c1c2

    if c_new in cluster_states:
        delta_u_loss -= cluster_states[c_new]["Sx"] ** 2 / cluster_states[c_new]["n"]
    if c_old in cluster_states:
        delta_u_loss -= cluster_states[c_old]["Sx"] ** 2 / cluster_states[c_old]["n"]

    # Update c_assignment
    c_assignment[u] = c_new

    return delta_u_loss, delta_p_loss


def move_split(c1, curr_cluster_states, A_sparse, x, curr_clusters, new_cluster_color, w, cluster_indices=None):
    """Calculates the best way to split a cluster into two clusters
    All splits are contiguous in the katz centrality i.e. only consider splitting

    Important: cluster_indices must be sorted in ascending order of x!
    """
    # logger.info("move_split called")

    # We set cluster_indices for special case from merge_split call to this function
    if cluster_indices is None:
        cluster_indices = sorted(curr_cluster_states[c1]["cluster_idxs"])

    best_cluster_update = Dict.empty(key_type=types.int64, value_type=types.int64)
    best_delta_obj = 0

    best_loop_idx = -1  # -1 indicates no improvement found at all
    accum_u_loss = 0.0
    accum_p_loss = 0.0

    # Notice 1:
    # Reassign all nodes in cluster to new color - ordered by katz centrality - the end state is the same as the start state up to relabeling of c1 to new_cluster_color
    # Se we can continue using the curr_cluster_states as it is equivalent to the old one
    delta_obj = 0.0
    for loop_idx in range(len(cluster_indices)):
        u = cluster_indices[loop_idx]

        delta_u_loss, delta_p_loss = update_cluster_states(
            curr_cluster_states, u, c1, new_cluster_color, A_sparse, x, curr_clusters
        )
        accum_u_loss += delta_u_loss
        accum_p_loss += delta_p_loss

        delta_obj = accum_u_loss + w * accum_p_loss
        if delta_obj < best_delta_obj:
            best_delta_obj = delta_obj
            best_loop_idx = loop_idx

    # Found a split that was better than currently best
    if best_loop_idx > -1:
        best_cluster_update = Dict.empty(key_type=types.int64, value_type=types.int64)
        for j in range(best_loop_idx + 1):
            # -1 is convention for new cluster actual new cluster label will be inferred dynamically
            best_cluster_update[cluster_indices[j]] = -1

    return (
        create_interaction_move(delta_obj=best_delta_obj, cluster_update=best_cluster_update, move_type=MOVE_SPLIT),
        delta_obj,
    )


def move_pam_swap(curr_cluster_states, A_sparse, x, curr_clusters, w, nearest_clusters, eager_swaps=False):
    n, k_max = nearest_clusters.shape

    best_delta_obj = 0.0
    best_cluster_update = Dict.empty(key_type=types.int64, value_type=types.int64)

    if eager_swaps:
        # When doing Eager Swaps we want to evaluate the moves in a random order
        u_permutation = np.random.permutation(n)
        k_permutation = np.random.permutation(k_max)
    else:
        u_permutation = range(n)
        k_permutation = range(k_max)

    for u in u_permutation:
        accum_u_loss = 0.0
        accum_p_loss = 0.0
        original_u_color = curr_clusters[u]

        for k in k_permutation:
            if nearest_clusters[u, k] > -1:
                new_u_color_repr = nearest_clusters[u, k]  # Relabel invariant representant of the l cluster

                # Get new color from representant of cluster (relabel invariant)
                new_u_color = curr_clusters[new_u_color_repr]

                # Caution! Both curr_clusters and states change in place. The old color w.r.t to the current iteration is always curr_clusters[u] - but not constant
                delta_u_loss, delta_p_loss = update_cluster_states(
                    curr_cluster_states, u, curr_clusters[u], new_u_color, A_sparse, x, curr_clusters
                )

                # As our changes to u are w.r.t to the color from the last iteration we need to accumulate to get the true delta
                accum_u_loss += delta_u_loss
                accum_p_loss += delta_p_loss
                delta_obj = accum_u_loss + w * accum_p_loss

                if delta_obj < best_delta_obj:
                    best_delta_obj = delta_obj

                    best_cluster_update = Dict.empty(key_type=types.int64, value_type=types.int64)
                    best_cluster_update[u] = new_u_color_repr

        # Reset u to original color
        delta_u_loss, delta_p_loss = update_cluster_states(
            curr_cluster_states, u, curr_clusters[u], original_u_color, A_sparse, x, curr_clusters
        )

        accum_u_loss += delta_u_loss
        accum_p_loss += delta_p_loss
        delta_obj = accum_u_loss + w * accum_p_loss

        assert np.allclose(delta_obj, 0), f"{delta_obj}"

        if eager_swaps:
            # Pass back current best swap move if eager
            yield create_interaction_move(
                delta_obj=best_delta_obj, cluster_update=best_cluster_update, move_type=MOVE_PAM_SWAP
            )

    yield create_interaction_move(delta_obj=best_delta_obj, cluster_update=best_cluster_update, move_type=MOVE_PAM_SWAP)


def move_two_swap(curr_cluster_states, A_sparse, x, curr_clusters, w, k_nearest_neighbors, eager_swaps=False):
    n, k_max = k_nearest_neighbors.shape

    best_delta_obj = 0.0
    best_cluster_update = Dict.empty(key_type=types.int64, value_type=types.int64)

    if eager_swaps:
        # When doing Eager Swaps we want to evaluate the moves in a random order
        u_permutation = np.random.permutation(n)
        k_permutation = np.random.permutation(k_max)
    else:
        u_permutation = range(n)
        k_permutation = range(k_max)

    for u in u_permutation:
        for k in k_permutation:
            if k_nearest_neighbors[u, k] > -1:
                v = int(k_nearest_neighbors[u, k])  # Relabel invariant representant of the l cluster
                color_v = curr_clusters[v]
                color_u = curr_clusters[u]

                assert color_u != color_v

                # Swap
                delta_u_loss, delta_p_loss = update_cluster_states(
                    curr_cluster_states, u, color_u, color_v, A_sparse, x, curr_clusters
                )
                delta_u_loss_v, delta_p_loss_v = update_cluster_states(
                    curr_cluster_states, v, color_v, color_u, A_sparse, x, curr_clusters
                )

                # As our changes to u are w.r.t to the color from the last iteration we need to accumulate to get the true delta
                delta_obj = (delta_u_loss + delta_u_loss_v) + w * (delta_p_loss + delta_p_loss_v)

                if delta_obj < best_delta_obj:
                    best_delta_obj = delta_obj

                    best_cluster_update = Dict.empty(key_type=types.int64, value_type=types.int64)
                    best_cluster_update[u] = v
                    best_cluster_update[v] = u

                # Swap back
                delta_u_loss, delta_p_loss = update_cluster_states(
                    curr_cluster_states, u, color_v, color_u, A_sparse, x, curr_clusters
                )
                delta_u_loss_v, delta_p_loss_v = update_cluster_states(
                    curr_cluster_states, v, color_u, color_v, A_sparse, x, curr_clusters
                )

                delta_obj += (delta_u_loss + delta_u_loss_v) + w * (delta_p_loss + delta_p_loss_v)

                assert np.allclose(delta_obj, 0)

                if eager_swaps:
                    # Pass back current best swap move if eager
                    yield create_interaction_move(
                        delta_obj=best_delta_obj, cluster_update=best_cluster_update, move_type=MOVE_TWO_SWAP
                    )

    yield create_interaction_move(delta_obj=best_delta_obj, cluster_update=best_cluster_update, move_type=MOVE_TWO_SWAP)


def move_merge_split_two_clusters(c1, c2, curr_cluster_states, A_sparse, x, curr_clusters, w):
    c1_indices = sorted(curr_cluster_states[c1]["cluster_idxs"])
    c2_indices = sorted(curr_cluster_states[c2]["cluster_idxs"])

    assert c1 != c2

    # Determine which cluster is the right and which one is the left one
    if np.mean(c1_indices) < np.mean(c2_indices):
        cl = c1
        cr = c2
        cl_indices = c1_indices
        cr_indices = c2_indices
    else:
        cl = c2
        cr = c1
        cl_indices = c2_indices
        cr_indices = c1_indices

    # We can execute a merge_split via two splits where we use an already used color instead of using an truely new color for the newely split off cluster:
    # Example:
    # Let cl = 1, cr = 2
    # and the cluster vector: 1,1,1,1,2,2,2,2
    # Then executing the move split on cr with the new cluster color set as cl (this is infact not a new cluster color)
    # will lead the move split to finding the best state of:
    # 1,1,1,1,2,2,2,2 (inital state)
    # 1,1,1,1,1,2,2,2
    # 1,1,1,1,1,1,2,2
    # 1,1,1,1,1,1,1,2
    # 1,1,1,1,1,1,1,1
    best_r_move, r_delta_obj = move_split(
        cr, curr_cluster_states, A_sparse, x, curr_clusters, cl, w, cluster_indices=cr_indices
    )
    # Now we set the cl_indices to 2, i.e.
    # 2,1,1,1,1,1,1,1
    # 2,2,1,1,1,1,1,1
    # 2,2,2,1,1,1,1,1
    # 2,2,2,2,1,1,1,1 (inital state - up to relabel)
    best_l_move, l_delta_obj = move_split(
        cl, curr_cluster_states, A_sparse, x, curr_clusters, cr, w, cluster_indices=cl_indices
    )

    assert np.allclose(r_delta_obj + l_delta_obj, 0)

    if best_r_move[INTER_DELTA_OBJ] < r_delta_obj + best_l_move[INTER_DELTA_OBJ]:
        #  cluster update to any representant of the cl cluster
        for u in best_r_move[INTER_CLUSTER_UPDATE]:
            best_r_move[INTER_CLUSTER_UPDATE][u] = cl_indices[0]
        return best_r_move[INTER_DELTA_OBJ], best_r_move[INTER_CLUSTER_UPDATE]
    else:
        #  cluster update to any representant of the cr cluster
        cluster_update = Dict.empty(key_type=types.int64, value_type=types.int64)
        for u in cl_indices:
            # !! CAUTION: We need to take the inverse of best_l_move[INTER_CLUSTER_UPDATE] w.r.t cl_indices due to the relabeling we do.
            if u not in best_l_move[INTER_CLUSTER_UPDATE]:
                cluster_update[u] = cr_indices[0]

        return r_delta_obj + best_l_move[INTER_DELTA_OBJ], cluster_update


def move_merge_split(
    curr_cluster_states,
    A_sparse,
    x,
    curr_clusters,
    w,
    nearest_clusters,
    merge_split_k=2,
):
    best_move = create_interaction_move(
        np.inf, Dict.empty(key_type=types.int64, value_type=types.int64), MOVE_SPLIT_MERGE
    )
    if len(curr_cluster_states) <= 1:
        return best_move  # there is nothing to merge

    # to ensure the affinity matrix is relabel invariance, as move_merge_split_two_clusters relabels clusters
    color_representants = {}
    for c in curr_cluster_states.keys():
        # get arbitrary node as representant of the cluster
        color_representants[c] = next(iter(curr_cluster_states[c]["cluster_idxs"]))

    # Affinity[c1, c2] is the number of nodes with color c1 whos nearest node that is not in the same cluster has color c2.
    # Tells us which clusters are "close" to each other.
    affinity_matrix = {}
    for src_c, dest in zip(curr_clusters, nearest_clusters[:, 0]):
        # ! dest might be another representing than src, this however is no issue as we keep src and dest separate
        src = color_representants[src_c]
        if src not in affinity_matrix:
            affinity_matrix[src] = {}
        affinity_matrix[src][dest] = affinity_matrix[src].get(dest, 0) + 1

    for src, dest_dict in affinity_matrix.items():
        top_k_affine_dests = sorted(dest_dict.items(), key=lambda x: x[1], reverse=True)[:merge_split_k]
        for dest, _ in top_k_affine_dests:
            c1, c2 = curr_clusters[src], curr_clusters[dest]
            best_delta_obj, best_update = move_merge_split_two_clusters(
                c1, c2, curr_cluster_states, A_sparse, x, curr_clusters, w
            )

            best_merge_split_move = create_interaction_move(best_delta_obj, best_update, MOVE_SPLIT_MERGE)
            if best_merge_split_move[INTER_DELTA_OBJ] < best_move[INTER_DELTA_OBJ]:
                best_move = best_merge_split_move

    return best_move


def hill_climbing(
    A_sparse,
    x,
    inital_clusters,
    w,
    max_iter=100000000,
    num_nearest_clusters=2,
    num_k_nearest_neighbors=2,
    merge_split_k=2,
    do_swaps=False,
    do_split_and_merge_clusters=True,
    eager_swaps=False,
    seed=None,
):
    assert num_nearest_clusters >= 1, "num_nearest_clusters >= 1 is required"
    assert np.array_equal(x, np.sort(x)), "Vector x is not sorted! Assert that A_sparse is sorted in the same way as x!"

    # Due to floating point errors, we need a minimum objective improvement per iteration
    # if no move that improves the objective by this value is found the algorithm terminates
    min_delta_obj_improvement = 1e-9

    logger.debug("Getting Initial Objective")
    curr_clusters = inital_clusters
    curr_cluster_states = init_cluster_states(A_sparse, x, curr_clusters)

    curr_objective = get_utility_loss_from_states(curr_cluster_states, x) + w * get_privacy_loss_from_states(
        curr_cluster_states
    )

    logger.debug(f"init obj: {curr_objective}")

    new_cluster_color = max(curr_cluster_states.keys()) + 1

    for iteration in range(max_iter):
        # logger.info("Calculating best moves")

        # Theoretically the bottleneck of per iteration, but in practice constant pre factors make this function extremely fast compared to the rest i.e.:
        # For V = 10_000 and E = 50_000 and num_nearest_clusters = 6, num_k_nearest_neighbors = 10 takes only 20ms - neglibile compared to rest
        nearest_clusters, k_nearest_neighbors = get_pam_swaps_and_two_swaps(
            curr_clusters, num_nearest_clusters=num_nearest_clusters, num_k_nearest_neighbors=num_k_nearest_neighbors
        )

        best_move = create_interaction_move(0, Dict.empty(key_type=types.int64, value_type=types.int64), MOVE_NONE)

        # (I) Splits & Merges (cannot be executed eagerly)
        if do_split_and_merge_clusters:
            # Step 1: Split Moves (|V| calls to update_cluster_states)
            for c1 in list(curr_cluster_states.keys()):
                best_split_move, relable_delta_obj = move_split(
                    c1, curr_cluster_states, A_sparse, x, curr_clusters, new_cluster_color, w
                )

                # Assert that the objective didn't change after checking move splits.
                assert np.allclose(relable_delta_obj, 0.0), f"{relable_delta_obj}"

                # As splitting always relabels the c1 cluster to new_cluster_color we need to increment new_cluster_color by 1 to ensure we get a new cluster
                new_cluster_color += 1

                if best_split_move[INTER_DELTA_OBJ] < best_move[INTER_DELTA_OBJ]:
                    best_move = best_split_move

            # Step 2: Merge Split Moves
            best_merge_split_move = move_merge_split(
                curr_cluster_states,
                A_sparse,
                x,
                curr_clusters,
                w,
                nearest_clusters,
                merge_split_k=merge_split_k,
            )
            if best_merge_split_move[INTER_DELTA_OBJ] < best_move[INTER_DELTA_OBJ]:
                best_move = best_merge_split_move

        # (II) PAM-Swaps and TWO-Swaps (can be executed eagerly)
        if do_swaps:
            if eager_swaps:
                generator_pam_moves = move_pam_swap(
                    curr_cluster_states, A_sparse, x, curr_clusters, w, nearest_clusters, eager_swaps=True
                )
                generator_two_swap_moves = move_two_swap(
                    curr_cluster_states, A_sparse, x, curr_clusters, w, k_nearest_neighbors, eager_swaps=True
                )

                while True:
                    # Get Next PAM Move
                    next_pam_move = next(generator_pam_moves, None)
                    if next_pam_move is not None and next_pam_move[INTER_DELTA_OBJ] < -min_delta_obj_improvement:
                        best_move = next_pam_move
                        break

                    # Get Next TWO Move
                    next_two_move = next(generator_two_swap_moves, None)
                    if next_two_move is not None and next_two_move[INTER_DELTA_OBJ] < -min_delta_obj_improvement:
                        best_move = next_two_move
                        break

                    if next_pam_move is None and next_two_move is None:
                        break

            else:
                # move_pam_swap
                best_pam_move = next(
                    move_pam_swap(
                        curr_cluster_states, A_sparse, x, curr_clusters, w, nearest_clusters, eager_swaps=False
                    )
                )
                if best_pam_move[INTER_DELTA_OBJ] < best_move[INTER_DELTA_OBJ]:
                    best_move = best_pam_move

                # move_two_swap
                best_two_swap_move = next(
                    move_two_swap(
                        curr_cluster_states, A_sparse, x, curr_clusters, w, k_nearest_neighbors, eager_swaps=False
                    )
                )
                if best_two_swap_move[INTER_DELTA_OBJ] < best_move[INTER_DELTA_OBJ]:
                    best_move = best_two_swap_move

        best_delta_obj = best_move[INTER_DELTA_OBJ]
        best_cluster_update = best_move[INTER_CLUSTER_UPDATE]
        best_move_type = best_move[INTER_MOVE_TYPE]

        if best_delta_obj >= -min_delta_obj_improvement:
            break  # No improvement found, stop the hill climbing

        ##################
        # UPDATE STEP
        ##################

        # Convert representants to color (we need to do this first, as we might otherwise change the color of a representant)
        for u in best_cluster_update:
            new_u_color = curr_clusters[best_cluster_update[u]] if best_cluster_update[u] > -1 else new_cluster_color
            best_cluster_update[u] = new_u_color

        # Update the cluster state
        for u in best_cluster_update:
            update_cluster_states(
                curr_cluster_states, u, curr_clusters[u], best_cluster_update[u], A_sparse, x, curr_clusters
            )
        new_cluster_color += 1  # In case we used a new color in best_cluter_update

        # Update curr_objective
        curr_objective += best_delta_obj

        move_type_str = {
            MOVE_TWO_SWAP: "two-swap",
            MOVE_SPLIT: "split",
            MOVE_PAM_SWAP: "pam-swap",
            MOVE_SPLIT_MERGE: "merge-split",
        }
        logger.debug(
            f"LOG: Iteration {iteration}, "
            f"Obj: {curr_objective:.4f}, "
            f"Obj Delta: {best_delta_obj:.4f}, "
            f"Num Clusters: {len(curr_cluster_states.keys())}, "
            f"Move Type: {move_type_str.get(best_move_type, 'ERROR')}"
        )

        assert curr_objective >= 0, "The objective must be positive at all times."

    curr_cluster_states = init_cluster_states(A_sparse, x, curr_clusters)
    loss_u = get_utility_loss_from_states(curr_cluster_states, x)
    loss_p = get_privacy_loss_from_states(curr_cluster_states)
    total_loss = loss_u + w * loss_p

    assert np.allclose(total_loss, curr_objective), "The final objective is not close to the exact value"

    return curr_clusters, total_loss, loss_u, loss_p
