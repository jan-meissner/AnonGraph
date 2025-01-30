import copy
import logging

import kmeans1d
import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs

from anonymigraph.anonymization._method_private_colors.hill_climbing import (
    hill_climbing,
)
from anonymigraph.anonymization._method_private_colors.losses import (
    undirected_p_loss,
    undirected_u_katz_loss,
)
from anonymigraph.utils import calculate_katz

logger = logging.getLogger(__name__)


def get_adjacency_dict(A_csr):
    adjacency_dict = {}
    for u in range(A_csr.shape[0]):
        neighbors = set(A_csr.indices[A_csr.indptr[u] : A_csr.indptr[u + 1]].tolist())
        adjacency_dict[u] = neighbors
    return adjacency_dict


class LocalSearchColorOptimizer:
    def __init__(
        self,
        G,
        w=1,
        alpha=0.01,
        beta=1,
        is_eager=True,
    ):
        assert not G.is_multigraph() and nx.number_of_selfloops(G) == 0, "Graph is not simple"

        self.G = G
        self.w = w
        self.is_eager = is_eager

        self.alpha = alpha
        self.beta = beta

        A_scipy_unordered = nx.adjacency_matrix(self.G, nodelist=G.nodes()).astype(np.float64)

        eigenvalues, _ = eigs(A_scipy_unordered, k=1, which="LM")  # 'LM': Largest Magnitude, tol is tolerance
        spectral_radius_G = np.abs(eigenvalues).max()
        if alpha > 1 / spectral_radius_G:
            raise ValueError(f"alpha exceedes alpha_max: alpha={alpha}, alpha_max={1 / spectral_radius_G}")

        katz_centrality_unordered = calculate_katz(A_scipy_unordered, alpha=self.alpha, beta=self.beta)

        # Reorder nodes based on Katz centrality
        sorted_indices = np.argsort(katz_centrality_unordered)

        # Order all algebraic objects ascending in katz centrality
        nodes = list(G.nodes())
        self.sorted_nodes = [nodes[i] for i in sorted_indices]
        self.A_csr = nx.adjacency_matrix(G, nodelist=self.sorted_nodes)
        self.katz_centrality = katz_centrality_unordered[sorted_indices]
        self.A_neigh_dict = get_adjacency_dict(self.A_csr)

        assert np.allclose(self.katz_centrality, calculate_katz(self.A_csr.astype(np.float64), alpha=alpha, beta=beta))

        self.colors = None

    def fit(self, seed=None, initial_colors=None):
        if initial_colors:
            initial_colors = copy.copy(initial_colors)
        else:
            initial_colors = [0] * len(self.katz_centrality)

        if self.is_eager:
            # eager is a randomized algorithm
            self.eager_fit(initial_colors, seed=seed)
        else:
            self.greedy_fit(initial_colors)

    def greedy_fit(self, initial_colors):
        colors_vec, total_loss, loss_u, loss_p = hill_climbing(
            self.A_neigh_dict,
            self.katz_centrality.tolist(),
            initial_colors,
            self.w,
            max_iter=int(1e12),  # terminates also when no improvement found
            num_nearest_clusters=2,  # higher values cause major slow downs - essentially no impact on performance
            num_k_nearest_neighbors=2,  # higher values cause major slow downs - essentially no impact on performance
            merge_split_k=2,  # higher values cause major slow downs - essentially no impact on performance
            do_swaps=True,
            do_split_and_merge_clusters=True,
        )
        logger.debug(f"Greedy Fit: total_loss = {total_loss}, loss_u = {loss_u}, loss_p = {loss_p}")
        _, colors_vec = np.unique(colors_vec, return_inverse=True)

        self.colors = {u: colors_vec[i] for i, u in enumerate(self.sorted_nodes)}

    def eager_fit(self, initial_colors, seed=None, max_phases=None):
        if seed:
            np.random.seed(seed)

        if max_phases is None:
            max_phases = len(
                self.katz_centrality
            )  # Any further phases have no effect as all n node are contained in the top m where m>n

        # Uneager Phase 1: Splits & Merge-Splits
        colors_vec, total_loss, loss_u, loss_p = hill_climbing(
            self.A_neigh_dict,
            self.katz_centrality.tolist(),
            initial_colors,
            self.w,
            max_iter=int(1e12),  # terminates also when no improvement found
            num_nearest_clusters=1,  # Doesn't matter as do_swaps=False
            num_k_nearest_neighbors=1,  # Doesn't matter as do_swaps=False
            merge_split_k=2,
            do_swaps=False,
            do_split_and_merge_clusters=True,
            eager_swaps=False,
        )
        logger.debug(f"Phase 1: total_loss = {total_loss}, loss_u = {loss_u}, loss_p = {loss_p}")

        # Eager Phase 2+: PAM-Swaps and TWO-Swaps
        num_nearest_clusters = 2
        num_k_nearest_neighbors = 2
        previous_loss = total_loss
        for phase_num in range(2, max_phases + 2):
            # Run the eager local search for each iteration, using the old colors_vec as input
            colors_vec, total_loss, loss_u, loss_p = hill_climbing(
                self.A_neigh_dict,
                self.katz_centrality.tolist(),
                colors_vec,  # Use the colors from the previous iteration
                self.w,
                max_iter=int(1e12),  # terminates also when no improvement found
                num_nearest_clusters=num_nearest_clusters,
                num_k_nearest_neighbors=num_k_nearest_neighbors,
                merge_split_k=-1,  # Doesn't matter as do_split_and_merge_clusters = False
                do_swaps=True,
                do_split_and_merge_clusters=False,
                eager_swaps=True,
            )

            logger.debug(f"Phase {phase_num}: total_loss = {total_loss}, loss_u = {loss_u}, loss_p = {loss_p}")

            if not (total_loss < previous_loss):
                break

            num_nearest_clusters += 1
            num_k_nearest_neighbors += 1

            previous_loss = total_loss

        _, colors_vec = np.unique(colors_vec, return_inverse=True)

        self.colors = {u: colors_vec[i] for i, u in enumerate(self.sorted_nodes)}


class Optimal1dColorOptimizer:
    def __init__(
        self,
        G,
        w=1,
        use_katz_utility=True,
        alpha=0.01,
        beta=1,
    ):
        assert not G.is_multigraph() and nx.number_of_selfloops(G) == 0, "Graph is not simple"

        self.G = G
        self.w = w  # linearization parameter w

        # katz centrality
        self.use_katz_utility = use_katz_utility  # if true uses katz centrality otherwise short term
        self.alpha = alpha
        self.beta = beta

        self.A_scipy = nx.adjacency_matrix(self.G, nodelist=G.nodes()).astype(np.float64)
        self.A_coo = self.A_scipy.tocoo()
        self.n = self.A_scipy.shape[0]

        self.katz_centrality = calculate_katz(self.A_scipy, alpha=self.alpha, beta=self.beta)

        self.colors = None

    def fit(self, seed=None):
        best_k = None
        best_colors = None
        best_total_loss = float("inf")
        k_max = 2
        k = 1

        while k <= k_max:
            colors, _ = kmeans1d.cluster(self.katz_centrality, k)
            total_loss, loss_U, loss_P = self.get_losses(colors)

            logger.info(f"Optimal 1d: k={k},  total_loss={total_loss:.4f}, loss_U={loss_U:.4f}, loss_P={loss_P:.4f}")
            if total_loss < best_total_loss:
                best_total_loss = total_loss
                best_k = k
                best_colors = colors

            if best_k >= 0.75 * k_max:
                k_max = int(k_max / 0.75) + 1

            k += 1

        _, best_colors = np.unique(best_colors, return_inverse=True)

        total_loss, loss_U, loss_P = self.get_losses(best_colors)
        logger.info(
            f"Optimal 1d: k_best={best_k},  total_loss={total_loss:.4f}, loss_U={loss_U:.4f}, loss_P={loss_P:.4f}"
        )

        self.colors = {u: best_colors[i] for i, u in enumerate(self.G.nodes())}

    def get_losses(self, colors):
        if self.use_katz_utility:
            loss_u = undirected_u_katz_loss(self.katz_centrality, colors)
        else:
            pass
            # loss_U = utility_short_term(AH, H)
        loss_p, _ = undirected_p_loss(self.A_coo, colors)

        total_loss = loss_u + self.w * loss_p

        return total_loss, loss_u, loss_p
