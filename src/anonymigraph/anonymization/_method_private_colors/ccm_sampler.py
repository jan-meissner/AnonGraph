import networkx as nx
import numpy as np
from scipy.sparse import coo_matrix

from anonymigraph.anonymization._external.nest_model._rewire import _rewire
from anonymigraph.utils import _validate_input_graph


class CCMSampler:
    def __init__(self, G, colors, use_pseudograph_to_sample=False, r=10):
        """
        Samples from the CCM as defined in the thesis.

        Args:
            G (nx.Graph): The input graph.
            colors (dict): The coloring.
            use_configuration_to_sample (bool): Whether to use the pseudograph sampling approach or the markov chain.
            r (int): The number of rewiring steps for the Markov chain sampling.
        """

        _validate_input_graph(G)

        self.r = r
        self.use_configuration_to_sample = use_pseudograph_to_sample
        self.G = G
        self.A_csr = nx.adjacency_matrix(self.G).astype(np.float64)
        self.A_coo = self.A_csr.tocoo()

        self.color_arr = np.array([colors[node] for node in list(G.nodes())], dtype=np.uint32)
        _, self.color_arr = np.unique(self.color_arr, return_inverse=True)
        self.unique_colors = np.unique(self.color_arr)

        self.edges = np.array(G.edges(), dtype=np.uint32)

        self.num_colors = len(self.unique_colors)
        self.col_colors = self.color_arr[self.A_coo.col]
        color_degree_coo = coo_matrix(
            (self.A_coo.data, (self.A_coo.row, self.col_colors)), shape=(self.A_coo.shape[0], self.num_colors)
        )
        self.color_degree = color_degree_coo.tocsr()  # tocsr() also sums duplicates

    def sample(self, seed=None):
        """
        Samples a graph from the CCM.

        Args:
            seed (int): Random seed.

        Returns:
            tuple: Adjacency Matrix of G and G'.
        """
        if self.use_configuration_to_sample:
            return self.sample_configuration_model()
        else:
            return self.sample_markov_chain(random_seed=seed)

    def sample_markov_chain(self, random_seed: int = None) -> nx.Graph:
        edges_rewired = _rewire(
            self.edges, self.color_arr.reshape(1, -1), r=self.r, parallel=False, random_seed=random_seed
        )

        I = edges_rewired[:, 0]
        J = edges_rewired[:, 1]
        I_sym = np.hstack([I, J])
        J_sym = np.hstack([J, I])
        A_csr_sample = coo_matrix((np.ones_like(I_sym), (I_sym, J_sym))).tocsr().astype(np.float64)

        return self.A_csr, A_csr_sample

    def sample_configuration_model(self, seed=None):
        merged_row = np.zeros((0,))
        merged_col = np.zeros((0,))
        merged_data = np.zeros((0,))

        for colorA in self.unique_colors:
            color_indices_A = np.where(colorA == self.color_arr)[0]
            for colorB in self.unique_colors:
                if colorA == colorB:
                    color_degrees_sub = self.color_degree[color_indices_A, colorA].astype(int).toarray().ravel()

                    G_color_subgraph = nx.configuration_model(color_degrees_sub)

                    A_sub = nx.adjacency_matrix(G_color_subgraph).tocoo()

                    merged_row = np.concatenate([merged_row, color_indices_A[A_sub.row]])
                    merged_col = np.concatenate([merged_col, color_indices_A[A_sub.col]])
                    merged_data = np.concatenate([merged_data, A_sub.data])

                elif colorA < colorB:
                    color_indices_B = np.where(colorB == self.color_arr)[0]

                    color_degrees_from_A_to_B = self.color_degree[color_indices_A, colorB].astype(int).toarray().ravel()
                    color_degrees_from_B_to_A = self.color_degree[color_indices_B, colorA].astype(int).toarray().ravel()

                    G_color_subgraph = nx.bipartite.configuration_model(
                        color_degrees_from_A_to_B, color_degrees_from_B_to_A
                    )

                    # G_color_subgraph node list order is np.concat(color_indices_A, color_indices_B), that is first A then B
                    A_sub = nx.adjacency_matrix(G_color_subgraph).tocoo()

                    node_indices = np.concat([color_indices_A, color_indices_B])
                    merged_row = np.concatenate([merged_row, node_indices[A_sub.row]])
                    merged_col = np.concatenate([merged_col, node_indices[A_sub.col]])
                    merged_data = np.concatenate([merged_data, A_sub.data])

        # For the color degree to be preserved we need self loops to be counted twice (as the edges have two stubs to the self node)
        merged_data[merged_row == merged_col] *= 2
        A_coo_sample = coo_matrix((merged_data, (merged_row, merged_col)), shape=self.A_coo.shape)

        col_colors_sample = self.color_arr[A_coo_sample.col]
        color_degree_sample = coo_matrix(
            (A_coo_sample.data, (A_coo_sample.row, col_colors_sample)), shape=(A_coo_sample.shape[0], self.num_colors)
        )
        color_degree_sample = color_degree_sample.tocsr()
        assert (color_degree_sample != self.color_degree).nnz == 0  # assert that AH = A'H

        return self.A_csr, A_coo_sample.tocsr().astype(np.float64)
