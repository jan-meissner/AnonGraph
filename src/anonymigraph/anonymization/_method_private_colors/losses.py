import numpy as np
from scipy.sparse import coo_matrix


def undirected_u_short_term(A, colors):
    """
    Calculates the short term utility loss.
    """
    if not isinstance(A, coo_matrix):
        A_coo = A.tocoo()

    _, relabeled_colors = np.unique(colors, return_inverse=True)
    num_colors = relabeled_colors.max() + 1

    # A_coo.col and A_coo.row contain the source and destination nodes of all edges in A (reverse too as undirected)
    # Get colors of source (cols) and destination (rows) edges
    labeled_cols = relabeled_colors[A_coo.col]

    # Get color degree efficiently by creating n x num_colors:
    # For each edge (u,v) \in E we add (u, color(v)) to a new multiset.
    # Use coo_matrix as a efficient multiset representation - giving us the multiplicities (color degrees) after summing duplicates out.
    color_degree_coo = coo_matrix((A_coo.data, (A_coo.row, labeled_cols)), shape=(A_coo.shape[0], num_colors))
    color_degree_coo.sum_duplicates()

    # for each (u, c_v) (source, color) pair we map it to (color(u), c_v) = (c_u, c_v) to a multiset and again count all occurenes. Giving us m
    labeled_deg_row = relabeled_colors[color_degree_coo.row]
    m_coo = coo_matrix((color_degree_coo.data, (labeled_deg_row, color_degree_coo.col)), shape=(num_colors, num_colors))
    # tocsr() below calls sum_duplicates() internally for m

    # Convert coo to csr to allow for fancy indexing
    n = np.bincount(relabeled_colors)
    AH = color_degree_coo.tocsr()
    HtAH = m_coo.todense()

    loss_U = np.sum(AH.data**2) - (HtAH / n).sum()
    return loss_U


def undirected_u_katz_loss(katz_centrality, colors):
    """
    Calculates the Katz-based utility loss.
    """
    _, relabeled_colors = np.unique(colors, return_inverse=True)

    n = np.bincount(relabeled_colors)
    Ht_x = np.bincount(relabeled_colors, weights=katz_centrality)
    Stx_sqr = Ht_x**2

    loss_U = np.sum(katz_centrality**2) - np.sum(Stx_sqr / n)
    return loss_U


def undirected_p_loss(A, colors, return_expected_num_parallel_edges=False):
    """
    Calculates the privacy loss based on the majorant  of the edge intersection.
    """
    if not isinstance(A, coo_matrix):
        A_coo = A.tocoo()

    _, relabeled_colors = np.unique(colors, return_inverse=True)
    num_colors = relabeled_colors.max() + 1

    # A_coo.col and A_coo.row contain the source and destination nodes of all edges in A (reverse too as undirected)
    # Get colors of source (cols) and destination (rows) edges
    labeled_cols = relabeled_colors[A_coo.col]
    labeled_rows = relabeled_colors[A_coo.row]

    # Get color degree efficiently by creating n x num_colors:
    # For each edge (u,v) \in E we add (u, color(v)) to a new multiset.
    # Use coo_matrix as a efficient multiset representation - giving us the multiplicities (color degrees) after summing duplicates out.
    color_degree_coo = coo_matrix((A_coo.data, (A_coo.row, labeled_cols)), shape=(A_coo.shape[0], num_colors))
    color_degree_coo.sum_duplicates()

    # for each (u, c_v) (source, color) pair we map it to (color(u), c_v) = (c_u, c_v) to a multiset and again count all occurenes. Giving us m
    labeled_deg_row = relabeled_colors[color_degree_coo.row]
    m_coo = coo_matrix((color_degree_coo.data, (labeled_deg_row, color_degree_coo.col)), shape=(num_colors, num_colors))
    # tocsr() below calls sum_duplicates() internally for m

    # Convert coo to csr to allow for fancy indexing
    color_degree_csr = color_degree_coo.tocsr()
    m_csr = m_coo.tocsr()

    d_row = color_degree_csr[A_coo.row, labeled_cols].A1
    d_col = color_degree_csr[A_coo.col, labeled_rows].A1
    m = m_csr[labeled_cols, labeled_rows].A1

    p_loss = np.sum(d_row * d_col / m) / 2.0

    if return_expected_num_parallel_edges:
        # In the configuration model the expected multiplicity mult(row,col) is approx. sampled from Binom(d_row, d_col / m)
        # If we want to find expected number of edges that have atleast one parallel edge - we calculate:
        # E(mult(row,col)) - 1 * P(mult(row,col) = 1) = d_row * p * (1 - (1-p)^(d_row - 1))
        # where p = d_col / m
        # This number tells us in relation to number of edges approx. how bad our approximation is.
        p = d_col / m
        expected_number_multi_edges = np.sum(d_row * p * (1 - (1 - p) ** (d_row - 1)))

        return p_loss, expected_number_multi_edges
    else:
        return p_loss, None
