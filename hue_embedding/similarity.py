"""
Ego's Alter-Network Structural Similarity computation.

This module implements the core similarity metric used in HUE:
the ego's alter-network structural similarity, which captures
second-order proximity by considering both common neighbors
and the connectivity pattern between them.

Reference:
    Johnson, M. (1985). Relating metrics, lines and variables
    defined on graphs to problems in medicinal chemistry.
"""

import numpy as np
import scipy
from scipy.sparse import csr_matrix
from multiprocessing import Pool
from functools import partial


def _num_edges_in_ego_alter_network(target_node_index, bipartite_adj, general_adj):
    """Count edges among neighbors of a target node in the general network.

    For a given target (core) node, find all its neighbors in the bipartite
    graph and count the edges between those neighbors in the general
    (user-to-user) network.

    Parameters
    ----------
    target_node_index : int
        Index of the target (core) node.
    bipartite_adj : scipy.sparse matrix
        Adjacency matrix of the bipartite graph (users x cores).
    general_adj : scipy.sparse matrix
        Adjacency matrix of the general (user-to-user) network.

    Returns
    -------
    int
        Number of edges among the target's neighbors.
    """
    neighbors_idx = bipartite_adj.T[target_node_index, :].nonzero()
    neighbors = neighbors_idx[-1] if len(neighbors_idx) > 1 else neighbors_idx[0]
    return np.int32(np.sum(general_adj[:, neighbors][neighbors, :]) / 2)


def _ego_alter_edge_similarity(target_node_index, bipartite_adj, general_adj):
    """Compute the edge-based similarity between a target node's ego alter network and all others.

    For each target (core) node, this measures how many edges are shared
    between its ego's alter network and those of all other target nodes.

    Parameters
    ----------
    target_node_index : int
        Index of the target (core) node.
    bipartite_adj : scipy.sparse matrix
        Adjacency matrix of the bipartite graph (users x cores).
    general_adj : scipy.sparse matrix
        Adjacency matrix of the general (user-to-user) network.

    Returns
    -------
    scipy.sparse.csr_matrix
        Row vector of edge similarities with all other target nodes.
    """
    row = bipartite_adj.T[target_node_index, :]
    if hasattr(row, 'toarray'):
        mask = (row.toarray() > 0).ravel()
    elif hasattr(row, 'A'):
        mask = (row.A > 0).ravel()
    else:
        mask = (np.asarray(row.todense()) > 0).ravel()
    ego_net_edge_similarity = (
        np.sum(
            bipartite_adj.T.multiply(mask)
            .dot(general_adj)
            .multiply(bipartite_adj.T.multiply(mask)),
            axis=1,
        ).T
        / 2
    ).astype(int)
    return csr_matrix(ego_net_edge_similarity)


def find_similarity_matrix(bipartite_adj, general_adj, n_jobs=2, chunksize=3):
    """Compute the ego's alter-network structural similarity matrix for all core vertices.

    This is the main similarity computation in HUE. For each pair of core vertices,
    it measures how similar their ego's alter networks are by considering both:

    1. **Common neighbors** (node similarity): How many users follow both cores.
    2. **Shared edges** (edge similarity): How many connections exist between
       their shared neighbors.

    The similarity function is:

        Sim(G̃_A, G̃_B) = (|V(G̃_A, G̃_B)| + |E(G̃_A, G̃_B)|)^2
                          / ((|V(G̃_A)| + |E(G̃_A)|) * (|V(G̃_B)| + |E(G̃_B)|))

    Parameters
    ----------
    bipartite_adj : scipy.sparse matrix
        Adjacency matrix of the bipartite graph with shape (num_users, num_cores).
        Rows represent users and columns represent core vertices.
    general_adj : scipy.sparse matrix
        Adjacency matrix of the general (user-to-user) network with shape
        (num_users, num_users).
    n_jobs : int, default=2
        Number of parallel processes for computation.
    chunksize : int, default=3
        Chunk size for multiprocessing pool.

    Returns
    -------
    scipy.sparse.csr_matrix
        Similarity matrix of shape (num_cores, num_cores) where entry (i, j)
        represents the ego's alter-network structural similarity between
        core vertices i and j. Values range from 0 to 1.

    Examples
    --------
    >>> from hue_embedding.similarity import find_similarity_matrix
    >>> sim_mat = find_similarity_matrix(bipartite_adj, user_adj, n_jobs=4)
    """
    num_cores = bipartite_adj.shape[1]

    # Count neighbors for each core vertex
    target_neighbors_node_count = np.array(bipartite_adj.sum(axis=0)).reshape(-1)

    # Count edges in each core's ego alter network (parallelized)
    with Pool(processes=n_jobs) as pool:
        pool_fn = partial(
            _num_edges_in_ego_alter_network,
            bipartite_adj=bipartite_adj,
            general_adj=general_adj,
        )
        target_neighbors_edge_count = np.array(
            pool.map(pool_fn, range(num_cores), chunksize)
        )

    # Node similarity: number of common neighbors between core pairs
    node_sim_mat = bipartite_adj.T.dot(bipartite_adj).tocsr()

    # Edge similarity: number of shared edges between ego alter networks
    edge_sim_mat = csr_matrix((num_cores, num_cores), dtype=int).tolil()

    with Pool(processes=n_jobs) as pool:
        pool_fn = partial(
            _ego_alter_edge_similarity,
            bipartite_adj=bipartite_adj,
            general_adj=general_adj,
        )
        results = pool.map(pool_fn, range(num_cores), chunksize)

    for idx, row in enumerate(results):
        edge_sim_mat[idx, :] = row
    edge_sim_mat = edge_sim_mat.tocsr()
    del results

    # Combined similarity: nodes + edges
    combined_sim_mat = node_sim_mat + edge_sim_mat
    sum_node_edge = (target_neighbors_node_count + target_neighbors_edge_count).reshape(-1, 1)

    # Denominator: product of totals, masked where no similarity exists
    denom = sum_node_edge.dot(sum_node_edge.T)
    denom[np.asarray(combined_sim_mat.todense()) == 0] = 0
    denom = csr_matrix(denom)

    # Final similarity: Sim = (common_nodes + common_edges)^2 / (total_A * total_B)
    # Computed in log-space for numerical stability
    denom_log = np.log(denom.data)
    combined_sim_mat.data = 2 * np.log(combined_sim_mat.data) - denom_log
    combined_sim_mat.data = np.exp(combined_sim_mat.data)

    return combined_sim_mat
