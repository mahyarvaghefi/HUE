"""
Homophily-based User Embedding (HUE) algorithm.

This module implements the complete HUE pipeline:
    1. Selection of core vertices (or accept pre-built networks)
    2. Formation of the extended bipartite graph
    3. Network simplification (similarity computation)
    4. Core clustering via modularity-based optimization (Surprise)
    5. Measurement of embedding features

Reference:
    Vaghefi, M. S., & Nazareth, D. L. (2021).
    Mining Online Social Networks: Deriving User Preferences through Node Embedding.
    Journal of the Association for Information Systems, 22(6).
    doi: 10.17705/1jais.00711
"""

import warnings
import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import csr_matrix, issparse

from .similarity import find_similarity_matrix


class HUE:
    """Homophily-based User Embedding for social network analysis.

    HUE embeds graph nodes into a low-dimensional space where each
    dimension corresponds to an interpretable community of interest.
    It leverages the concept of homophily and ego's alter-network
    structural similarity to produce meaningful embeddings.

    The algorithm operates on an **extended bipartite graph** — a
    structure that separates a social network into two components:

    - **Core vertices** (columns): representative nodes such as social
      media pages (@nytimes, @espn, @vogue) that embody specific topics
      or preferences.
    - **Users** (rows): ordinary users who follow the core vertices.

    Additionally, a **user-to-user network** captures friendships or
    mutual follows among users. HUE measures how similar two core
    vertices are by examining the overlap and connectivity of their
    follower ego-networks, then clusters cores into communities and
    embeds every user based on their connections to each community.

    There are two ways to use HUE:

    1. **From a single graph** (``fit`` / ``fit_transform``): provide
       one NetworkX graph and let HUE select cores and extract the
       bipartite structure automatically.
    2. **From two separate networks** (``fit_from_networks``): provide
       the bipartite adjacency (users x cores) and the user-to-user
       adjacency directly. This is the recommended approach for real
       social network data where the bipartite structure is already
       known (e.g., users following social pages).

    Parameters
    ----------
    n_cores : int or float, default=0.15
        Number of core vertices to select. If float in (0, 1), interpreted
        as a fraction of total nodes. If int >= 1, used as absolute count.
        Ignored when using ``fit_from_networks``.
    core_selection : str, default='random'
        Strategy for selecting core vertices:
        - ``'random'``: Random selection from all nodes.
        - ``'degree'``: Select nodes with highest degree centrality.
        - ``'predefined'``: Use user-provided core node IDs (pass via ``fit``).
        Ignored when using ``fit_from_networks``.
    n_clustering_iterations : int, default=50
        Number of clustering runs to find the best partition.
        The similarity matrix is converted to a weighted graph and
        community detection is applied. Multiple runs help find a
        higher-quality partition.
    partition_method : str, default='surprise'
        Quality function for modularity-based community detection:
        - ``'surprise'``: SurpriseVertexPartition (used in the paper).
        - ``'modularity'``: ModularityVertexPartition (standard modularity).
    n_jobs : int, default=2
        Number of parallel workers for similarity computation.
    chunksize : int, default=3
        Chunk size for multiprocessing.
    normalize : bool, default=True
        If True, normalize embedding rows so each row sums to 1
        (proportional community membership). If False, return raw
        weighted connectivity values.
    random_state : int or None, default=None
        Seed for reproducibility in core selection and clustering.

    Attributes
    ----------
    core_nodes_ : numpy.ndarray
        IDs of core vertices after fitting.
    similarity_matrix_ : scipy.sparse.csr_matrix
        Pairwise structural similarity matrix of core vertices. This is
        effectively a weighted graph where edge weights represent how
        similar two cores' follower ego-networks are.
    clusters_ : dict
        Mapping of cluster label to list of core vertex IDs.
    n_clusters_ : int
        Number of clusters discovered by modularity optimization.
    core_weights_ : numpy.ndarray
        Weight of each core vertex within its cluster.
    embedding_ : pandas.DataFrame
        Embedding matrix with shape (n_nodes, n_clusters).

    Examples
    --------
    **From a single graph (synthetic data, exploration):**

    >>> import networkx as nx
    >>> from hue_embedding import HUE
    >>> G = nx.random_partition_graph([100, 100, 100], 0.4, 0.05)
    >>> model = HUE(n_cores=0.15, core_selection='random', random_state=42)
    >>> embedding = model.fit_transform(G)

    **From two separate networks (real social network data):**

    >>> model = HUE(random_state=42)
    >>> embedding = model.fit_from_networks(
    ...     bipartite_adj,   # sparse matrix: users x pages
    ...     user_adj,        # sparse matrix: users x users
    ...     user_ids=user_node_list,
    ...     core_ids=page_node_list,
    ... )
    """

    def __init__(
        self,
        n_cores=0.15,
        core_selection="random",
        n_clustering_iterations=50,
        partition_method="surprise",
        n_jobs=2,
        chunksize=3,
        normalize=True,
        random_state=None,
    ):
        self.n_cores = n_cores
        self.core_selection = core_selection
        self.n_clustering_iterations = n_clustering_iterations
        self.partition_method = partition_method
        self.n_jobs = n_jobs
        self.chunksize = chunksize
        self.normalize = normalize
        self.random_state = random_state

    # ==================================================================
    # Public API — Option 1: Single graph
    # ==================================================================

    def fit(self, G, core_nodes=None):
        """Fit the HUE model to a single NetworkX graph.

        HUE will select core vertices (or use the provided ones),
        extract the extended bipartite structure, compute pairwise
        similarity, build a weighted core graph, apply modularity-based
        clustering, and compute embeddings for all nodes.

        Parameters
        ----------
        G : networkx.Graph
            Input undirected graph.
        core_nodes : array-like or None, default=None
            Pre-selected core node IDs. Required when
            ``core_selection='predefined'``.

        Returns
        -------
        self
            Fitted HUE instance.
        """
        self._validate_input(G, core_nodes)
        rng = np.random.RandomState(self.random_state)

        # Step 1: Select core vertices
        self.core_nodes_ = self._select_cores(G, core_nodes, rng)

        # Step 2: Form extended bipartite graph
        bipartite_adj, general_adj, self._user_nodes, self._core_order = (
            self._build_extended_bipartite(G)
        )

        # Steps 3–4: Similarity -> weighted graph -> clustering
        self._fit_similarity_and_cluster(bipartite_adj, general_adj, rng)

        # Step 5: Compute embedding features
        self.embedding_ = self._compute_embedding(G)

        return self

    def transform(self, G=None):
        """Return the embedding matrix.

        Parameters
        ----------
        G : networkx.Graph or None
            If None, returns the embedding computed during ``fit``.
            If provided, computes embedding for potentially new nodes
            using the fitted core clusters and weights.

        Returns
        -------
        pandas.DataFrame
            Embedding matrix with shape (n_nodes, n_clusters).
        """
        if G is None:
            return self.embedding_
        return self._compute_embedding(G)

    def fit_transform(self, G, core_nodes=None):
        """Fit and return the embedding matrix.

        Parameters
        ----------
        G : networkx.Graph
            Input undirected graph.
        core_nodes : array-like or None
            Pre-selected core node IDs.

        Returns
        -------
        pandas.DataFrame
            Embedding matrix with shape (n_nodes, n_clusters).
        """
        return self.fit(G, core_nodes).transform()

    # ==================================================================
    # Public API — Option 2: Two separate networks
    # ==================================================================

    def fit_from_networks(self, bipartite_adj, user_adj,
                          user_ids=None, core_ids=None):
        """Fit HUE from pre-built bipartite and user-to-user networks.

        This is the recommended method for real social network data where
        the bipartite structure is already known — for example, a dataset
        of users following social media pages.

        The bipartite adjacency matrix has **users as rows** and **core
        vertices (social pages) as columns**. The user adjacency matrix
        captures the friendship or mutual-follow network among users.

        After computing the ego's alter-network structural similarity
        between all pairs of cores, HUE constructs a **weighted graph**
        of cores and applies **modularity-based community detection**
        (Surprise optimization) to discover preference clusters.

        Parameters
        ----------
        bipartite_adj : scipy.sparse matrix
            Adjacency matrix of shape (n_users, n_cores). Rows are users,
            columns are core vertices (e.g., social pages).
        user_adj : scipy.sparse matrix
            Adjacency matrix of shape (n_users, n_users). Captures the
            user-to-user network (mutual follows, friendships).
        user_ids : array-like or None, default=None
            Node IDs for users (rows). If None, uses integer indices.
        core_ids : array-like or None, default=None
            Node IDs for core vertices (columns). If None, uses integer
            indices.

        Returns
        -------
        pandas.DataFrame
            Embedding matrix with shape (n_users, n_clusters).
        """
        rng = np.random.RandomState(self.random_state)

        if not issparse(bipartite_adj):
            bipartite_adj = csr_matrix(bipartite_adj)
        if not issparse(user_adj):
            user_adj = csr_matrix(user_adj)

        n_users, n_cores = bipartite_adj.shape
        if user_adj.shape[0] != n_users or user_adj.shape[1] != n_users:
            raise ValueError(
                f"user_adj shape {user_adj.shape} does not match "
                f"bipartite_adj row count {n_users}."
            )

        if user_ids is None:
            user_ids = np.arange(n_users)
        else:
            user_ids = np.asarray(user_ids)
        if core_ids is None:
            core_ids = np.arange(n_cores)
        else:
            core_ids = np.asarray(core_ids)

        self.core_nodes_ = core_ids.copy()
        self._user_nodes = user_ids.copy()
        self._core_order = core_ids.copy()

        # Filter empty rows/cols
        row_mask = np.array(bipartite_adj.sum(axis=1)).reshape(-1) > 0
        col_mask = np.array(bipartite_adj.sum(axis=0)).reshape(-1) > 0
        bip_filtered = bipartite_adj[row_mask][:, col_mask]
        usr_filtered = user_adj[row_mask][:, row_mask]

        self._user_nodes = user_ids[row_mask]
        self._core_order = core_ids[col_mask]
        self.core_nodes_ = self._core_order.copy()

        # Steps 3–4: Similarity -> weighted graph -> clustering
        self._fit_similarity_and_cluster(bip_filtered, usr_filtered, rng)

        # Step 5: Compute embedding
        self.embedding_ = self._compute_embedding_from_networks(
            bipartite_adj, user_ids, core_ids
        )

        return self.embedding_

    # ==================================================================
    # Internal methods
    # ==================================================================

    def _fit_similarity_and_cluster(self, bipartite_adj, general_adj, rng):
        """Steps 3–4: compute similarity, build weighted graph, cluster."""
        self.similarity_matrix_ = find_similarity_matrix(
            bipartite_adj, general_adj,
            n_jobs=self.n_jobs, chunksize=self.chunksize
        )
        self.clusters_, self.n_clusters_, self.core_weights_ = (
            self._cluster_cores(rng)
        )

    def _validate_input(self, G, core_nodes):
        if not isinstance(G, nx.Graph):
            raise TypeError("G must be a networkx.Graph instance.")
        if G.is_directed():
            raise ValueError(
                "HUE expects an undirected graph. "
                "Convert with G.to_undirected() first."
            )
        if self.core_selection == "predefined" and core_nodes is None:
            raise ValueError(
                "core_nodes must be provided when core_selection='predefined'."
            )

    def _select_cores(self, G, core_nodes, rng):
        """Step 1: Select core vertices."""
        nodes = np.array(list(G.nodes()))
        n_total = len(nodes)

        if self.core_selection == "predefined":
            return np.array(core_nodes)

        if isinstance(self.n_cores, float) and 0 < self.n_cores < 1:
            n = int(n_total * self.n_cores)
        else:
            n = int(self.n_cores)
        n = min(n, n_total)

        if self.core_selection == "degree":
            deg = dict(G.degree())
            sorted_nodes = sorted(deg, key=deg.get, reverse=True)
            return np.array(sorted_nodes[:n])
        else:  # random
            indices = rng.choice(n_total, size=n, replace=False)
            return nodes[indices]

    def _build_extended_bipartite(self, G):
        """Step 2: Form extended bipartite graph from a single graph."""
        core_set = set(self.core_nodes_)

        edges = pd.DataFrame(list(G.edges()), columns=["Source", "Target"])
        edges_rev = edges.rename(columns={"Source": "Target", "Target": "Source"})
        edges_all = pd.concat([edges, edges_rev], ignore_index=True)

        non_core_edges = edges_all[~edges_all["Source"].isin(core_set)]
        core_df = pd.DataFrame(list(core_set), columns=["Target"])
        bipartite_net = pd.merge(non_core_edges, core_df, on="Target")

        user_nodes = list(set(bipartite_net["Source"]))
        user_df = pd.DataFrame(user_nodes, columns=["Target"])
        user_source_df = pd.DataFrame(user_nodes, columns=["Source"])
        user_edges = pd.merge(edges_rev, user_df, on="Target")
        user_net = pd.merge(user_edges, user_source_df, on="Source")

        bip_graph = nx.from_pandas_edgelist(
            bipartite_net, "Source", "Target", create_using=nx.DiGraph()
        )
        user_graph = nx.from_pandas_edgelist(
            user_net, "Source", "Target", create_using=nx.Graph()
        )

        bip_nodes_order = np.array(bip_graph.nodes())
        bip_adj = nx.adjacency_matrix(bip_graph, bip_nodes_order)

        row_mask = np.array(bip_adj.sum(axis=1)).reshape(-1) > 0
        col_mask = np.array(bip_adj.sum(axis=0)).reshape(-1) > 0
        bip_adj = bip_adj[row_mask][:, col_mask]

        user_node_order = bip_nodes_order[row_mask]
        core_node_order = bip_nodes_order[col_mask]

        user_adj = nx.adjacency_matrix(user_graph, user_node_order)

        return bip_adj, user_adj, user_node_order, core_node_order

    def _cluster_cores(self, rng):
        """Step 4: Modularity-based community detection on the weighted core graph."""
        try:
            import leidenalg as la
            import igraph as ig
            return self._cluster_with_leidenalg(rng, la, ig)
        except ImportError:
            pass

        try:
            import louvain
            import igraph as ig
            return self._cluster_with_louvain(rng, louvain, ig)
        except ImportError:
            pass

        warnings.warn(
            "Neither 'leidenalg' nor 'louvain' package found. "
            "Falling back to sklearn spectral clustering. "
            "Install leidenalg for best results: pip install leidenalg",
            UserWarning,
        )
        return self._cluster_fallback()

    def _cluster_with_leidenalg(self, rng, la, ig):
        sim = self.similarity_matrix_.tolil()
        sim.setdiag(0)
        A = sim.toarray()
        g = ig.Graph.Adjacency((A > 0).tolist())
        g.es["weight"] = A[A.nonzero()]
        g.vs["label"] = list(self._core_order)

        if self.partition_method == "modularity":
            partition_type = la.ModularityVertexPartition
        else:
            partition_type = la.SurpriseVertexPartition

        best_quality = -1
        best_membership = None
        for _ in range(self.n_clustering_iterations):
            partition = la.find_partition(
                g, partition_type, weights="weight"
            )
            if partition.quality() > best_quality:
                best_quality = partition.quality()
                best_membership = list(partition.membership)

        return self._build_cluster_output(best_membership)

    def _cluster_with_louvain(self, rng, louvain, ig):
        sim = self.similarity_matrix_.tolil()
        sim.setdiag(0)
        A = sim.toarray()
        g = ig.Graph.Adjacency((A > 0).tolist())
        g.es["weight"] = A[A.nonzero()]
        g.vs["label"] = list(self._core_order)

        if self.partition_method == "modularity":
            partition_type = louvain.ModularityVertexPartition
        else:
            partition_type = louvain.SurpriseVertexPartition

        best_quality = -1
        best_membership = None
        for _ in range(self.n_clustering_iterations):
            partition = louvain.find_partition(
                g, partition_type, weights="weight"
            )
            if partition.quality() > best_quality:
                best_quality = partition.quality()
                best_membership = list(partition.membership)

        return self._build_cluster_output(best_membership)

    def _cluster_fallback(self):
        from sklearn.cluster import SpectralClustering
        from scipy.linalg import eigvalsh

        sim = self.similarity_matrix_.copy().tolil()
        sim.setdiag(0)
        A = sim.toarray()
        A = (A + A.T) / 2
        np.fill_diagonal(A, 0)

        L = np.diag(A.sum(axis=1)) - A
        eigenvalues = np.sort(eigvalsh(L))
        diffs = np.diff(eigenvalues[:20])
        n_clusters = int(np.argmax(diffs) + 2)
        n_clusters = max(2, min(n_clusters, len(A) // 2))

        sc = SpectralClustering(
            n_clusters=n_clusters, affinity="precomputed",
            random_state=self.random_state
        )
        labels = sc.fit_predict(np.maximum(A, 0))
        return self._build_cluster_output(list(labels))

    def _build_cluster_output(self, membership):
        membership = np.array(membership)
        n_clusters = len(set(membership))

        sim = self.similarity_matrix_.tolil()
        sim.setdiag(1)
        sim_dense = sim.toarray()

        cluster_mat = csr_matrix(
            (np.ones(len(membership)), (membership, np.arange(len(membership)))),
            shape=(n_clusters, len(membership)),
        )

        inside_weights = np.array(
            (cluster_mat.dot(csr_matrix(sim_dense))).multiply(cluster_mat).sum(axis=0)
        ).reshape(-1)
        overall_weights = np.array(sim_dense.sum(axis=0)).reshape(-1)
        weights = np.where(overall_weights > 0, inside_weights / overall_weights, 0)

        clusters = {}
        for label in range(n_clusters):
            mask = membership == label
            clusters[label] = self._core_order[mask]

        return clusters, n_clusters, weights

    def _compute_embedding(self, G):
        """Step 5: Compute embedding (single-graph mode)."""
        cluster_mat = self._build_weighted_cluster_matrix()
        n_clusters = self.n_clusters_

        all_nodes = list(G.nodes())
        full_adj = nx.adjacency_matrix(G, all_nodes).tolil()
        full_adj.setdiag(1)

        node_to_idx = {n: i for i, n in enumerate(all_nodes)}
        core_indices = [node_to_idx[c] for c in self._core_order if c in node_to_idx]
        bip_adj = full_adj[:, core_indices]

        embedding_values = bip_adj.dot(cluster_mat.T)
        if hasattr(embedding_values, "toarray"):
            embedding_values = embedding_values.toarray()
        elif hasattr(embedding_values, "A"):
            embedding_values = embedding_values.A

        embedding_values = self._normalize_embedding(embedding_values, n_clusters)

        col_names = [f"E{i}" for i in range(n_clusters)]
        embedding_df = pd.DataFrame(embedding_values, index=all_nodes, columns=col_names)
        embedding_df.index.name = "node"
        return embedding_df

    def _compute_embedding_from_networks(self, bipartite_adj, user_ids, core_ids):
        """Step 5: Compute embedding (two-network mode)."""
        cluster_mat = self._build_weighted_cluster_matrix()
        n_clusters = self.n_clusters_

        core_ids_array = np.asarray(core_ids)
        core_to_col = {c: i for i, c in enumerate(core_ids_array)}
        col_indices = [core_to_col[c] for c in self._core_order if c in core_to_col]

        bip = bipartite_adj[:, col_indices]

        embedding_values = bip.dot(cluster_mat.T)
        if hasattr(embedding_values, "toarray"):
            embedding_values = embedding_values.toarray()
        elif hasattr(embedding_values, "A"):
            embedding_values = embedding_values.A

        embedding_values = self._normalize_embedding(embedding_values, n_clusters)

        col_names = [f"E{i}" for i in range(n_clusters)]
        embedding_df = pd.DataFrame(
            embedding_values, index=np.asarray(user_ids), columns=col_names
        )
        embedding_df.index.name = "node"
        return embedding_df

    def _build_weighted_cluster_matrix(self):
        """Build the normalized weighted cluster matrix from fitted results."""
        membership = np.zeros(len(self._core_order), dtype=int)
        for label, members in self.clusters_.items():
            for m in members:
                idx = np.where(self._core_order == m)[0]
                if len(idx) > 0:
                    membership[idx[0]] = label

        weights = self.core_weights_
        cluster_mat = np.zeros((self.n_clusters_, len(self._core_order)))
        for i in range(self.n_clusters_):
            mask = membership == i
            if weights[mask].sum() > 0:
                cluster_mat[i, mask] = weights[mask]

        row_sums = cluster_mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cluster_mat = cluster_mat / row_sums
        return cluster_mat

    def _normalize_embedding(self, embedding_values, n_clusters):
        if self.normalize:
            row_sums = embedding_values.sum(axis=1, keepdims=True)
            zero_mask = (row_sums == 0).ravel()
            row_sums[row_sums == 0] = 1
            embedding_values = embedding_values / row_sums
            if zero_mask.any():
                embedding_values[zero_mask] = 1.0 / n_clusters
        return embedding_values

    def get_cluster_labels(self):
        """Return a dict mapping each core node to its cluster label.

        Returns
        -------
        dict
            {node_id: cluster_label}
        """
        labels = {}
        for label, members in self.clusters_.items():
            for m in members:
                labels[m] = label
        return labels
