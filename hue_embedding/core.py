"""
Homophily-based User Embedding (HUE) algorithm.

This module implements the complete HUE pipeline:
    1. Selection of core vertices
    2. Formation of the extended bipartite graph
    3. Network simplification (similarity computation)
    4. Core clustering (Louvain with Surprise optimization)
    5. Measurement of embedding features

Reference:
    Sharif Vaghefi, M., & Nazareth, D. L. (2021).
    Mining Online Social Networks: Deriving User Preferences through Node Embedding.
    Journal of the Association for Information Systems, 22(6).
    doi: 10.17705/1jais.00711
"""

import warnings
import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import csr_matrix

from .similarity import find_similarity_matrix


class HUE:
    """Homophily-based User Embedding for social network analysis.

    HUE embeds graph nodes into a low-dimensional space where each
    dimension corresponds to an interpretable community of interest.
    It leverages the concept of homophily and ego's alter-network
    structural similarity to produce meaningful embeddings.

    Parameters
    ----------
    n_cores : int or float, default=0.15
        Number of core vertices to select. If float in (0, 1), interpreted
        as a fraction of total nodes. If int >= 1, used as absolute count.
    core_selection : str, default='random'
        Strategy for selecting core vertices:
        - ``'random'``: Random selection from all nodes.
        - ``'degree'``: Select nodes with highest degree centrality.
        - ``'predefined'``: Use user-provided core node IDs (pass via ``fit``).
    n_clustering_iterations : int, default=50
        Number of Louvain clustering runs to find the best partition.
    n_jobs : int, default=2
        Number of parallel workers for similarity computation.
    chunksize : int, default=3
        Chunk size for multiprocessing.
    random_state : int or None, default=None
        Seed for reproducibility in core selection and clustering.

    Attributes
    ----------
    core_nodes_ : numpy.ndarray
        Indices of selected core vertices after fitting.
    similarity_matrix_ : scipy.sparse.csr_matrix
        Pairwise similarity matrix of core vertices.
    clusters_ : dict
        Mapping of cluster label to list of core vertex indices.
    n_clusters_ : int
        Number of clusters discovered.
    core_weights_ : numpy.ndarray
        Weight of each core vertex within its cluster.
    embedding_ : pandas.DataFrame
        Embedding matrix with shape (n_nodes, n_clusters).

    Examples
    --------
    >>> import networkx as nx
    >>> from hue_embedding import HUE
    >>> G = nx.random_partition_graph([100, 100, 100], 0.4, 0.05)
    >>> model = HUE(n_cores=0.15, core_selection='random', random_state=42)
    >>> embedding = model.fit_transform(G)
    >>> print(embedding.shape)  # (300, n_clusters)
    """

    def __init__(
        self,
        n_cores=0.15,
        core_selection="random",
        n_clustering_iterations=50,
        n_jobs=2,
        chunksize=3,
        random_state=None,
    ):
        self.n_cores = n_cores
        self.core_selection = core_selection
        self.n_clustering_iterations = n_clustering_iterations
        self.n_jobs = n_jobs
        self.chunksize = chunksize
        self.random_state = random_state

    def fit(self, G, core_nodes=None):
        """Fit the HUE model to a graph.

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

        # Step 3: Network simplification (compute similarity matrix)
        self.similarity_matrix_ = find_similarity_matrix(
            bipartite_adj, general_adj, n_jobs=self.n_jobs, chunksize=self.chunksize
        )

        # Step 4: Core clustering
        self.clusters_, self.n_clusters_, self.core_weights_ = self._cluster_cores(rng)

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

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

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

        # Determine count
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
        """Step 2: Form extended bipartite graph.

        Returns bipartite adjacency (users x cores) and user-to-user adjacency.
        """
        core_set = set(self.core_nodes_)
        all_nodes = list(G.nodes())

        # Build edge list
        edges = pd.DataFrame(list(G.edges()), columns=["Source", "Target"])
        edges_rev = edges.rename(columns={"Source": "Target", "Target": "Source"})
        edges_all = pd.concat([edges, edges_rev], ignore_index=True)

        # Bipartite edges: non-core -> core
        non_core_edges = edges_all[~edges_all["Source"].isin(core_set)]
        core_df = pd.DataFrame(list(core_set), columns=["Target"])
        bipartite_net = pd.merge(non_core_edges, core_df, on="Target")

        # User-to-user edges among non-core nodes
        user_nodes = list(set(bipartite_net["Source"]))
        user_df = pd.DataFrame(user_nodes, columns=["Target"])
        user_source_df = pd.DataFrame(user_nodes, columns=["Source"])
        user_edges = pd.merge(edges_rev, user_df, on="Target")
        user_net = pd.merge(user_edges, user_source_df, on="Source")

        # Build networkx subgraphs
        bip_graph = nx.from_pandas_edgelist(
            bipartite_net, "Source", "Target", create_using=nx.DiGraph()
        )
        user_graph = nx.from_pandas_edgelist(
            user_net, "Source", "Target", create_using=nx.Graph()
        )

        # Adjacency matrices
        bip_nodes_order = np.array(bip_graph.nodes())
        bip_adj = nx.adjacency_matrix(bip_graph, bip_nodes_order)

        # Filter zero-sum rows/cols
        row_mask = np.array(bip_adj.sum(axis=1)).reshape(-1) > 0
        col_mask = np.array(bip_adj.sum(axis=0)).reshape(-1) > 0
        bip_adj = bip_adj[row_mask][:, col_mask]

        user_node_order = bip_nodes_order[row_mask]
        core_node_order = bip_nodes_order[col_mask]

        # User adjacency aligned to bipartite rows
        user_adj = nx.adjacency_matrix(user_graph, user_node_order)

        return bip_adj, user_adj, user_node_order, core_node_order

    def _cluster_cores(self, rng):
        """Step 4: Cluster core vertices using Louvain with Surprise."""
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

        # Fallback: spectral clustering via sklearn
        warnings.warn(
            "Neither 'leidenalg' nor 'louvain' package found. "
            "Falling back to sklearn spectral clustering. "
            "Install leidenalg for best results: pip install leidenalg",
            UserWarning,
        )
        return self._cluster_fallback()

    def _cluster_with_leidenalg(self, rng, la, ig):
        """Cluster using leidenalg (preferred)."""
        sim = self.similarity_matrix_.tolil()
        sim.setdiag(0)
        A = sim.toarray()
        g = ig.Graph.Adjacency((A > 0).tolist())
        g.es["weight"] = A[A.nonzero()]
        g.vs["label"] = list(self._core_order)

        best_quality = -1
        best_membership = None
        for _ in range(self.n_clustering_iterations):
            partition = la.find_partition(
                g, la.SurpriseVertexPartition, weights="weight"
            )
            if partition.quality() > best_quality:
                best_quality = partition.quality()
                best_membership = list(partition.membership)

        return self._build_cluster_output(best_membership)

    def _cluster_with_louvain(self, rng, louvain, ig):
        """Cluster using louvain package."""
        sim = self.similarity_matrix_.tolil()
        sim.setdiag(0)
        A = sim.toarray()
        g = ig.Graph.Adjacency((A > 0).tolist())
        g.es["weight"] = A[A.nonzero()]
        g.vs["label"] = list(self._core_order)

        best_quality = -1
        best_membership = None
        for _ in range(self.n_clustering_iterations):
            partition = louvain.find_partition(
                g, louvain.SurpriseVertexPartition, weights="weight"
            )
            if partition.quality() > best_quality:
                best_quality = partition.quality()
                best_membership = list(partition.membership)

        return self._build_cluster_output(best_membership)

    def _cluster_fallback(self):
        """Fallback clustering using sklearn spectral clustering."""
        from sklearn.cluster import SpectralClustering

        sim = self.similarity_matrix_.copy().tolil()
        sim.setdiag(0)
        A = sim.toarray()
        A = (A + A.T) / 2  # ensure symmetry
        np.fill_diagonal(A, 0)

        # Estimate number of clusters from eigenvalues
        from scipy.linalg import eigvalsh

        L = np.diag(A.sum(axis=1)) - A
        eigenvalues = eigvalsh(L)
        eigenvalues = np.sort(eigenvalues)
        diffs = np.diff(eigenvalues[:20])
        n_clusters = int(np.argmax(diffs) + 2)
        n_clusters = max(2, min(n_clusters, len(A) // 2))

        sc = SpectralClustering(
            n_clusters=n_clusters, affinity="precomputed", random_state=self.random_state
        )
        labels = sc.fit_predict(np.maximum(A, 0))
        return self._build_cluster_output(list(labels))

    def _build_cluster_output(self, membership):
        """Build cluster dict, weights from a membership list."""
        membership = np.array(membership)
        n_clusters = len(set(membership))

        sim = self.similarity_matrix_.tolil()
        sim.setdiag(1)
        sim_dense = sim.toarray()

        # Compute cluster membership matrix
        cluster_mat = csr_matrix(
            (np.ones(len(membership)), (membership, np.arange(len(membership)))),
            shape=(n_clusters, len(membership)),
        )

        # Weights: intra-cluster similarity / total similarity
        inside_weights = np.array(
            (cluster_mat.dot(csr_matrix(sim_dense))).multiply(cluster_mat).sum(axis=0)
        ).reshape(-1)
        overall_weights = np.array(sim_dense.sum(axis=0)).reshape(-1)
        weights = np.where(overall_weights > 0, inside_weights / overall_weights, 0)

        # Build clusters dict
        clusters = {}
        for label in range(n_clusters):
            mask = membership == label
            clusters[label] = self._core_order[mask]

        return clusters, n_clusters, weights

    def _compute_embedding(self, G):
        """Step 5: Compute embedding features for all nodes."""
        membership = np.zeros(len(self._core_order), dtype=int)
        for label, members in self.clusters_.items():
            for m in members:
                idx = np.where(self._core_order == m)[0]
                if len(idx) > 0:
                    membership[idx[0]] = label

        n_clusters = self.n_clusters_
        weights = self.core_weights_

        # Build weighted cluster matrix
        cluster_mat = np.zeros((n_clusters, len(self._core_order)))
        for i in range(n_clusters):
            mask = membership == i
            if weights[mask].sum() > 0:
                cluster_mat[i, mask] = weights[mask]

        # Normalize within each cluster
        row_sums = cluster_mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cluster_mat = cluster_mat / row_sums

        # Build full adjacency to core nodes
        all_nodes = list(G.nodes())
        full_adj = nx.adjacency_matrix(G, all_nodes).tolil()
        full_adj.setdiag(1)

        # Map core_order to positions in all_nodes
        node_to_idx = {n: i for i, n in enumerate(all_nodes)}
        core_indices = [node_to_idx[c] for c in self._core_order if c in node_to_idx]

        bip_adj = full_adj[:, core_indices]

        # Embedding = bipartite_adj @ cluster_mat^T
        embedding_values = bip_adj.dot(cluster_mat.T)
        if hasattr(embedding_values, "A"):
            embedding_values = embedding_values.toarray()

        # Normalize rows
        row_sums = embedding_values.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        embedding_values = embedding_values / row_sums

        col_names = [f"E{i}" for i in range(n_clusters)]
        embedding_df = pd.DataFrame(embedding_values, index=all_nodes, columns=col_names)
        embedding_df.index.name = "node"

        return embedding_df

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
