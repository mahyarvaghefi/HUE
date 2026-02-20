"""Tests for the HUE embedding algorithm."""

import numpy as np
import networkx as nx
import pytest
from hue_embedding import HUE


@pytest.fixture
def simple_graph():
    """Create a small graph with clear community structure."""
    return nx.random_partition_graph([30, 30, 30], 0.4, 0.05, seed=42)


@pytest.fixture
def larger_graph():
    """Create a larger graph with 7 clusters."""
    return nx.random_partition_graph(
        [50, 50, 50, 50, 50, 50, 50], 0.4, 0.08, seed=42
    )


class TestHUEInit:
    def test_default_parameters(self):
        model = HUE()
        assert model.n_cores == 0.15
        assert model.core_selection == "random"

    def test_custom_parameters(self):
        model = HUE(n_cores=100, core_selection="degree", n_jobs=4)
        assert model.n_cores == 100
        assert model.core_selection == "degree"
        assert model.n_jobs == 4


class TestCoreSelection:
    def test_random_selection_fraction(self, simple_graph):
        model = HUE(n_cores=0.15, random_state=42)
        model._validate_input(simple_graph, None)
        rng = np.random.RandomState(42)
        cores = model._select_cores(simple_graph, None, rng)
        expected_n = int(len(simple_graph.nodes()) * 0.15)
        assert len(cores) == expected_n

    def test_random_selection_absolute(self, simple_graph):
        model = HUE(n_cores=10, random_state=42)
        model._validate_input(simple_graph, None)
        rng = np.random.RandomState(42)
        cores = model._select_cores(simple_graph, None, rng)
        assert len(cores) == 10

    def test_degree_selection(self, simple_graph):
        model = HUE(n_cores=5, core_selection="degree", random_state=42)
        model._validate_input(simple_graph, None)
        rng = np.random.RandomState(42)
        cores = model._select_cores(simple_graph, None, rng)
        assert len(cores) == 5

    def test_predefined_selection(self, simple_graph):
        core_ids = [0, 1, 2, 3, 4]
        model = HUE(core_selection="predefined")
        model._validate_input(simple_graph, core_ids)
        rng = np.random.RandomState(42)
        cores = model._select_cores(simple_graph, core_ids, rng)
        assert list(cores) == core_ids


class TestHUEFit:
    def test_fit_returns_self(self, simple_graph):
        model = HUE(n_cores=0.2, random_state=42, n_clustering_iterations=5)
        result = model.fit(simple_graph)
        assert result is model

    def test_fit_creates_embedding(self, simple_graph):
        model = HUE(n_cores=0.2, random_state=42, n_clustering_iterations=5)
        model.fit(simple_graph)
        assert model.embedding_ is not None
        assert model.embedding_.shape[0] == len(simple_graph.nodes())

    def test_fit_transform(self, simple_graph):
        model = HUE(n_cores=0.2, random_state=42, n_clustering_iterations=5)
        embedding = model.fit_transform(simple_graph)
        assert embedding.shape[0] == len(simple_graph.nodes())
        assert embedding.shape[1] == model.n_clusters_

    def test_embedding_values_normalized(self, simple_graph):
        model = HUE(n_cores=0.2, normalize=True, random_state=42, n_clustering_iterations=5)
        embedding = model.fit_transform(simple_graph)
        row_sums = embedding.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, 1.0, decimal=5)

    def test_embedding_values_unnormalized(self, simple_graph):
        model = HUE(n_cores=0.2, normalize=False, random_state=42, n_clustering_iterations=5)
        embedding = model.fit_transform(simple_graph)
        row_sums = embedding.sum(axis=1)
        # Raw values should NOT all sum to 1
        assert not np.allclose(row_sums, 1.0)
        # All values should be non-negative
        assert (embedding.values >= 0).all()


class TestInputValidation:
    def test_rejects_directed_graph(self):
        G = nx.DiGraph()
        G.add_edges_from([(0, 1), (1, 2)])
        model = HUE()
        with pytest.raises(ValueError, match="undirected"):
            model.fit(G)

    def test_rejects_non_graph(self):
        model = HUE()
        with pytest.raises(TypeError):
            model.fit("not a graph")

    def test_predefined_without_cores(self):
        G = nx.complete_graph(10)
        model = HUE(core_selection="predefined")
        with pytest.raises(ValueError, match="core_nodes must be provided"):
            model.fit(G)


class TestSimilarityMatrix:
    def test_similarity_matrix_shape(self, simple_graph):
        model = HUE(n_cores=10, random_state=42, n_clustering_iterations=5)
        model.fit(simple_graph)
        n_cores = len(model.core_nodes_)
        assert model.similarity_matrix_.shape == (n_cores, n_cores)

    def test_similarity_values_nonnegative(self, simple_graph):
        model = HUE(n_cores=10, random_state=42, n_clustering_iterations=5)
        model.fit(simple_graph)
        assert (model.similarity_matrix_.data >= 0).all()
