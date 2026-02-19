# HUE: Homophily-based User Embedding

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Published in JAIS](https://img.shields.io/badge/Published-JAIS%202021-green.svg)](https://doi.org/10.17705/1jais.00711)

A node embedding method for online social networks that extracts **interpretable** user preferences through community-level structural similarity.

> **📄 Paper:** Sharif Vaghefi, M., & Nazareth, D. L. (2021). *Mining Online Social Networks: Deriving User Preferences through Node Embedding.* Journal of the Association for Information Systems, 22(6). [doi: 10.17705/1jais.00711](https://doi.org/10.17705/1jais.00711)

🌐 **[Project Website](https://mahyarsv.github.io/HUE/)**

---

## Overview

HUE is a graph-based node embedding algorithm that produces **meaningful, interpretable dimensions** from the structure of online social networks. Unlike methods such as DeepWalk, Node2Vec, or LINE—which produce latent, opaque dimensions—HUE embeddings correspond to identifiable communities of interest (e.g., fashion, sports, politics).

### Key Features

- **Interpretable embeddings** — Each dimension maps to a real community of interest
- **Scalable** — Only a small fraction of nodes (core vertices) are needed for similarity computation
- **Incremental** — New nodes can be embedded without recomputing the entire model
- **Versatile** — Supports community detection, link prediction, and recommendation systems
- **Competitive** — Outperforms DeepWalk, Node2Vec, and LINE on link prediction (AUC ≈ 0.95)

### How It Works

HUE operates in five steps:

1. **Core Selection** — Select representative vertices (e.g., social pages on Twitter)
2. **Extended Bipartite Graph** — Separate core vertices from the general network
3. **Network Simplification** — Compute ego's alter-network structural similarity between cores
4. **Core Clustering** — Cluster cores into communities using Louvain + Surprise optimization
5. **Embedding** — Represent all nodes by their weighted connectivity to each community

<p align="center">
  <img src="docs/assets/hue-pipeline.png" alt="HUE Pipeline" width="700"/>
</p>

## Installation

### Basic installation

```bash
pip install hue-embedding
```

### With clustering support (recommended)

```bash
pip install hue-embedding[clustering]
```

### With all optional dependencies

```bash
pip install hue-embedding[all]
```

### From source

```bash
git clone https://github.com/mahyarsv/HUE.git
cd HUE
pip install -e .[all]
```

## Quick Start

```python
import networkx as nx
from hue_embedding import HUE

# Create a graph with community structure
G = nx.random_partition_graph([200, 200, 200, 200], 0.4, 0.05)

# Fit HUE with 15% of nodes as cores
model = HUE(n_cores=0.15, core_selection='random', random_state=42)
embedding = model.fit_transform(G)

print(f"Embedding shape: {embedding.shape}")
print(f"Communities found: {model.n_clusters_}")
print(embedding.head())
```

### Using Pre-defined Core Vertices

For social network analysis, you can specify social pages or influencer accounts as core vertices:

```python
# Select nodes with highest degree as cores
model = HUE(n_cores=100, core_selection='degree')
embedding = model.fit_transform(G)
```

Or provide your own core nodes:

```python
core_pages = [...]  # list of social page node IDs
model = HUE(core_selection='predefined')
embedding = model.fit_transform(G, core_nodes=core_pages)
```

### Community Detection

```python
from sklearn.cluster import KMeans

model = HUE(n_cores=0.15, random_state=42)
embedding = model.fit_transform(G)

# Cluster all nodes using the embedding features
kmeans = KMeans(n_clusters=4, random_state=0)
labels = kmeans.fit_predict(embedding)
```

### Link Prediction

```python
import numpy as np

model = HUE(n_cores=0.15, random_state=42)
embedding = model.fit_transform(G)

# Hadamard operator for edge features (best performing)
def hadamard(u, v):
    return embedding.loc[u].values * embedding.loc[v].values

# Compute edge feature for a potential link
edge_feature = hadamard(node_a, node_b)
```

## Algorithm Details

### Ego's Alter-Network Structural Similarity

The core innovation in HUE is a similarity function that captures second-order proximity by considering both **common neighbors** and the **connection pattern** between them:

$$
Sim(\tilde{G}_A, \tilde{G}_B) = \frac{\left(|V(\tilde{G}_A, \tilde{G}_B)| + |E(\tilde{G}_A, \tilde{G}_B)|\right)^2}{\left(|V(\tilde{G}_A)| + |E(\tilde{G}_A)|\right) \cdot \left(|V(\tilde{G}_B)| + |E(\tilde{G}_B)|\right)}
$$

Where:
- $|V(\tilde{G}_A, \tilde{G}_B)|$ = number of common neighbors
- $|E(\tilde{G}_A, \tilde{G}_B)|$ = number of common edges between shared neighbors
- $|V(\tilde{G}_A)| + |E(\tilde{G}_A)|$ = total structural elements in A's alter network

This function ranges from 0 to 1 and preserves community structure better than simple common-neighbor counts.

### Embedding Computation

Each node is represented by a vector where each dimension corresponds to a discovered community. The value reflects the node's level of connectivity to core vertices in that community, weighted by each core's centrality within its cluster:

$$
Weight_A = \frac{\sum_{\forall B \in C_i} Sim(\tilde{G}_A, \tilde{G}_B)}{\sum_{\forall N} Sim(\tilde{G}_A, \tilde{G}_N)}
$$

## API Reference

### `HUE(n_cores, core_selection, n_clustering_iterations, n_jobs, chunksize, random_state)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_cores` | int or float | `0.15` | Number of cores (int) or fraction of nodes (float) |
| `core_selection` | str | `'random'` | `'random'`, `'degree'`, or `'predefined'` |
| `n_clustering_iterations` | int | `50` | Runs of Louvain to find best partition |
| `n_jobs` | int | `2` | Parallel workers for similarity computation |
| `chunksize` | int | `3` | Chunk size for multiprocessing |
| `random_state` | int or None | `None` | Seed for reproducibility |

**Methods:**

| Method | Description |
|---|---|
| `fit(G, core_nodes=None)` | Fit the model to graph G |
| `transform(G=None)` | Return embedding (recompute for new G) |
| `fit_transform(G, core_nodes=None)` | Fit and return embedding |
| `get_cluster_labels()` | Return core-node-to-cluster mapping |

**Attributes (after fitting):**

| Attribute | Type | Description |
|---|---|---|
| `core_nodes_` | ndarray | Selected core vertex IDs |
| `similarity_matrix_` | sparse matrix | Core-to-core similarity |
| `clusters_` | dict | Cluster label → core node list |
| `n_clusters_` | int | Number of discovered communities |
| `embedding_` | DataFrame | Full embedding matrix |

## Examples

See the [`examples/`](examples/) directory for complete Jupyter notebooks:

- **[`random_graph_demo.ipynb`](examples/random_graph_demo.ipynb)** — Full pipeline on a synthetic graph with visualization
- **[`random_graph_small.ipynb`](examples/random_graph_small.ipynb)** — Smaller network demonstration

## Dependencies

**Required:**
- NumPy ≥ 1.20
- Pandas ≥ 1.3
- NetworkX ≥ 2.6
- SciPy ≥ 1.7
- scikit-learn ≥ 1.0

**Recommended:**
- [leidenalg](https://github.com/vtraag/leidenalg) — For Surprise-optimized community detection
- [python-igraph](https://igraph.org/python/) — Required by leidenalg
- [matplotlib](https://matplotlib.org/) — For visualization
- [fa2](https://github.com/bhargavchippada/forceatlas2) — For ForceAtlas2 graph layout

## Citation

If you use HUE in your research, please cite:

```bibtex
@article{sharifvaghefi2021mining,
  title={Mining Online Social Networks: Deriving User Preferences through Node Embedding},
  author={Sharif Vaghefi, Mahyar and Nazareth, Derek L.},
  journal={Journal of the Association for Information Systems},
  volume={22},
  number={6},
  year={2021},
  doi={10.17705/1jais.00711}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Authors

- **[Mahyar Sharif Vaghefi](https://mahyarsv.github.io/)** — University of Texas at Arlington
- **Derek L. Nazareth** — University of Wisconsin-Milwaukee
