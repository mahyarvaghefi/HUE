# HUE: Homophily-based User Embedding

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Published in JAIS](https://img.shields.io/badge/Published-JAIS%202021-green.svg)](https://doi.org/10.17705/1jais.00711)

A node embedding method for online social networks that extracts **interpretable** user preferences through community-level structural similarity.

> **📄 Paper:** Vaghefi, M. S., & Nazareth, D. L. (2021). *Mining Online Social Networks: Deriving User Preferences through Node Embedding.* Journal of the Association for Information Systems, 22(6). [doi: 10.17705/1jais.00711](https://doi.org/10.17705/1jais.00711)

🌐 **[Documentation](https://mahyarvaghefi.github.io/projects/hue/)**

---

## Overview

HUE is a graph-based node embedding algorithm that produces **meaningful, interpretable dimensions** from the structure of online social networks. Unlike methods such as DeepWalk, Node2Vec, or LINE — which produce latent, opaque dimensions — HUE embeddings correspond to identifiable communities of interest (e.g., news, sports, fashion).

### Key Features

- **Interpretable embeddings** — Each dimension maps to a real community of interest
- **Scalable** — Only a small fraction of nodes (core vertices) are needed for similarity computation
- **Incremental** — New nodes can be embedded without recomputing the entire model
- **Versatile** — Supports community detection, user preference identification, and recommendation systems

## The Extended Bipartite Graph

The central idea in HUE is the **extended bipartite graph**. In a social network like Twitter, there are two types of accounts:

- **Social pages** (core vertices): public accounts like @nytimes, @espn, @vogue that represent specific topics or interests. These accounts typically have many followers.
- **Regular users**: ordinary users who follow some set of social pages and are also connected to each other (mutual follows, friendships).

This gives rise to two networks that can be provided separately:

1. **Bipartite network** (users → pages): who follows which social pages. This is a directed bipartite graph where rows are users and columns are pages.
2. **User-to-user network**: the friendship or mutual-follow graph among regular users.

HUE uses both networks together. The bipartite network tells us *which pages each user follows*, and the user network tells us *how those followers are connected to each other*. By examining the overlap and internal connectivity of each page's follower community (its **ego's alter-network**), HUE computes a structural similarity between every pair of pages.

### How It Works

HUE operates in five steps:

1. **Core Selection** — Identify core vertices. In real social networks, these are social media pages sorted by follower count. In synthetic graphs, they can be selected randomly or by degree.
2. **Extended Bipartite Graph** — Separate the network into a bipartite graph (users × cores) and a user-to-user graph.
3. **Network Simplification** — Compute the ego's alter-network structural similarity between all pairs of core vertices. This produces a **weighted graph of cores** where edge weights reflect how similar two pages' follower communities are.
4. **Core Clustering** — Apply modularity-based community detection (Louvain/Leiden with Surprise optimization) on the weighted core graph to discover preference clusters.
5. **Embedding** — Represent every user by their weighted connectivity to each discovered community.

## Installation

```bash
git clone https://github.com/mahyarvaghefi/HUE.git
cd HUE
pip install -e .
```

### With clustering support (recommended)

```bash
pip install -e ".[clustering]"
```

### With all optional dependencies

```bash
pip install -e ".[all]"
```

## Usage

### Option 1: From Two Separate Networks (Recommended for Real Data)

When you have a social network dataset with a known bipartite structure (users following pages), provide the two networks as CSV files:

```python
import numpy as np
import pandas as pd
import networkx as nx
from hue_embedding import HUE

# Load CSV files
# bipartite.csv:    columns [Source, Target] — Source=user, Target=page
# user_network.csv: columns [Source, Target] — both are users
bipartite_df = pd.read_csv('bipartite.csv')
user_net_df  = pd.read_csv('user_network.csv')

# Build bipartite graph (users → pages)
bip_graph = nx.from_pandas_edgelist(
    bipartite_df, 'Source', 'Target', create_using=nx.DiGraph())

# Build user-to-user graph (keep mutual follows only)
usr_graph = nx.from_pandas_edgelist(
    user_net_df, 'Source', 'Target', create_using=nx.DiGraph())
to_remove = [(u, v) for u, v in usr_graph.edges()
             if not usr_graph.has_edge(v, u)]
usr_graph.remove_edges_from(to_remove)
usr_graph = usr_graph.to_undirected()

# Identify users and pages, extract adjacency matrices
pages = sorted(set(bipartite_df['Target']))
users = sorted(set(bipartite_df['Source']) - set(pages))
bip_adj = nx.bipartite.biadjacency_matrix(bip_graph, users, pages)
usr_adj = nx.adjacency_matrix(usr_graph, users)

# Fit HUE — all parameters are set on the constructor
model = HUE(
    n_clustering_iterations=50,   # runs of community detection
    partition_method='surprise',   # or 'modularity'
    n_jobs=2,                      # parallel workers for similarity
    chunksize=3,                   # multiprocessing chunk size
    normalize=True,                # rows sum to 1
    random_state=42
)
embedding = model.fit_from_networks(
    bip_adj, usr_adj,
    user_ids=np.array(users), core_ids=np.array(pages)
)

# Output 1: Which cluster each page belongs to
cluster_labels = model.get_cluster_labels()
page_clusters = pd.DataFrame([
    {'page': p, 'cluster': c} for p, c in cluster_labels.items()
]).sort_values('cluster')
page_clusters.to_csv('page_clusters.csv', index=False)

# Output 2: User embedding vectors
embedding.to_csv('user_embeddings.csv', index=True)
```

### Option 2: From a Single Graph (Synthetic Data, Exploration)

For quick experiments on synthetic or general graphs, HUE can select cores and extract the bipartite structure automatically:

```python
import networkx as nx
from hue_embedding import HUE

G = nx.random_partition_graph([200, 200, 200, 200], 0.4, 0.05)

# Random core selection
model = HUE(n_cores=0.15, core_selection='random', random_state=42)
embedding = model.fit_transform(G)

# Or select highest-degree nodes as cores
model = HUE(n_cores=100, core_selection='degree')
embedding = model.fit_transform(G)

# Or provide your own core nodes
model = HUE(core_selection='predefined')
embedding = model.fit_transform(G, core_nodes=[...])
```

### Community Detection

```python
from sklearn.cluster import KMeans

model = HUE(n_cores=0.15, random_state=42)
embedding = model.fit_transform(G)

# Cluster all nodes using the embedding
kmeans = KMeans(n_clusters=4, random_state=0)
labels = kmeans.fit_predict(embedding)
```

## Algorithm Details

### Ego's Alter-Network Structural Similarity

The core innovation in HUE is a similarity function that captures second-order proximity by considering both **common neighbors** and the **connection pattern** between them:

$$
Sim(\tilde{G}_A, \tilde{G}_B) = \frac{\left(|V(\tilde{G}_A, \tilde{G}_B)| + |E(\tilde{G}_A, \tilde{G}_B)|\right)^2}{\left(|V(\tilde{G}_A)| + |E(\tilde{G}_A)|\right) \cdot \left(|V(\tilde{G}_B)| + |E(\tilde{G}_B)|\right)}
$$

This similarity is computed for every pair of core vertices. The resulting similarity matrix is a **weighted graph** where each edge weight reflects how similar two pages' follower communities are. Modularity-based community detection is then applied to this weighted graph to discover clusters of structurally similar pages.

### Embedding Computation

Each user is represented by a vector where each dimension corresponds to a discovered community. The value reflects the user's level of connectivity to core vertices in that community, weighted by each core's centrality within its cluster.

## API Reference

### `HUE(n_cores, core_selection, n_clustering_iterations, partition_method, n_jobs, chunksize, normalize, random_state)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_cores` | int or float | `0.15` | Number of cores (int) or fraction of nodes (float). Ignored by `fit_from_networks`. |
| `core_selection` | str | `'random'` | `'random'`, `'degree'`, or `'predefined'`. Ignored by `fit_from_networks`. |
| `n_clustering_iterations` | int | `50` | Runs of modularity optimization to find best partition |
| `partition_method` | str | `'surprise'` | `'surprise'` (SurpriseVertexPartition) or `'modularity'` (ModularityVertexPartition) |
| `n_jobs` | int | `2` | Parallel workers for similarity computation |
| `chunksize` | int | `3` | Chunk size for multiprocessing |
| `normalize` | bool | `True` | Normalize rows to sum to 1 (proportional membership) |
| `verbose` | bool | `True` | Print progress messages during fitting |
| `random_state` | int or None | `None` | Seed for reproducibility |

**Methods:**

| Method | Description |
|---|---|
| `fit(G, core_nodes=None)` | Fit from a single NetworkX graph |
| `transform(G=None)` | Return embedding (recompute for new G) |
| `fit_transform(G, core_nodes=None)` | Fit and return embedding |
| `fit_from_networks(bipartite_adj, user_adj, user_ids, core_ids)` | Fit from two separate adjacency matrices |
| `get_cluster_labels()` | Return core-node-to-cluster mapping |

**Attributes (after fitting):**

| Attribute | Type | Description |
|---|---|---|
| `core_nodes_` | ndarray | Core vertex IDs |
| `similarity_matrix_` | sparse matrix | Weighted core-to-core similarity graph |
| `clusters_` | dict | Cluster label → core node list |
| `n_clusters_` | int | Number of discovered communities |
| `embedding_` | DataFrame | Full embedding matrix |

## Examples

See [`examples/full_pipeline.ipynb`](examples/full_pipeline.ipynb) for a complete demonstration on a synthetic 1,500-node network with visualization of core nodes, core clusters, and community detection results.

## Dependencies

**Required:** NumPy ≥ 1.20, Pandas ≥ 1.3, NetworkX ≥ 2.6, SciPy ≥ 1.7, scikit-learn ≥ 1.0

**Recommended:** [leidenalg](https://github.com/vtraag/leidenalg) (Surprise-optimized community detection), [python-igraph](https://igraph.org/python/), [matplotlib](https://matplotlib.org/), [fa2](https://github.com/bhargavchippada/forceatlas2) (ForceAtlas2 graph layout)

## Citation

If you use HUE in your research, please cite:

```bibtex
@article{vaghefi2021mining,
  title={Mining Online Social Networks: Deriving User Preferences through Node Embedding},
  author={Vaghefi, Mahyar S. and Nazareth, Derek L.},
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

- **[Mahyar S. Vaghefi](https://mahyarsv.github.io/)** — University of Texas at Arlington
- **Derek L. Nazareth** — University of Wisconsin-Milwaukee
