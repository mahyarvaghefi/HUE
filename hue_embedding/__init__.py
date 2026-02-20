"""
HUE: Homophily-based User Embedding

A node embedding method for online social networks that extracts
interpretable user preferences through community-level structural similarity.

Reference:
    Vaghefi, M. S., & Nazareth, D. L. (2021).
    Mining Online Social Networks: Deriving User Preferences through Node Embedding.
    Journal of the Association for Information Systems, 22(6).
    doi: 10.17705/1jais.00711
"""

from .core import HUE
from .similarity import find_similarity_matrix
from .version import __version__

__all__ = ["HUE", "find_similarity_matrix", "__version__"]
