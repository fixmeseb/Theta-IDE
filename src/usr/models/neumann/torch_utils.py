"""Torch helpers for NEUMANN.

The implementations live once, in nsfr.utils.torch. NEUMANN keeps its historical softor default
(gamma=0.015, where NSFR's is 0.01) so callers that rely on the default get the same results as before.
"""
from nsfr.utils.torch import logsumexp, print_valuation, weight_sum  # noqa: F401  (re-exported)
from nsfr.utils.torch import softor as _softor


def softor(xs, dim=0, gamma=0.015):
    """Smooth logical OR (log-sum-exp) with NEUMANN's default gamma=0.015; see nsfr.utils.torch.softor."""
    return _softor(xs, dim=dim, gamma=gamma)
