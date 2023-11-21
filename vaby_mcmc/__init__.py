"""
Implementation of MCMC Bayesian inference for fitting timeseries data
"""
try:
    from ._version import __version__, __timestamp__
except ImportError:
    __version__ = "Unknown version"
    __timestamp__ = "Unknown timestamp"

from .mcmc import Mcmc

__all__ = [
    "__version__",
    "__timestamp__",
    "Mcmc",
]
