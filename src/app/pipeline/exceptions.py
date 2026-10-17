"""Pipeline-level exceptions for Theta."""


class ConfigurationError(Exception):
    """Raised when an experiment configuration is incompatible with its declared paradigm.

    Caught at startup by run_pipeline.py — the run is aborted before any training begins.
    """

    pass
