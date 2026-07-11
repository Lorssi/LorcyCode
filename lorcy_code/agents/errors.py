"""Exceptions shared by agent construction, middleware, and the CLI."""


class ModelSwitchError(RuntimeError):
    """Signal that the current model is exhausted and a fallback is required."""
