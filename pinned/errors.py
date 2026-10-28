class PinnedRuntimeError(RuntimeError):
    """A pinned source, package, checkpoint or asset is missing or differs from its pin. Startup must stop: there is no fallback."""
