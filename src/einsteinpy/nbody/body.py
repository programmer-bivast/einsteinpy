"""Body validation helpers for :mod:`einsteinpy.nbody`."""

__all__ = ["InvalidBodyError"]


class InvalidBodyError(ValueError):
    """Raised when an invalid body is supplied to an N-Body system."""

    pass
