NBody
=====

This module defines the ``NBody`` class, which evolves a system of mutually
interacting :py:class:`~einsteinpy.bodies.Body` objects, all defined in a
common inertial frame.

Unlike :py:mod:`~einsteinpy.geodesic`, which propagates a single test particle
through a *fixed* background metric, this module models the self-consistent
gravitational interaction between every body in the system, i.e. each body both
sources and responds to the gravitational field. Newtonian gravity is used by
default, with an optional first-order post-Newtonian (1PN) correction.

.. automodule:: einsteinpy.nbody.nbody
    :members:
    :show-inheritance:
