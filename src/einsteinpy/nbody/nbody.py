"""N-Body numerical computation.

Unlike :py:mod:`~einsteinpy.geodesic`, which propagates a single test
particle through a *fixed* background metric, this module treats a
collection of :py:class:`~einsteinpy.bodies.Body` objects, all defined in a
common frame, that mutually interact gravitationally. The dynamics of every
body is influenced by every other body in the system.

The equations of motion are integrated numerically. By default, the
Newtonian gravitational acceleration is used, with an optional first-order
Post-Newtonian (1PN) correction, which captures the leading order General
Relativistic effects (e.g. the perihelion precession of Mercury).

"""

import numpy as np
from astropy import units as u
from scipy.integrate import solve_ivp

from einsteinpy.bodies import Body
from einsteinpy.constant import G, c
from einsteinpy.coordinates.differential import CartesianDifferential

from .body import InvalidBodyError

__all__ = ["NBody", "InvalidBodyError"]

_G = G.value
_c = c.value


def _to_seconds(value):
    if hasattr(value, "to"):
        return float(value.to(u.s).value)
    return float(value)


class NBody:
    """
    Class for handling gravitational N-Body numerical computation.

    Accepts a list of :py:class:`~einsteinpy.bodies.Body` objects (with
    ``len(bodies) > 1``), all defined in a common frame, and evolves their
    trajectories under their mutual gravitational interaction.

    """

    def __init__(self, bodies, t=0.0 * u.s, softening_length=0.0 * u.m,
                 pn_correction=False):
        """
        Constructor

        Parameters
        ----------
        bodies : list of ~einsteinpy.bodies.Body
            List of ``Body`` objects, all defined in a common frame.
            Every body must carry Cartesian coordinates
            (:py:class:`~einsteinpy.coordinates.differential.CartesianDifferential`)
            and a non-negative mass. The list must contain more than one body.
        t : ~astropy.units.quantity.Quantity, optional
            Initial coordinate time of the system.
            Defaults to ``0 s``.
        softening_length : ~astropy.units.quantity.Quantity, optional
            Plummer softening length, used to avoid numerical divergences
            during close encounters between bodies.
            Defaults to ``0 m`` (no softening).
        pn_correction : bool, optional
            Whether to include the first-order Post-Newtonian (1PN)
            correction by default.
            Defaults to ``False``.

        Raises
        ------
        InvalidBodyError
            If less than two bodies are supplied, if any body is not a
            ``Body`` instance, or if any body is missing Cartesian
            coordinates.

        """
        try:
            body_list = list(bodies)
        except TypeError:
            raise InvalidBodyError("bodies must be an iterable of Body objects.")

        if len(body_list) <= 1:
            raise InvalidBodyError(
                "NBody requires a list of more than one Body object. "
                f"Got {len(body_list)} bodies."
            )

        for body in body_list:
            if not isinstance(body, Body):
                raise InvalidBodyError(
                    f"Expected a Body instance, got {type(body).__name__}."
                )
            if not isinstance(body.coords, CartesianDifferential):
                raise InvalidBodyError(
                    f"Body '{body.name}' must carry a CartesianDifferential, "
                    f"so that all bodies share a common Cartesian frame. "
                    f"Got {type(body.coords).__name__ if body.coords is not None else None}."
                )

        self.bodies = list(body_list)
        self.t = t
        self.softening_length = softening_length
        self.pn_correction = bool(pn_correction)

        self._masses = np.array(
            [body.mass.to(u.kg).value for body in self.bodies], dtype=float
        )

        pos, vel = [], []
        for body in self.bodies:
            coords = body.coords
            pos.append(
                [
                    coords.x.to(u.m).value,
                    coords.y.to(u.m).value,
                    coords.z.to(u.m).value,
                ]
            )
            vel.append(
                [
                    coords.v_x.to(u.m / u.s).value,
                    coords.v_y.to(u.m / u.s).value,
                    coords.v_z.to(u.m / u.s).value,
                ]
            )

        self._positions = np.array(pos, dtype=float)
        self._velocities = np.array(vel, dtype=float)

        self._trajectory = None

    def __repr__(self):
        return (
            f"NBody Object:(\n"
            f"    Number of Bodies : ({len(self.bodies)}),\n"
            f"    Bodies : ({[body.name for body in self.bodies]}),\n"
            f"    Coordinate Time : ({self.t})\n"
            f")"
        )

    def __str__(self):
        return self.__repr__()

    @property
    def trajectory(self):
        """
        Returns the trajectory of the system, as computed by the last call to
        :py:meth:`~einsteinpy.nbody.nbody.NBody.simulate`.

        Returns
        -------
        ~numpy.ndarray or None
            ``None``, if no trajectory has been computed yet. Otherwise, a
            tuple ``(t, states)``, as described in
            :py:meth:`~einsteinpy.nbody.nbody.NBody.simulate`.

        """
        return self._trajectory

    def _accelerations(self, positions, velocities, post_newtonian=False):
        """
        Computes the acceleration of every body due to every other body.

        Parameters
        ----------
        positions : ~numpy.ndarray
            Shape-(N, 3) array of body positions (in meters)
        velocities : ~numpy.ndarray
            Shape-(N, 3) array of body velocities (in meters / second)
        post_newtonian : bool, optional
            Whether to add the first-order Post-Newtonian (1PN) correction
            Defaults to ``False``

        Returns
        -------
        ~numpy.ndarray
            Shape-(N, 3) array of accelerations (in meters / second^2)

        """
        n = positions.shape[0]
        m = self._masses
        eps2 = self.softening_length.to(u.m).value ** 2

        acc = np.zeros_like(positions)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                r_ij = positions[j] - positions[i]
                dist2 = r_ij @ r_ij + eps2
                dist = np.sqrt(dist2)
                inv_dist3 = 1.0 / (dist2 * dist)

                a_newton = _G * m[j] * r_ij * inv_dist3
                acc[i] += a_newton

                if post_newtonian:
                    v_i = velocities[i]
                    v_j = velocities[j]
                    n_ij = r_ij / dist

                    term = (
                        4.0 * _G * m[j] / dist
                        + 5.0 * _G * m[i] / dist
                        + (v_i @ v_i)
                        - 4.0 * (v_i @ v_j)
                        + 2.0 * (v_j @ v_j)
                        - 1.5 * (n_ij @ v_j) ** 2
                    )

                    acc_pn = (
                        _G
                        * m[j]
                        * inv_dist3
                        / (_c ** 2)
                        * (
                            r_ij * term
                            + dist2 * (n_ij @ (4.0 * v_i - 3.0 * v_j)) * (v_i - v_j)
                        )
                    )

                    acc[i] += acc_pn

        return acc

    def _rhs(self, t, y, post_newtonian):
        """
        Right-hand side of the system of first-order ODEs.

        """
        n = len(self.bodies)
        positions = y[: 3 * n].reshape(n, 3)
        velocities = y[3 * n :].reshape(n, 3)

        acc = self._accelerations(positions, velocities, post_newtonian)

        return np.concatenate((velocities.ravel(), acc.ravel()))

    def simulate(
        self,
        end_time,
        steps=50,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
        post_newtonian=None,
    ):
        """
        Calculate the trajectories of all bodies in the system.

        Parameters
        ----------
        end_time : ~astropy.units.quantity.Quantity or float, optional
            Coordinate time, up to which, the system should be evolved
            (measured from the initial time ``t``).
            Defaults to ``1 s``.
        steps : int, optional
            Number of time steps, at which the state is reported.
            Defaults to ``50``.
        method : str, optional
            Integration method to pass to
            :py:func:`scipy.integrate.solve_ivp`.
            Defaults to ``"RK45"``.
        rtol : float, optional
            Relative tolerance for the integrator.
            Defaults to ``1e-8``.
        atol : float, optional
            Absolute tolerance for the integrator.
            Defaults to ``1e-10``.
        post_newtonian : bool, optional
            Whether to include the first-order Post-Newtonian (1PN)
            correction. Defaults to the value set at construction.

        Returns
        -------
        ~numpy.ndarray
            Shape-(steps,) array, containing the coordinate times (in seconds)
        ~numpy.ndarray
            Shape-(steps, N, 6) array, containing, for each reported time and
            each body, the state ``(x, y, z, v_x, v_y, v_z)`` in SI units.

        """
        if post_newtonian is None:
            post_newtonian = self.pn_correction

        n = len(self.bodies)
        t0 = _to_seconds(self.t)
        dt = _to_seconds(end_time)
        t1 = t0 + dt

        y0 = np.concatenate(
            (self._positions.ravel(), self._velocities.ravel())
        )

        t_eval = np.linspace(t0, t1, steps)

        sol = solve_ivp(
            fun=self._rhs,
            t_span=(t0, t1),
            y0=y0,
            method=method,
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
            args=(bool(post_newtonian),),
        )

        if not sol.success:
            raise RuntimeError(
                f"N-Body integration failed: {sol.message}"
            )

        states = sol.y.T.reshape(-1, 2, n, 3)
        states = np.concatenate((states[:, 0], states[:, 1]), axis=2)

        times = np.asarray(sol.t, dtype=float)
        results = np.asarray(states, dtype=float)

        # sync internal state to final
        final = results[-1]
        self._positions = np.array(final[:, :3], dtype=float)
        self._velocities = np.array(final[:, 3:], dtype=float)
        self.t = (t1) * u.s

        # sync body objects
        for idx, body in enumerate(self.bodies):
            coords = CartesianDifferential(
                self.t,
                self._positions[idx, 0] * u.m,
                self._positions[idx, 1] * u.m,
                self._positions[idx, 2] * u.m,
                self._velocities[idx, 0] * u.m / u.s,
                self._velocities[idx, 1] * u.m / u.s,
                self._velocities[idx, 2] * u.m / u.s,
            )
            body.coords = coords
            body.pos_vec = [coords.x, coords.y, coords.z]
            body.vel_vec = [coords.v_x, coords.v_y, coords.v_z]

        self._trajectory = (times, results)

        return self._trajectory

    def calculate_trajectory(self, *args, **kwargs):
        return self.simulate(*args, **kwargs)

    def kinetic_energy(self, positions=None, velocities=None):
        """
        Returns the total (Newtonian) kinetic energy of the system.

        Parameters
        ----------
        velocities : ~numpy.ndarray, optional
            Shape-(N, 3) array of body velocities (in meters / second).
            Defaults to the initial velocities of the system.
        positions : ~numpy.ndarray, optional
            Present for API symmetry; unused.

        Returns
        -------
        float
            Total kinetic energy (in joules)

        """
        if velocities is None:
            velocities = self._velocities

        v2 = np.einsum("ij,ij->i", velocities, velocities)

        return 0.5 * float(np.sum(self._masses * v2))

    def potential_energy(self, positions=None):
        """
        Returns the total (Newtonian) gravitational potential energy of the
        system.

        Parameters
        ----------
        positions : ~numpy.ndarray, optional
            Shape-(N, 3) array of body positions (in meters).
            Defaults to the initial positions of the system.

        Returns
        -------
        float
            Total potential energy (in joules)

        """
        if positions is None:
            positions = self._positions

        n = positions.shape[0]
        m = self._masses
        eps2 = self.softening_length.to(u.m).value ** 2

        pe = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                r_ij = positions[j] - positions[i]
                dist = np.sqrt(r_ij @ r_ij + eps2)
                pe -= _G * m[i] * m[j] / dist

        return float(pe)

    def total_energy(self, positions=None, velocities=None):
        """
        Returns the total (Newtonian) energy of the system, i.e. the sum of
        the kinetic and potential energies.

        Parameters
        ----------
        positions : ~numpy.ndarray, optional
            Shape-(N, 3) array of body positions (in meters).
            Defaults to the initial positions of the system.
        velocities : ~numpy.ndarray, optional
            Shape-(N, 3) array of body velocities (in meters / second).
            Defaults to the initial velocities of the system.

        Returns
        -------
        float
            Total energy (in joules)

        """
        return self.kinetic_energy(velocities=velocities) + self.potential_energy(
            positions=positions
        )
