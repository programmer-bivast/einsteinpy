"""
Verification suite for ``einsteinpy.nbody`` (an N-body gravitational solver).

These tests check *physical* correctness using position-only invariants, so
they do not depend on the solver exposing energy/momentum diagnostics:

* circular binary: Kepler period, stationary barycenter, radii ~ mass ratio
* eccentric binary: closed orbit / no Newtonian precession
* three-body figure-8 orbit (Chenciner-Montgomery): periodicity + frozen CM
* restart: continuing a simulation from its synced state equals a single run
* input validation (wrong bodies, missing coordinates)

Evolved ``NBody`` API (as implemented):

.. code-block:: python

    sim = NBody(bodies, pn_correction=False)  # list of einsteinpy.bodies.Body
    times, results = sim.simulate(
        end_time, steps=100, method="RK45", rtol=..., atol=...
    )
    # times   : (steps,) float array of times, seconds
    # results : (steps, N, 6) array of (x, y, z, v_x, v_y, v_z) per body, SI
    sim.trajectory  # same (times, results) tuple from the last simulate() call

Errors are raised as ``InvalidBodyError`` (a ``ValueError``). All tests are
skipped until ``einsteinpy.nbody.NBody`` can be imported.

"""
import numpy as np
import pytest
from astropy import units as u
from astropy.constants import G

from einsteinpy.bodies import Body
from einsteinpy.coordinates.differential import CartesianDifferential

try:
    from einsteinpy.nbody import NBody, InvalidBodyError
except (ImportError, ModuleNotFoundError):
    try:
        from einsteinpy.nbody.nbody import NBody, InvalidBodyError
    except (ImportError, ModuleNotFoundError):
        NBody = None
        InvalidBodyError = None

requires_nbody = pytest.mark.skipif(
    NBody is None, reason="einsteinpy.nbody.NBody is not implemented yet"
)

FIGURE8_PERIOD = 6.324  # dimensionless period of the Chenciner-Montgomery 8-orbit


def _body(name, position, velocity, mass_kg):
    """Build a Body with Cartesian SI quantities (positions/velocities in m, m/s)."""
    x, y, z = position
    vx, vy, vz = velocity
    return Body(
        name=name,
        mass=mass_kg * u.kg,
        coords=CartesianDifferential(
            0.0 * u.s,
            x * u.m,
            y * u.m,
            z * u.m,
            vx * u.m / u.s,
            vy * u.m / u.s,
            vz * u.m / u.s,
        ),
    )


def circular_binary(m1, m2, d):
    """
    Two bodies on a circular orbit about their common center of mass.

    m1, m2 : masses in kg (floats); d : separation in m.
    Returns (body1, body2, period_seconds).
    """
    mu = G.value * (m1 + m2)
    omega = np.sqrt(mu / d ** 3)
    r1, r2 = m2 / (m1 + m2) * d, m1 / (m1 + m2) * d
    b1 = _body("b1", (r1, 0.0, 0.0), (0.0, r1 * omega, 0.0), m1)
    b2 = _body("b2", (-r2, 0.0, 0.0), (0.0, -r2 * omega, 0.0), m2)
    period = 2 * np.pi * np.sqrt(d ** 3 / mu)
    return b1, b2, period


def eccentric_binary_at_periapsis(m1, m2, a, e):
    """
    Two bodies on an eccentric orbit (semi-major axis ``a`` m, eccentricity ``e``),
    initialized at periapsis in the center-of-mass frame.
    """
    mu = G.value * (m1 + m2)
    r_p = a * (1 - e)
    v_rel = np.sqrt(mu * (2 / r_p - 1 / a))
    cm = m2 / (m1 + m2)
    b1 = _body("b1", (-cm * r_p, 0.0, 0.0), (0.0, -cm * v_rel, 0.0), m1)
    b2 = _body("b2", ((1 - cm) * r_p, 0.0, 0.0), (0.0, (1 - cm) * v_rel, 0.0), m2)
    period = 2 * np.pi * np.sqrt(a ** 3 / mu)
    return b1, b2, period


def _simulate(bodies, end_time, steps):
    """Shortcut: build an NBody instance and simulate with tight tolerances."""
    sim = NBody(bodies=bodies)
    return sim.simulate(end_time, steps=steps, rtol=1e-9, atol=1e-9)


def measure_period(times, x):
    """
    Period = time between two successive positive-to-negative crossings of x.
    Returns None when fewer than two such crossings are found.
    """
    flip = np.where((x[:-1] >= 0) & (x[1:] < 0))[0]
    if len(flip) < 2:
        return None
    return times[flip[1]] - times[flip[0]]


@requires_nbody
def test_circular_binary_kepler_period_matches_formula():
    b1, b2, expected = circular_binary(1.989e30, 5.972e24, 1.496e11)
    # about 2.6 orbits, enough to observe two positive-to-negative crossings
    t_bound, n = 2.6 * expected, 6000
    times, results = _simulate([b1, b2], t_bound, n)
    assert results.shape == (n, 2, 6)

    measured = measure_period(times, results[:, 0, 0])
    assert measured is not None
    rel = abs(measured - expected) / expected
    # Sun-Earth-like system: ~1 sidereal year, also checks SI unit handling
    assert rel < 0.01, f"period off by {rel:.2%} (expected {expected / 86400:.2f} d)"


@requires_nbody
def test_circular_binary_barycenter_is_fixed():
    d = 3.0e11
    b1, b2, _ = circular_binary(4e30, 1e30, d)
    period = 2 * np.pi * np.sqrt(d ** 3 / (G.value * 5e30))
    _, results = _simulate([b1, b2], 0.6 * period, 3000)
    pos = results[:, :, :3]
    cm = (4 * pos[:, 0] + pos[:, 1]) / 5.0
    drift = np.max(np.linalg.norm(cm - cm[0], axis=1))
    assert drift < 1e-3 * d, f"barycenter drifted by {drift / d:.2%} of separation"


@requires_nbody
def test_circular_binary_radii_scale_as_inverse_mass_ratio():
    m1, m2, d = 4e30, 1e30, 3.0e11
    b1, b2, _ = circular_binary(m1, m2, d)
    r1_exp, r2_exp = m2 / (m1 + m2) * d, m1 / (m1 + m2) * d
    _, results = _simulate([b1, b2], 600 * 24 * 3600, 3000)
    pos = results[:, :, :3]
    r1_obs = np.max(np.linalg.norm(pos[:, 0], axis=1))
    r2_obs = np.max(np.linalg.norm(pos[:, 1], axis=1))
    assert abs(r1_obs - r1_exp) / r1_exp < 0.02
    assert abs(r2_obs - r2_exp) / r2_exp < 0.02


@requires_nbody
def test_eccentric_binary_orbit_closes_without_precession():
    a, e = 2.0e11, 0.3
    b1, b2, period = eccentric_binary_at_periapsis(1.989e30, 1e29, a, e)
    n = 8000
    t_bound = 2.7 * period  # cover two more apoapsides (at 1.5 T and 2.5 T)
    times, results = _simulate([b1, b2], t_bound, n)
    r = np.linalg.norm(results[:, 1, :3] - results[:, 0, :3], axis=1)
    r_p, r_a = a * (1 - e), a * (1 + e)

    # after one full period the separation returns to the periapsis value
    idx = np.searchsorted(times, period)
    assert abs(r[idx] - r_p) / r_p < 0.02, "orbit did not close after one period"
    # no Newtonian precession: apoapsis distance stable in each later orbit
    for k in range(2):
        lo = np.searchsorted(times, (k + 1.05) * period)
        hi = np.searchsorted(times, (k + 1.95) * period)
        assert abs(r[lo:hi].max() - r_a) / r_a < 0.03


@requires_nbody
def test_three_body_figure8_periodic_and_barycenter_frozen():
    # Chenciner-Montgomery figure-8 orbit, scaled to SI.
    # Dimensionless ICs (G=m=1): x1=(-1,0,0), x2=(0,0,0), x3=(1,0,0),
    # v1=(vx,vy,0), v2=(-2vx,-2vy,0), v3=(vx,vy,0).
    vx, vy = 0.3471112818256140, 0.5327285284700955
    M, L = 1e30, 1e11
    T_s = np.sqrt(L ** 3 / (G.value * M))  # seconds per dimensionless time unit

    bodies = [
        _body("s1", (-1.0 * L, 0.0, 0.0), (vx * L / T_s, vy * L / T_s, 0.0), M),
        _body("s2", (0.0, 0.0, 0.0), (-2 * vx * L / T_s, -2 * vy * L / T_s, 0.0), M),
        _body("s3", (1.0 * L, 0.0, 0.0), (vx * L / T_s, vy * L / T_s, 0.0), M),
    ]
    t_bound, n = 1.1 * FIGURE8_PERIOD * T_s, 4000
    times, results = _simulate(bodies, t_bound, n)
    pos = results[:, :, :3]

    # barycenter must stay frozen
    cm = pos.mean(axis=1)
    assert np.max(np.linalg.norm(cm - cm[0], axis=1)) < 1e-3 * L

    # must not blow up (escape => sign error in the force law)
    span = np.max(np.linalg.norm(pos - cm[:, None, :], axis=2)) / L
    assert span < 3 * L, f"system escaped (max separation {span:.1f} L)"

    # periodic: after exactly one figure-8 period the stars return to their start
    idx = np.searchsorted(times, FIGURE8_PERIOD * T_s)
    shift = np.max(np.linalg.norm(pos[idx] - pos[0], axis=1)) / L
    assert shift < 0.05, f"figure-8 rejected: position shift after one period {shift * 100:.1f}% L"


@requires_nbody
def test_restart_from_synced_state_matches_single_run():
    # simulate() must (a) sync the Body objects to the final state, and
    # (b) allow continuing: a second simulate() from that state must equal
    # one uninterrupted run of the combined duration.
    d = 3.0e11
    b1, b2, T = circular_binary(4e30, 1e30, d)
    half = 0.31 * T

    sim = NBody(bodies=[b1, b2])
    t1, r1 = sim.simulate(half, steps=800, rtol=1e-9, atol=1e-9)
    assert abs(b1.coords.x.si.value - r1[-1, 0, 0]) < 1.0
    assert abs(b2.coords.y.si.value - r1[-1, 1, 1]) < 1.0

    t2, r2 = sim.simulate(half, steps=800, rtol=1e-9, atol=1e-9)
    rb1, rb2, _ = circular_binary(4e30, 1e30, d)
    _, ref = _simulate([rb1, rb2], 2 * half, 1600)
    rel = np.max(np.linalg.norm(r2[-1, :, :3] - ref[-1, :, :3], axis=1)) / d
    assert rel < 1e-6, f"restart drifted: {rel:.2e}"
    for body, ref_pos in zip([b1, b2], ref[-1, :, :3]):
        cur = np.array(
            [
                body.coords.x.si.value,
                body.coords.y.si.value,
                body.coords.z.si.value,
            ]
        )
        assert np.linalg.norm(cur - ref_pos) < 1e-6 * d


def test_nbody_rejects_non_body_input():
    if NBody is None:
        pytest.skip("NBody not implemented")
    b1, b2, _ = circular_binary(4e30, 1e30, 3.0e11)
    with pytest.raises(InvalidBodyError):
        NBody(bodies=[b1, "not a body"])


def test_nbody_rejects_too_few_bodies():
    if NBody is None:
        pytest.skip("NBody not implemented")
    b1, _, _ = circular_binary(4e30, 1e30, 3.0e11)
    with pytest.raises(InvalidBodyError):
        NBody(bodies=[b1])


def test_nbody_rejects_body_without_coordinates():
    if NBody is None:
        pytest.skip("NBody not implemented")
    good, _, _ = circular_binary(4e30, 1e30, 3.0e11)
    naked = Body(name="no coords", mass=1e30 * u.kg)
    with pytest.raises(InvalidBodyError):
        NBody(bodies=[good, naked])