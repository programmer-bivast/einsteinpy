import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_allclose

from einsteinpy.bodies import Body
from einsteinpy.coordinates import CartesianDifferential, SphericalDifferential
from einsteinpy.nbody import InvalidBodyError, NBody


def _cart(x, y, z, v_x, v_y, v_z):
    return CartesianDifferential(
        0.0 * u.s,
        x * u.m,
        y * u.m,
        z * u.m,
        v_x * u.m / u.s,
        v_y * u.m / u.s,
        v_z * u.m / u.s,
    )


@pytest.fixture()
def sun_earth_system():
    """
    A simplified, circular Sun-Earth system.

    """
    au = 1.496e11
    v_earth = 29780.0

    sun = Body(
        name="Sun",
        mass=1.989e30 * u.kg,
        coords=_cart(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    earth = Body(
        name="Earth",
        mass=5.972e24 * u.kg,
        coords=_cart(au, 0.0, 0.0, 0.0, v_earth, 0.0),
    )

    return NBody([sun, earth])


def test_nbody_requires_at_least_two_bodies():
    sun = Body(name="Sun", mass=1.989e30 * u.kg, coords=_cart(0, 0, 0, 0, 0, 0))

    with pytest.raises(InvalidBodyError):
        NBody([sun])

    with pytest.raises(InvalidBodyError):
        NBody([])


def test_nbody_rejects_body_without_coords():
    sun = Body(name="Sun", mass=1.989e30 * u.kg, coords=_cart(0, 0, 0, 0, 0, 0))
    earth = Body(name="Earth", mass=5.972e24 * u.kg)

    with pytest.raises(InvalidBodyError):
        NBody([sun, earth])


def test_nbody_rejects_non_cartesian_coords():
    sun = Body(name="Sun", mass=1.989e30 * u.kg, coords=_cart(0, 0, 0, 0, 0, 0))
    spherical = SphericalDifferential(
        0.0 * u.s,
        1.0 * u.m,
        0.0 * u.rad,
        0.0 * u.rad,
        0.0 * u.m / u.s,
        0.0 * u.rad / u.s,
        0.0 * u.rad / u.s,
    )
    earth = Body(name="Earth", mass=5.972e24 * u.kg, coords=spherical)

    with pytest.raises(InvalidBodyError):
        NBody([sun, earth])


def test_nbody_initial_state(sun_earth_system):
    assert len(sun_earth_system.bodies) == 2
    assert sun_earth_system.trajectory is None
    assert sun_earth_system._positions.shape == (2, 3)
    assert sun_earth_system._velocities.shape == (2, 3)


def test_nbody_repr_contains_body_names(sun_earth_system):
    text = repr(sun_earth_system)

    assert "Sun" in text
    assert "Earth" in text
    assert "NBody" in text


def test_simulate_returns_expected_shapes(sun_earth_system):
    steps = 40
    times, results = sun_earth_system.simulate(3.15e7 * u.s, steps=steps)

    assert times.shape == (steps,)
    assert results.shape == (steps, 2, 6)
    assert sun_earth_system.trajectory is not None


def test_simulate_accepts_plain_float_time(sun_earth_system):
    times, results = sun_earth_system.simulate(3.15e7, steps=10)

    assert_allclose(times[-1], 3.15e7)
    assert results.shape == (10, 2, 6)


def test_earth_orbit_radius_is_conserved(sun_earth_system):
    _, results = sun_earth_system.simulate(3.15e7 * u.s, steps=100)

    sun_pos = results[:, 0, :3]
    earth_pos = results[:, 1, :3]

    radii = np.linalg.norm(earth_pos - sun_pos, axis=1)

    # Near-circular orbit: the orbital radius should stay bounded, well within
    # an order of magnitude of the initial 1 AU separation
    assert np.all(radii > 0.5 * 1.496e11)
    assert np.all(radii < 1.5 * 1.496e11)


def test_total_momentum_is_conserved(sun_earth_system):
    _, results = sun_earth_system.simulate(3.15e7 * u.s, steps=50)

    masses = sun_earth_system._masses

    initial_v = results[0, :, 3:]
    final_v = results[-1, :, 3:]

    p_initial = np.sum(masses[:, None] * initial_v, axis=0)
    p_final = np.sum(masses[:, None] * final_v, axis=0)

    # Total linear momentum must be conserved. Compare against the magnitude
    # of the total momentum, to account for floating point round-off at the
    # (very large) SI scales involved.
    scale = np.linalg.norm(p_initial)
    assert_allclose(p_final, p_initial, atol=1e-6 * scale)


def test_simulate_updates_body_coords(sun_earth_system):
    earth = sun_earth_system.bodies[1]
    initial_x = earth.coords.x.si.value

    sun_earth_system.simulate(3.15e7 * u.s, steps=50)

    assert earth.coords.x.si.value != initial_x
    assert isinstance(earth.coords, CartesianDifferential)
    assert earth.pos_vec[0] == earth.coords.x


def test_pn_correction_changes_trajectory():
    au = 1.496e11
    v_earth = 29780.0

    def _make_bodies():
        sun = Body(
            name="Sun", mass=1.989e30 * u.kg, coords=_cart(0, 0, 0, 0, 0, 0)
        )
        earth = Body(
            name="Earth",
            mass=5.972e24 * u.kg,
            coords=_cart(au, 0.0, 0.0, 0.0, v_earth, 0.0),
        )
        return [sun, earth]

    newtonian = NBody(_make_bodies(), pn_correction=False)
    relativistic = NBody(_make_bodies(), pn_correction=True)

    _, res_n = newtonian.simulate(3.15e7 * u.s, steps=50)
    _, res_r = relativistic.simulate(3.15e7 * u.s, steps=50)

    # 1PN corrections are tiny but non-zero, so the trajectories must differ
    assert not np.allclose(res_n, res_r)
