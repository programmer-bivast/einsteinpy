import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_allclose

from einsteinpy.coordinates.conversion import (
    BoyerLindquistConversion,
    CartesianConversion,
    SphericalConversion,
)
from einsteinpy.coordinates.differential import (
    BoyerLindquistDifferential,
    CartesianDifferential,
    SphericalDifferential,
)
from einsteinpy.metric import Kerr, Schwarzschild
from einsteinpy.utils import CoordinateError


@pytest.fixture()
def cartesian_conversion():
    return CartesianConversion(
        1e3 * u.s,
        10e3 / np.sqrt(2) * u.m,
        10e3 / np.sqrt(2) * u.m,
        0 * u.m,
    )


@pytest.fixture()
def spherical_conversion():
    return SphericalConversion(
        1e3 * u.s,
        10e3 * u.m,
        1.5707963267948966 * u.rad,
        0.7853981633974483 * u.rad,
    )


@pytest.fixture()
def bl_conversion():
    return BoyerLindquistConversion(
        1e3 * u.s,
        10e3 * u.m,
        1.5707963267948966 * u.rad,
        0.7853981633974483 * u.rad,
    )


@pytest.fixture()
def cartesian_differential():
    return CartesianDifferential(
        1e3 * u.s,
        10e3 / np.sqrt(2) * u.m,
        10e3 / np.sqrt(2) * u.m,
        0 * u.m,
        -190e3 / np.sqrt(2) * u.m / u.s,
        210e3 / np.sqrt(2) * u.m / u.s,
        200e3 * u.m / u.s,
    )


@pytest.fixture()
def spherical_differential():
    return SphericalDifferential(
        1e3 * u.s,
        10e3 * u.m,
        1.5707963267948966 * u.rad,
        0.7853981633974483 * u.rad,
        10e3 * u.m / u.s,
        -20.0 * u.rad / u.s,
        20.0 * u.rad / u.s,
    )


@pytest.fixture()
def bl_differential():
    return BoyerLindquistDifferential(
        1e3 * u.s,
        10e3 * u.m,
        1.5707963267948966 * u.rad,
        0.7853981633974483 * u.rad,
        10e3 * u.m / u.s,
        -20.0 * u.rad / u.s,
        20.0 * u.rad / u.s,
    )


def test_conversions_return_positions_only():
    for conv in (CartesianConversion(1e3*u.s, 1e3*u.m, 1e3*u.m, 0*u.m),
                 SphericalConversion(1e3*u.s, 1e3*u.m, 1.0*u.rad, 0.5*u.rad),
                 BoyerLindquistConversion(1e3*u.s, 1e3*u.m, 1.0*u.rad, 0.5*u.rad)):
        vals = conv.values()
        pos = conv.position()
        assert len(vals) == len(pos) == 4


def test_conversion_convert_methods_return_tuples(cartesian_conversion):
    sph = cartesian_conversion.convert_spherical()
    bl = cartesian_conversion.convert_bl(M=1e24 * u.kg, a=0.0 * u.one)
    assert isinstance(sph, tuple) and len(sph) == 4
    assert isinstance(bl, tuple) and len(bl) == 4


def test_spherical_to_bl_to_spherical_roundtrip(spherical_conversion):
    M = 1e24 * u.kg
    a = 0.75 * u.one
    blt = spherical_conversion.convert_bl(M=M, a=a)
    assert isinstance(blt, tuple) and len(blt) == 4
    back = BoyerLindquistConversion(*blt).convert_spherical(M=M, a=a)
    assert isinstance(back, tuple) and len(back) == 4
    expected = spherical_conversion.values()

    def scalar(q):
        q = u.Quantity(q)
        return float(q.to_value(q.unit))

    assert_allclose(scalar(back[1]), scalar(expected[1]), atol=1e-6)
    assert_allclose(scalar(back[2]), scalar(expected[2]), atol=1e-6)
    assert_allclose(scalar(back[3]), scalar(expected[3]), atol=1e-6)
def test_differential_values_shape(spherical_differential, bl_differential):
    from einsteinpy.metric import Schwarzschild as _S
    ms = _S(coords=spherical_differential, M=1e24 * u.kg)
    for diff in (spherical_differential, bl_differential):
        assert len(diff.values()) == 7
        assert len(diff.position()) == 4
    vel = spherical_differential.velocity(ms)
    assert len(vel) == 4
def test_differential_cross_objects(spherical_differential):
    M = 1e24 * u.kg
    a = 0.75 * u.one
    bl = spherical_differential.bl_differential(M=M, a=a)
    assert isinstance(bl, BoyerLindquistDifferential)
    back = bl.spherical_differential(M=M, a=a)
    assert isinstance(back, SphericalDifferential)
    assert_allclose(np.asarray(back.position(), float),
                    np.asarray(spherical_differential.position(), float), atol=1e-6)


def test_v_t_setter_getter(spherical_differential, bl_differential):
    M = 1e24 * u.kg
    sch = Schwarzschild(coords=spherical_differential, M=M)
    spherical_differential.v_t = (sch,)
    assert spherical_differential.v_t >= 0 * u.m / u.s
    with pytest.raises(CoordinateError):
        bl_differential.v_t = (sch,)


def test_metrics_accept_coords_kwarg(spherical_differential, bl_differential):
    M = 1e24 * u.kg
    a = 0.75 * u.one
    sch = Schwarzschild(coords=spherical_differential, M=M)
    ker = Kerr(coords=bl_differential, M=M, a=a)
    assert isinstance(sch, Schwarzschild)
    assert isinstance(ker, Kerr)
