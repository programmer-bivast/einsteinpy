"""
Tests for symbolic parallel transport.

A vector field :math:`V` is parallel transported along a curve with tangent
:math:`u` iff its covariant derivative along the curve vanishes:

.. math::

    u^\\sigma \\nabla_\\sigma V^\\mu = 0 .

Expanding the covariant derivative along the curve this is equivalent to

.. math::

    \\frac{dV^\\mu}{d\\lambda} + \\Gamma^\\mu_{\\nu\\sigma} u^\\nu V^\\sigma = 0 ,

so that, in particular, the tangent of a geodesic is *auto-parallel*
(:math:`\\nabla_u u = 0`) exactly when the geodesic equation holds.

These tests all project the covariant derivative onto a curve tangent via
``Tensor.covariant_derivative``; they are skipped until that method is
implemented (the tests act as the contract the implementation must satisfy).

"""
import numpy as np
import pytest
import sympy
from sympy import simplify, symbols

from einsteinpy.symbolic import (
    BaseRelativityTensor,
    ChristoffelSymbols,
    GenericVector,
    MetricTensor,
    simplify_sympy_array,
)
from einsteinpy.symbolic.predefined import MinkowskiCartesian

requires_covariant_derivative = pytest.mark.skipif(
    not hasattr(BaseRelativityTensor, "covariant_derivative"),
    reason="BaseRelativityTensor.covariant_derivative is not implemented yet",
)


def schwarzschild_metric():
    symbolstr = "t r theta phi"
    syms = symbols(symbolstr)
    G, M, c, a = symbols("G M c a")
    # using metric values of Schwarzschild space-time (signature: +---)
    # a is the Schwarzschild radius
    list2d = np.zeros((4, 4), dtype=int).tolist()
    list2d[0][0] = 1 - (a / syms[1])
    list2d[1][1] = -1 / ((1 - (a / syms[1])) * (c ** 2))
    list2d[2][2] = -1 * (syms[1] ** 2) / (c ** 2)
    list2d[3][3] = -1 * (syms[1] ** 2) * (sympy.sin(syms[2]) ** 2) / (c ** 2)
    return MetricTensor(list2d, syms)


def directional_covariant_derivative(tangent, tensor, metric):
    """
    Project the covariant derivative of ``tensor`` onto the curve tangent:
    ``A = u^sigma grad_sigma T`` (configuration of ``tensor`` is preserved).
    """
    grad = tensor.covariant_derivative(metric=metric)
    # grad.config == tensor.config + "l"; the appended (sigma) index is the
    # last axis of the product u (tensor) x grad (tensor)
    contracted = sympy.tensorcontraction(
        sympy.tensorproduct(tangent.tensor(), grad.tensor()),
        (0, grad.order),
    )
    return BaseRelativityTensor(
        contracted,
        syms=tensor.syms,
        config=tensor.config,
        parent_metric=metric,
        variables=tensor.variables,
        functions=tensor.functions,
    )


@requires_covariant_derivative
def test_flat_space_covariant_derivative_is_partial_derivative():
    metric = MinkowskiCartesian()
    syms = metric.symbols()
    V = GenericVector([syms[0], syms[0] * syms[1], syms[2], syms[0] * syms[2]], syms)

    grad = V.covariant_derivative(metric=metric)

    assert grad.config == "ul"
    assert grad.order == 2
    for i in range(4):
        for sigma in range(4):
            assert simplify(grad[i, sigma] - sympy.diff(V[i], syms[sigma])) == 0


@requires_covariant_derivative
def test_shared_vector_is_parallel_transported_in_flat_space():
    metric = MinkowskiCartesian()
    syms = metric.symbols()
    v = symbols("v")
    constants_a = symbols("a b c d")
    u = GenericVector([1, v, 0, 0], syms)
    V = GenericVector(list(constants_a), syms)

    A = directional_covariant_derivative(u, V, metric)

    assert A.config == "u"
    zero_arr = sympy.Array(np.zeros((4,), dtype=int))
    assert simplify_sympy_array(A.tensor()) == zero_arr


@requires_covariant_derivative
def test_metric_compatible_connection_preserves_norms():
    metric = schwarzschild_metric()

    grad_g = metric.covariant_derivative(metric=metric)

    assert grad_g.config == "lll"
    assert grad_g.order == 3
    # nabla_sigma g_mu_nu = 0 is what guarantees the norm of a
    # parallel-transported vector is conserved along the curve
    zero_arr = sympy.Array(np.zeros((4, 4, 4), dtype=int))
    assert simplify_sympy_array(grad_g.tensor()) == zero_arr


@requires_covariant_derivative
def test_geodesic_tangent_is_auto_parallel():
    metric = schwarzschild_metric()
    coords = tuple(symbols("u_t u_r u_th u_ph"))
    u = GenericVector(coords, metric.symbols())

    # u^sigma grad_sigma u^mu with coordinate-independent components equals
    # the left-hand side of the geodesic equation (Gamma^mu_nu_sigma u^nu u^sigma)
    A = directional_covariant_derivative(u, u, metric)
    chl = ChristoffelSymbols.from_metric(metric).tensor()

    assert A.config == "u"
    for mu in range(4):
        expected = sum(
            chl[mu, nu, sigma] * u[nu] * u[sigma]
            for nu in range(4)
            for sigma in range(4)
        )
        assert simplify(A[mu] - expected) == 0


@requires_covariant_derivative
def test_generic_vector_field_is_not_parallel_transported():
    metric = schwarzschild_metric()
    V = GenericVector(symbols("V_t V_r V_th V_ph"), metric.symbols())
    # a purely time-directed observer: u^mu = (1, 0, 0, 0)
    u = GenericVector([1, 0, 0, 0], metric.symbols())

    A = directional_covariant_derivative(u, V, metric)

    # a generic 4-vector is NOT parallel transported along a generic time-like
    # observer; the radial component is
    # A^r = Gamma^r_tt u^t V^t = (a (1 - a/r) c^2 / (2 r^2)) V^t != 0
    chl = ChristoffelSymbols.from_metric(metric).tensor()
    expected_r = chl[1, 0, 0] * u[0] * V[0]
    assert simplify(A[1] - expected_r) == 0
    assert simplify(A[1]) != 0


@requires_covariant_derivative
def test_covariant_derivative_without_metric_raises():
    V = GenericVector(symbols("V_t V_r V_th V_ph"), symbols("t r theta phi"))
    with pytest.raises(ValueError):
        V.covariant_derivative()