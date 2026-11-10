import sympy

from einsteinpy.symbolic.christoffel import ChristoffelSymbols
from einsteinpy.symbolic.helpers import simplify_sympy_array
from einsteinpy.symbolic.metric import MetricTensor
from einsteinpy.symbolic.ricci import RicciScalar, RicciTensor


def verify_bianchi_identity(metric):
    """
    Verify the contracted Bianchi identity for a given metric tensor.

    The contracted Bianchi identity states that

    .. math::

        \\nabla_\\rho R^\\rho{}_\\mu = \\frac{1}{2} \\nabla_\\mu R

    where :math:`R^\\rho{}_\\mu` is the Ricci tensor (with mixed indices),
    :math:`R` is the Ricci scalar and :math:`\\nabla` is the covariant
    derivative.

    Parameters
    ----------
    metric : ~einsteinpy.symbolic.metric.MetricTensor
        Metric Tensor

    Returns
    -------
    bool
        ``True`` if the metric satisfies the contracted Bianchi identity,
        ``False`` otherwise.

    Raises
    ------
    TypeError
        Raised when ``metric`` is not an instance of
        ~einsteinpy.symbolic.metric.MetricTensor

    """
    if not isinstance(metric, MetricTensor):
        raise TypeError("metric should be an instance of MetricTensor")

    syms = metric.symbols()
    dims = metric.dims

    chris = ChristoffelSymbols.from_metric(metric)
    christoffels = chris.tensor()

    ricci = RicciTensor.from_metric(metric).change_config("ul", metric=metric)
    ricci_arr = ricci.tensor()

    ricci_scalar = RicciScalar.from_metric(metric).expr

    # Left-hand side: covariant divergence of Ricci tensor, nabla_rho R^rho_mu
    lhs = [sympy.S.Zero] * dims
    for mu in range(dims):
        term = sympy.S.Zero
        for rho in range(dims):
            term += sympy.diff(ricci_arr[rho, mu], syms[rho])
            for sigma in range(dims):
                term += christoffels[rho, rho, sigma] * ricci_arr[sigma, mu]
                term -= christoffels[sigma, rho, mu] * ricci_arr[rho, sigma]
        lhs[mu] = term

    # Right-hand side: half the gradient of the Ricci scalar, (1/2) nabla_mu R
    rhs = [sympy.Rational(1, 2) * sympy.diff(ricci_scalar, syms[mu]) for mu in range(dims)]

    difference = simplify_sympy_array(sympy.Array(lhs) - sympy.Array(rhs))

    return all(sympy.simplify(component) == 0 for component in difference)
