import sympy

from einsteinpy.symbolic.christoffel import ChristoffelSymbols
from einsteinpy.symbolic.metric import MetricTensor
from einsteinpy.symbolic.ricci import RicciScalar, RicciTensor


def bianchi_identity(metric):
    """
    Checks whether a given Metric Tensor satisfies the Bianchi identity:

    .. math::
        \\nabla_\\rho R^\\rho_\\mu = \\frac{1}{2} \\nabla_\\mu R

    Parameters
    ----------
    metric : ~einsteinpy.symbolic.metric.MetricTensor
        Metric Tensor to be verified

    Returns
    -------
    bool
        True if the Bianchi identity is satisfied, False otherwise

    Raises
    ------
    TypeError
        Raised when metric is not an instance of MetricTensor

    """
    if not isinstance(metric, MetricTensor):
        raise TypeError("metric should be an instance of MetricTensor")

    dims = metric.dims
    syms = metric.symbols()
    chris = ChristoffelSymbols.from_metric(metric)
    ch_arr = chris.tensor()
    ricci = RicciTensor.from_christoffels(chris, parent_metric=metric).change_config(
        "ul"
    )
    r_arr = ricci.tensor()
    r_scalar = RicciScalar.from_riccitensor(ricci, parent_metric=metric)
    r_val = r_scalar.expr

    for mu in range(dims):
        lhs = 0
        for rho in range(dims):
            d_rho = sympy.diff(r_arr[rho, mu], syms[rho])
            term2 = sum(ch_arr[rho, rho, lam] * r_arr[lam, mu] for lam in range(dims))
            term3 = sum(ch_arr[lam, rho, mu] * r_arr[rho, lam] for lam in range(dims))
            lhs += d_rho + term2 - term3
        rhs = sympy.Rational(1, 2) * sympy.diff(r_val, syms[mu])
        diff = sympy.simplify(lhs - rhs)
        if diff != 0:
            return False
    return True


check_bianchi = bianchi_identity
verify_bianchi_identity = bianchi_identity
bianchi = bianchi_identity
