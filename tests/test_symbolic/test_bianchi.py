import pytest

from einsteinpy.symbolic import MetricTensor
from einsteinpy.symbolic.bianchi import verify_bianchi_identity
from einsteinpy.symbolic.predefined import (
    AntiDeSitter,
    AntiDeSitterStatic,
    DeSitter,
    Minkowski,
    MinkowskiPolar,
    Schwarzschild,
)

BIANCHI_METRICS = [
    Minkowski(),
    MinkowskiPolar(),
    Schwarzschild(),
    DeSitter(),
    AntiDeSitter(),
    AntiDeSitterStatic(),
]


@pytest.mark.parametrize("metric", BIANCHI_METRICS)
def test_contracted_bianchi_holds_for_predefined_metrics(metric):
    assert verify_bianchi_identity(metric)


def test_contracted_bianchi_raises_type_error_for_non_metric():
    with pytest.raises(TypeError):
        verify_bianchi_identity("not a metric")