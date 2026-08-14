import numpy as np

from llm_router.utils.calibration import apply_platt, fit_platt


def test_constant_platt_target_returns_prevalence():
    parameters = fit_platt(np.array([-2.0, 2.0]), np.array([1.0, 1.0]))
    probabilities = apply_platt(np.array([[-100.0], [100.0]]), [parameters])
    assert parameters[0] == 0.0
    assert np.allclose(probabilities, 0.9999)


def test_apply_platt_is_monotonic():
    probabilities = apply_platt(np.array([[-2.0], [0.0], [2.0]]), [(1.0, 0.0)])[:, 0]
    assert np.all(np.diff(probabilities) > 0)
