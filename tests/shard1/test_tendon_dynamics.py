from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from casadi import DM, Function, MX

from bioptim import DynamicsFunctions, TendonBiorbdModel
from tests.utils import TestUtils


class FakeNlp:
    def __init__(self, model):
        self.model = model
        self.controls = {"tendons": [0] * model.nb_tendons}
        self.parameters = SimpleNamespace(cx=MX())
        self.cx = MX


def _evaluate_tau_from_tendons(nlp, q_value, qdot_value, tendon_pull_forces_value):
    q = MX.sym("q", nlp.model.nb_q, 1)
    qdot = MX.sym("qdot", nlp.model.nb_qdot, 1)
    tendon_pull_forces = MX.sym("tendon_pull_forces", nlp.model.nb_tendons, 1)

    tau = DynamicsFunctions.compute_tau_from_tendons(nlp, q, qdot, tendon_pull_forces)
    tau_fun = Function("tau_from_tendons", [q, qdot, tendon_pull_forces], [tau])
    return tau_fun(q_value, qdot_value, tendon_pull_forces_value)


@pytest.mark.parametrize(
    "q_value,qdot_value,tendon_pull_forces_value,expected_tau",
    [
        (DM([0.0]), DM([0.0]), DM([0.0]), 0.0),
        (DM([0.0]), DM([0.0]), DM([400.0]), 0.0),
        (DM([np.pi / 2]), DM([0.0]), DM([10.0]), 10.0 * np.cos(np.pi / 4) * 0.5),
        (DM([0.2]), DM([0.0]), DM([10.0]), 0.499167),
    ],
)
def test_compute_tau_from_tendons_matches_expected_values(
    q_value, qdot_value, tendon_pull_forces_value, expected_tau
):
    model_path = TestUtils.bioptim_folder() + "/examples/models/tendon_manipulator.bioMod"
    model = TendonBiorbdModel(model_path)
    nlp = FakeNlp(model)

    tau = _evaluate_tau_from_tendons(nlp, q_value, qdot_value, tendon_pull_forces_value)

    np.testing.assert_allclose(np.array(tau).squeeze(), expected_tau, rtol=1e-6, atol=1e-6)
