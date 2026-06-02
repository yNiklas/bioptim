from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from casadi import DM, Function, MX, vertcat

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

def _integrate_RK4(times, start_states, controls, dyn_fun, time_between_nodes, n_shooting, n_steps, nb_q, nb_qdot):
    integrated_states = DM.zeros((nb_q+nb_qdot, n_shooting+1))
    integrated_states[:, 0] = start_states
    h = time_between_nodes / n_steps
    for i in range(n_shooting):
        state_at_i = integrated_states[:, i]
        control_at_i = controls[:, i]
        for j in range(n_steps):
            k1 = dyn_fun(state_at_i, control_at_i)
            k2 = dyn_fun(state_at_i + h/2 * k1, control_at_i)
            k3 = dyn_fun(state_at_i + h/2 * k2, control_at_i)
            k4 = dyn_fun(state_at_i + h * k3, control_at_i)
            state_at_i += h/6 * (k1 + 2 * k2 + 2 * k3 + k4)
        integrated_states[:, i+1] = state_at_i
    return integrated_states


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

@pytest.mark.parametrize(
    "q_start_value,qdot_start_value,tendon_pull_forces_value",
    [
        (DM([0.0, 0.0]), DM([0.0, 0.0]), DM([[50]*10])),
    ]
)
def test_dynamics_explicitly(q_start_value, qdot_start_value, tendon_pull_forces_value):
    model_path = TestUtils.bioptim_folder() + "/examples/models/tendon_test_finger.bioMod"
    model = TendonBiorbdModel(model_path)
    nlp = FakeNlp(model)

    states_sym = MX.sym("Q_Qdot", model.nb_q + model.nb_qdot, 1)
    controls_sym = MX.sym("Tendons", 1, 1)
    tau_dyn = DynamicsFunctions.compute_tau_from_tendons(
        nlp,
        states_sym[:model.nb_q],
        states_sym[model.nb_q:],
        controls_sym
    )
    rhs = vertcat(
        states_sym[model.nb_q :],
        nlp.model.forward_dynamics()(states_sym[: model.nb_q], states_sym[model.nb_q :], tau_dyn, [], [])
    )
    dyn_fun = Function(
        "dynamics",
        [states_sym, controls_sym],
        [rhs]
    )

    time_between_nodes = 0.05
    n_shooting = 10
    n_steps = 5
    times = [time_between_nodes*i for i in range(n_shooting)]
    start_states = vertcat(q_start_value, qdot_start_value)
    controls = tendon_pull_forces_value

    states_integrated = _integrate_RK4(
        times,
        start_states,
        controls,
        dyn_fun,
        time_between_nodes,
        n_shooting,
        n_steps,
        model.nb_q,
        model.nb_qdot
    )

    print("Integrated states:\n", states_integrated)
    np.testing.assert_almost_equal(np.array(states_integrated), np.array([[0]*(n_shooting+1)]*(model.nb_q+model.nb_qdot)))
