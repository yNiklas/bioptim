import math

import numpy as np
from casadi import DM, Function, MX, vertcat
import bioviz
from bioptim import DynamicsFunctions, TendonBiorbdModel
from bioptim.examples.utils import ExampleUtils
from types import SimpleNamespace

class FakeNlp:
    def __init__(self, model):
        self.model = model
        self.controls = {"tendons": [0] * model.nb_tendons}
        self.parameters = SimpleNamespace(cx=MX())
        self.cx = MX

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

def main():
    model_path = ExampleUtils.folder + "/models/tendon_test_finger.bioMod"
    model = TendonBiorbdModel(model_path)
    nlp = FakeNlp(model)

    n_shooting = 100

    states_sym = MX.sym("Q_Qdot", model.nb_q + model.nb_qdot, 1)
    q_sym = states_sym[:model.nb_q]
    qdot_sym = states_sym[model.nb_q:]
    controls_sym = MX.sym("Tendons", 1, 1)

    # Calculate the joint torques from tendons
    tau_dyn = DynamicsFunctions.compute_tau_from_tendons(
        nlp,
        q_sym,
        qdot_sym,
        controls_sym
    )
    # Add remaining torques from the ligaments
    tau_dyn += DynamicsFunctions.collect_tau(nlp, q_sym, qdot_sym, nlp.parameters.cx)

    rhs = vertcat(
        states_sym[model.nb_q :],
        nlp.model.forward_dynamics()(states_sym[: model.nb_q], states_sym[model.nb_q :], tau_dyn, [], [])
    )
    dyn_fun = Function(
        "dynamics",
        [states_sym, controls_sym],
        [rhs]
    )

    time_between_nodes = 0.02
    n_steps = 5
    times = [time_between_nodes*i for i in range(n_shooting)]
    q_start_value = DM([0., 0.])
    qdot_start_value = DM([0.0, 0.0])
    start_states = vertcat(q_start_value, qdot_start_value)
    decline = np.array([10 * math.cos((t/times[-1])*math.pi) for t in times])
    decline = np.maximum(decline, np.array([0 for _ in times]))
    #decline = ([7] * (n_shooting//2)) + ([1] * (n_shooting//2))
    #decline = ([7] * (n_shooting//2)) + [6*(n_shooting-i)/n_shooting for i in range(n_shooting//2, n_shooting)]
    decline = ([20] * (n_shooting//2)) + [6*(n_shooting-i)/n_shooting for i in range(n_shooting//2, n_shooting)]
    #decline = [0.1] * n_shooting
    controls = DM([decline])
    print(controls)

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
    
    q_integrated = np.array(states_integrated[:model.nb_q, :])
    
    # Try visualizing with bioviz
    b = bioviz.Viz(model_path)
    b.load_movement(q_integrated)
    b.exec()

if __name__ == "__main__":
    main()
