import signal
from functools import partial
from pathlib import Path

import bioviz
import numpy as np
from bioptim import ContactType, ObjectiveList, ObjectiveFcn, ConstraintList, \
    ConstraintFcn, \
    Node, BoundsList, InitialGuessList, OptimalControlProgram, DynamicsOptions, OdeSolver, \
    HolonomicConstraintsList, BiMappingList, Axis, CostType, Solver, SolutionMerge, \
    PhaseTransitionList, PhaseTransitionFcn, DynamicsOptionsList, PenaltyController, BiMapping, \
    MultinodeObjectiveList
from casadi import MX, Function, jacobian, DM, vertcat
from bioptim.examples.toy_examples.tendon_driven_ocp.holonomic_tendon import marker_position, proportional_joint_constraint
from bioptim.examples.utils import ExampleUtils, IterationsControllerCallback
from bioptim.gui.online_callback_abstract import OnlineCallbackAbstract
from bioptim.limits.multinode_penalty import MultinodePenaltyFunctions
from bioptim.models.biorbd.model_dynamics import HolonomicTendonBiorbdModel

def prepare_left_steering_ocp(bio_model_path: str, n_threads: int = 8):
    holonomic_constraints = HolonomicConstraintsList()
    holonomic_constraints.add(
        key="middle_pip_dip",
        constraints_fcn=proportional_joint_constraint(pip_idx=10, dip_idx=11, coef=0.849),
    )
    holonomic_constraints.add(
        key="little_pip_dip",
        constraints_fcn=proportional_joint_constraint(pip_idx=13, dip_idx=14, coef=0.849),
    )

    bio_model = HolonomicTendonBiorbdModel(
            bio_model_path,
            holonomic_constraints=holonomic_constraints,
            independent_joint_index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13],
            dependent_joint_index=[11, 14],
            contact_types=[ContactType.RIGID_EXPLICIT],
            torque_driven_dofs=["thumb_proxy_RotY"]
        )

    state_mapping = BiMappingList()
    state_mapping.add("q",
                      to_second=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None, 11, 12, None],
                      to_first=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13])
    state_mapping.add("qdot",
                      to_second=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None, 11, 12, None],
                      to_first=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13])

    objectives = ObjectiveList()
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tendons", weight=0.001)
    objectives.add(marker_position, custom_type=ObjectiveFcn.Mayer, marker_name="base_contact_right_marker",
                   axis=Axis.Y, target=0, quadratic=True, weight=5)

    constraints = ConstraintList()
    constraints.add(  # base_contact_right
        ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES,
        min_bound=0,
        max_bound=np.inf,
        node=Node.ALL,
        contact_index=0,
    )
    constraints.add(  # thumb
        ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES,
        min_bound=0,
        max_bound=np.inf,
        node=Node.ALL,
        contact_index=1,
    )
    constraints.add(  # middle finger
        ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES,
        min_bound=0,
        max_bound=np.inf,
        node=Node.ALL,
        contact_index=4,
    )
    constraints.add(  # little finger
        ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES,
        min_bound=0,
        max_bound=np.inf,
        node=Node.ALL,
        contact_index=5,
    )

    q0 = [
        0.0, 0.0, 0.0271, -0.41, 0.0, 0.0,
        -0.43, 0.86, 1.01,
        0.47, 0.91, 0.77259,
        0.69, 0.44, 0.37356
    ]
    q0_u = q0[:11] + q0[12:14]
    q0_v = [q0[11], q0[14]]
    bio_model.q_v_init_guess = DM(q0_v)

    x_bounds = BoundsList()
    x_bounds.add("q_u", bio_model.bounds_from_ranges("q", mapping=state_mapping))
    x_bounds.add("qdot_u", bio_model.bounds_from_ranges("qdot", mapping=state_mapping))
    x_bounds["q_u"][:, 0] = q0_u
    x_bounds["qdot_u"][:, 0] = 0
    x_bounds["qdot_u"][:, -1] = 0

    x_init = InitialGuessList()
    x_init.add("q_u", q0_u)
    x_init.add("qdot_u", [0] * bio_model.nb_independent_joints)

    u_bounds = BoundsList()
    u_bounds.add("tendons", min_bound=[0] * bio_model.nb_tendons, max_bound=[200] * bio_model.nb_tendons)
    u_bounds.add("non_tendon_tau", min_bound=[-20], max_bound=[20])

    return bio_model, OptimalControlProgram(
        bio_model,
        n_shooting=40,
        phase_time=1,
        objective_functions=objectives,
        constraints=constraints,
        dynamics=DynamicsOptions(ode_solver=OdeSolver.COLLOCATION(polynomial_degree=3)),
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        variable_mappings=state_mapping,
        n_threads=n_threads
    )


def prepare_soft_steering_ocp(bio_model_path: str, n_threads=8):
    holonomic_constraints = HolonomicConstraintsList()
    holonomic_constraints.add(
        key="middle_pip_dip",
        constraints_fcn=proportional_joint_constraint(pip_idx=10, dip_idx=11, coef=0.849),
    )
    holonomic_constraints.add(
        key="little_pip_dip",
        constraints_fcn=proportional_joint_constraint(pip_idx=13, dip_idx=14, coef=0.849),
    )
    bio_model = HolonomicTendonBiorbdModel(
        bio_model_path,
        holonomic_constraints=holonomic_constraints,
        independent_joint_index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13],
        dependent_joint_index=[11, 14],
        contact_types=[],
        torque_driven_dofs=["thumb_proxy_RotY"]
    )

    state_mapping = BiMappingList()
    state_mapping.add("q",
                      to_second=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None, 11, 12, None],
                      to_first=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13])
    state_mapping.add("qdot",
                      to_second=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None, 11, 12, None],
                      to_first=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13])

    objectives = ObjectiveList()
    #objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tendons", weight=0.001)
    objectives.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q_u", index=5, target=0.5, quadratic=True, weight=40)

    constraints = ConstraintList()

    # Starting posture (fingers pre-flexed and in contact with the ground).
    q0 = [
        0.0, 0.0, 0.0235, -0.41, 0.0, 0.0,
        -0.43, 0.86, 1.01,
        0.47, 0.91, 0.77259,
        0.69, 0.44, 0.37356
    ]
    q0_u = q0[:11] + q0[12:14]
    q0_v = [q0[11], q0[14]]
    bio_model.q_v_init_guess = DM(q0_v)

    x_bounds = BoundsList()
    x_bounds.add("q_u", bio_model.bounds_from_ranges("q", mapping=state_mapping))
    x_bounds.add("qdot_u", bio_model.bounds_from_ranges("qdot", mapping=state_mapping))
    x_bounds["q_u"][:, 0] = q0_u
    x_bounds["qdot_u"][:, 0] = 1e-10
    x_bounds["qdot_u"][:, -1] = 0

    x_init = InitialGuessList()
    x_init.add("q_u", q0_u)
    x_init.add("qdot_u", [1e-10] * bio_model.nb_independent_joints)

    u_bounds = BoundsList()
    u_bounds.add("tendons", min_bound=[0] * bio_model.nb_tendons, max_bound=[200] * bio_model.nb_tendons)
    u_bounds.add("non_tendon_tau", min_bound=[-20], max_bound=[20])

    u_init = InitialGuessList()
    u_init.add("tendons", [2.2993, 18.9657, 1.7572])
    u_init.add("non_tendon_tau", [-0.0024])

    return bio_model, OptimalControlProgram(
        bio_model,
        n_shooting=40,
        phase_time=1,
        objective_functions=objectives,
        constraints=constraints,
        dynamics=DynamicsOptions(ode_solver=OdeSolver.COLLOCATION(polynomial_degree=3)),
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        u_init=u_init,
        variable_mappings=state_mapping,
        n_threads=n_threads
    )


def soft_contact_main():
    model_path = str(Path(__file__).with_name("holonomic_soft_contact_three_finger.bioMod"))
    bio_model, ocp = prepare_soft_steering_ocp(
        model_path,
        n_threads=8,
    )
    ocp.add_plot_penalty(CostType.CONSTRAINTS)
    solver = Solver.IPOPT()
    solver.set_maximum_iterations(1_000_000)
    ocp.set_ocp_solver(solver)
    ocp.ocp_solver.options_common["iteration_callback"] = IterationsControllerCallback(ocp, budget=5000, default_extension=500)
    sol = ocp.solve(solver)
    sol.print_cost()
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    q = bio_model.compute_q_from_u_iterative(states["q_u"])
    viz = bioviz.Viz(model_path)
    viz.load_movement(q)
    viz.exec()
    sol.graphs(automatically_organize=False)
    ExampleUtils.save_solution(ocp, sol)
    ExampleUtils.save_control_data(ocp, sol, "solutions/soft_contact_steer.npz")

if __name__ == "__main__":
    soft_contact_main()
