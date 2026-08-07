import bioviz
import numpy as np
from bioptim import HolonomicTorqueBiorbdModel, ContactType, Objective, ObjectiveList, ObjectiveFcn, ConstraintList, \
    ConstraintFcn, \
    Node, BoundsList, InitialGuessList, VariableScalingList, OptimalControlProgram, DynamicsOptions, OdeSolver, \
    HolonomicConstraintsList, BiMappingList, Axis, CostType, Solver, HolonomicConstraintsFcn, SolutionMerge, \
    PhaseTransitionList, PhaseTransitionFcn, DynamicsOptionsList
from casadi import MX, Function, jacobian, DM
from bioptim.examples.utils import ExampleUtils
from bioptim.models.biorbd.model_dynamics import HolonomicTendonBiorbdModel


def proportional_joint_constraint(pip_idx: int, dip_idx: int, coef: float):
    def make(model):
        q = MX.sym("q", model.nb_q)
        qdot = MX.sym("qdot", model.nb_qdot)
        phi = Function("phi", [q], [q[dip_idx] - coef*q[pip_idx]])
        phi_jac = Function("phi_jac", [q], [jacobian(q[dip_idx] - coef*q[pip_idx], q)])
        bias = Function("bias", [q, qdot], [MX.zeros(1,1)])
        return phi, phi_jac, bias
    return make

def track_base_marker_y(controller, marker_name: str):
    return track_marker(controller, marker_name, axis=Axis.Y)

def track_marker(controller, marker_name: str, axis: int):
    q_u = controller.states["q_u"].cx
    q_v_init = getattr(controller.model, "q_v_init_guess", DM.zeros(controller.model.nb_dependent_joints, 1))
    q = controller.model.compute_q()(q_u, q_v_init)
    marker_index = controller.model.marker_index(marker_name)
    return controller.model.marker(marker_index)(q, controller.parameters.cx)[axis]

def bound_marker(controller, marker_name: str, axis: int):
    # Return the raw (symbolic) marker coordinate; the bounds are enforced by
    # the constraint's own min_bound/max_bound.
    return track_marker(controller, marker_name, axis)

def prepare_holonomic_tendon_crawl(
    bio_model_path: str,
    use_contacts: bool = False,
    n_threads: int = 1,
):
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
        independent_joint_index=[0,1,2,3,4,5, 6,7,8, 9,10, 12,13],
        dependent_joint_index=[11,14],
        contact_types=[ContactType.RIGID_EXPLICIT] if use_contacts else [],
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
    objectives.add(track_base_marker_y, custom_type=ObjectiveFcn.Mayer, marker_name="base_contact_right_marker", weight=5)

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

    # Starting posture (fingers pre-flexed and in contact with the ground).
    q0 = [
        0.0, 0.0, 0.0271, -0.41, 0.0, 0.0,
        -0.43, 0.86, 1.01,
        #0.46, 0.83, 0.70467, # idea: 0.47, 0.91, 0.77259
        0.47, 0.91, 0.77259,
        #0.52, 0.52, 0.44148, # 0.69, 0.44, 0.37356
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

def prepare_two_phase_holonomic_crawl(bio_model_path: str, no_contact_bio_model_path: str, n_threads=2):
    holonomic_constraints = HolonomicConstraintsList()
    holonomic_constraints.add(
        key="middle_pip_dip",
        constraints_fcn=proportional_joint_constraint(pip_idx=10, dip_idx=11, coef=0.849),
    )
    holonomic_constraints.add(
        key="little_pip_dip",
        constraints_fcn=proportional_joint_constraint(pip_idx=13, dip_idx=14, coef=0.849),
    )

    bio_model = (
        HolonomicTendonBiorbdModel(
            bio_model_path,
            holonomic_constraints=holonomic_constraints,
            independent_joint_index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13],
            dependent_joint_index=[11, 14],
            contact_types=[ContactType.RIGID_EXPLICIT],
            torque_driven_dofs=["thumb_proxy_RotY"]
        ),
        HolonomicTendonBiorbdModel(
            no_contact_bio_model_path,
            holonomic_constraints=holonomic_constraints,
            independent_joint_index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13],
            dependent_joint_index=[11, 14],
            contact_types=[ContactType.RIGID_EXPLICIT],
            torque_driven_dofs=["thumb_proxy_RotY"]
        )
    )

    state_mapping = BiMappingList()
    state_mapping.add("q",
                      to_second=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None, 11, 12, None],
                      to_first=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13])
    state_mapping.add("qdot",
                      to_second=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None, 11, 12, None],
                      to_first=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13])

    objectives = ObjectiveList()
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tendons", weight=0.001, phase=0)
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tendons", weight=0.001, phase=1)
    # objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="non_tendon_tau", weight=0.01, phase=0)
    # objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="non_tendon_tau", weight=0.01, phase=1)
    objectives.add(track_base_marker_y, custom_type=ObjectiveFcn.Mayer, marker_name="base_contact_right_marker", weight=5,
                   phase=0)
    objectives.add(track_base_marker_y, custom_type=ObjectiveFcn.Mayer, marker_name="base_contact_right_marker",
                   weight=5,
                   phase=1)
    # objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", index=[5], weight=40, phase=0)
    # objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", index=[5], weight=40, phase=1)

    constraints = ConstraintList()
    constraints.add(  # base_contact_right
        ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES,
        min_bound=0,
        max_bound=np.inf,
        node=Node.ALL,
        contact_index=0,
        phase=0
    )
    constraints.add(  # thumb
        ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES,
        min_bound=0,
        max_bound=np.inf,
        node=Node.ALL,
        contact_index=1,
        phase=0
    )
    constraints.add(  # middle finger
        ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES,
        min_bound=0,
        max_bound=np.inf,
        node=Node.ALL,
        contact_index=4,
        phase=0
    )
    constraints.add(  # little finger
        ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES,
        min_bound=0,
        max_bound=np.inf,
        node=Node.ALL,
        contact_index=5,
        phase=0
    )
    constraints.add(  # base_contact_right
        ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES,
        min_bound=0,
        max_bound=np.inf,
        node=Node.ALL,
        contact_index=2,
        phase=1
    )
    #constraints.add(
    #    ConstraintFcn.BOUND_STATE,
    #    key="q",
    #    index=[1],
    #    min_bound=[-5],
    #    max_bound=[0.05],
    #    phase=1,
    #    node=Node.ALL
    #)
    constraints.add(  # thumb contact
        ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES,
        min_bound=0,
        max_bound=np.inf,
        node=Node.ALL,
        contact_index=3,
        phase=1
    )
    constraints.add(  # little finger
        ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES,
        min_bound=0,
        max_bound=np.inf,
        node=Node.ALL,
        contact_index=4,
        phase=1
    )
    constraints.add( # Don't penetrate ground
        bound_marker,
        marker_name="middle_endeffector",
        node=Node.ALL,
        phase=1,
        axis=Axis.Z,
        min_bound=0,
        max_bound=np.inf,
    )
    constraints.add(  # Place middle finger to the ground for the contact establishment
        track_marker,
        marker_name="middle_endeffector",
        node=Node.END,
        phase=1,
        axis=Axis.Z
    )
    constraints.add(
        ConstraintFcn.BOUND_STATE,
        key="q_u",
        index=[9],
        min_bound=[0],
        max_bound=[0.3],
        node=Node.MID,
        phase=1
    )
    constraints.add(
        ConstraintFcn.BOUND_STATE,
        key="q_u",
        index=[10],
        min_bound=[0],
        max_bound=[0.3],
        node=Node.MID,
        phase=1
    )

    q0 = [
        0.0, 0.0, 0.0271, -0.41, 0.0, 0.0,
        -0.43, 0.86, 1.01,
        # 0.46, 0.83, 0.70467, # idea: 0.47, 0.91, 0.77259
        0.47, 0.91, 0.77259,
        # 0.52, 0.52, 0.44148, # 0.69, 0.44, 0.37356
        0.69, 0.44, 0.37356
    ]
    q0_u = q0[:11] + q0[12:14]
    q0_v = [q0[11], q0[14]]
    bio_model[0].q_v_init_guess = DM(q0_v)
    n_non_tendon = len(bio_model[0].non_tendon_tau_indices)

    x_bounds = BoundsList()
    x_bounds.add("q_u", bio_model[0].bounds_from_ranges("q", mapping=state_mapping), phase=0)
    x_bounds.add("q_u", bio_model[1].bounds_from_ranges("q", mapping=state_mapping), phase=1)
    x_bounds.add("qdot_u", bio_model[0].bounds_from_ranges("qdot", mapping=state_mapping), phase=0)
    x_bounds.add("qdot_u", bio_model[1].bounds_from_ranges("qdot", mapping=state_mapping), phase=1)
    x_bounds[0]["q_u"][:, 0] = q0_u
    x_bounds[0]["qdot_u"][:, 0] = 0
    x_bounds[0]["qdot_u"][:6, -1] = 0
    x_bounds[1]["qdot_u"][:6, -1] = 0

    x_init = InitialGuessList()
    x_init.add("q_u", q0_u, phase=0)
    x_init.add("qdot_u", [0] * bio_model[0].nb_independent_joints, phase=0)

    u_bounds = BoundsList()
    u_bounds.add("tendons", min_bound=[0] * bio_model[0].nb_tendons, max_bound=[200] * bio_model[0].nb_tendons, phase=0)
    u_bounds.add("tendons", min_bound=[0] * bio_model[1].nb_tendons, max_bound=[200] * bio_model[0].nb_tendons, phase=1)
    u_bounds.add("non_tendon_tau", min_bound=[-10] * n_non_tendon, max_bound=[10] * n_non_tendon, phase=0)
    u_bounds.add("non_tendon_tau", min_bound=[-10] * n_non_tendon, max_bound=[10] * n_non_tendon, phase=1)

    u_init = InitialGuessList()
    u_init.add("tendons", [5] * bio_model[0].nb_tendons, phase=0)
    u_init.add("non_tendon_tau", [0] * n_non_tendon, phase=0)

    phase_transitions = PhaseTransitionList()
    phase_transitions.add(PhaseTransitionFcn.CONTINUOUS, phase_pre_idx=0)

    dynamics = DynamicsOptionsList()
    dynamics.add(DynamicsOptions(ode_solver=OdeSolver.COLLOCATION(polynomial_degree=3)), phase=0)
    dynamics.add(DynamicsOptions(ode_solver=OdeSolver.COLLOCATION(polynomial_degree=3)), phase=1)

    return bio_model, OptimalControlProgram(
        bio_model,
        n_shooting=[26, 26],
        phase_time=(1, 0.5),  # idea: last phase time up
        objective_functions=objectives,
        constraints=constraints,
        dynamics=dynamics,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        u_init=u_init,
        phase_transitions=phase_transitions,
        variable_mappings=state_mapping,
        n_threads=n_threads
    )


def single_phase_main():
    model_path = ExampleUtils.folder + "/models/holonomic_three_finger_crawl.bioMod"
    bio_model, ocp = prepare_holonomic_tendon_crawl(
        model_path,
        use_contacts=True,
        n_threads=14,
    )
    #ocp.check_conditioning()
    # ocp.print(to_console=True, to_graph=False)
    ocp.add_plot_penalty(CostType.CONSTRAINTS)
    solver = Solver.IPOPT()
    #solver.set_check_derivatives_for_naninf(True)
    #solver.set_option_unsafe("first-order", "derivative_test")
    #solver.set_option_unsafe("yes", "derivative_test_print_all")
    solver.set_maximum_iterations(2000)
    sol = ocp.solve(solver)
    sol.print_cost()
    sol.graphs(automatically_organize=False)
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    q = bio_model.compute_q_from_u_iterative(states["q_u"])
    viz = bioviz.Viz(model_path)
    viz.load_movement(q)
    viz.exec()

def two_phase_main():
    model_path = ExampleUtils.folder + "/models/holonomic_three_finger_crawl.bioMod"
    model_path_no_contact = ExampleUtils.folder + "/models/holonomic_three_finger_crawl_no_contact.bioMod"
    bio_model, ocp = prepare_two_phase_holonomic_crawl(
        model_path,
        model_path_no_contact,
        n_threads=14,
    )
    #ocp.check_conditioning()
    # ocp.print(to_console=True, to_graph=False)
    ocp.add_plot_penalty(CostType.CONSTRAINTS)
    solver = Solver.IPOPT()
    #solver.set_check_derivatives_for_naninf(True)
    #solver.set_option_unsafe("first-order", "derivative_test")
    #solver.set_option_unsafe("yes", "derivative_test_print_all")
    solver.set_maximum_iterations(1000)
    sol = ocp.solve(solver)
    sol.print_cost()
    sol.graphs(automatically_organize=False)
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    q = bio_model[0].compute_q_from_u_iterative(states["q_u"])
    viz = bioviz.Viz(model_path)
    viz.load_movement(q)
    viz.exec()

if __name__ == "__main__":
    #single_phase_main()
    two_phase_main()
