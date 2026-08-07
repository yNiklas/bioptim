import bioviz
import numpy as np
from bioptim import HolonomicTorqueBiorbdModel, ContactType, Objective, ObjectiveList, ObjectiveFcn, ConstraintList, \
    ConstraintFcn, \
    Node, BoundsList, InitialGuessList, VariableScalingList, OptimalControlProgram, DynamicsOptions, OdeSolver, \
    HolonomicConstraintsList, BiMappingList, Axis, CostType, Solver, HolonomicConstraintsFcn, SolutionMerge
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
    q_u = controller.states["q_u"].cx
    q_v_init = getattr(controller.model, "q_v_init_guess", DM.zeros(controller.model.nb_dependent_joints, 1))
    q = controller.model.compute_q()(q_u, q_v_init)
    marker_index = controller.model.marker_index(marker_name)
    return controller.model.marker(marker_index)(q, controller.parameters.cx)[1]  # Axis.Y


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
    constraints.add(
        ConstraintFcn.BOUND_STATE,
        key="qdot_u",
        index=[1],
        min_bound=[-5],
        max_bound=[-0.02],
        node=Node.END
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

def main():
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

if __name__ == "__main__":
    main()
