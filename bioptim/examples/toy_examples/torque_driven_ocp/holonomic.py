import numpy as np
from bioptim import HolonomicTorqueBiorbdModel, ContactType, Objective, ObjectiveList, ObjectiveFcn, ConstraintList, ConstraintFcn, \
    Node, BoundsList, InitialGuessList, VariableScalingList, OptimalControlProgram, DynamicsOptions, OdeSolver, \
    HolonomicConstraintsList, BiMappingList, Axis, CostType, Solver, HolonomicConstraintsFcn
from casadi import MX, Function, jacobian, DM
from bioptim.examples.utils import ExampleUtils


def track_base_marker_y(controller, marker_name: str):
    q_u = controller.states["q_u"].cx
    q = controller.model.compute_q()(q_u, DM.zeros(controller.model.nb_dependent_joints, 1))
    marker_index = controller.model.marker_index(marker_name)
    return controller.model.marker(marker_index)(q, controller.parameters.cx)[1]  # Axis.Y


def _rigid_contacts_holonomic(model):
    phi, phi_jac, bias_fn = HolonomicConstraintsFcn.rigid_contacts(model)
    q = MX.sym("q_rc", model.nb_q)
    qdot = MX.sym("qdot_rc", model.nb_qdot)
    p = model.parameters  # MX() - empty when no model parameters are defined
    return (
        Function("phi_rc", [q], [phi(q, p)]),
        Function("phi_jac_rc", [q], [phi_jac(q, p)]),
        Function("bias_rc", [q, qdot], [bias_fn(q, qdot, p)]),
    )


def proportional_joint_constraint(pip_idx: int, dip_idx: int, coef: float):
    def make(model):
        q = MX.sym("q", model.nb_q)
        qdot = MX.sym("qdot", model.nb_qdot)
        phi = Function("phi", [q], [q[dip_idx] - coef*q[pip_idx]])
        phi_jac = Function("phi_jac", [q], [jacobian(q[dip_idx] - coef*q[pip_idx], q)])
        bias = Function("bias", [q, qdot], [MX.zeros(1,1)])
        return phi, phi_jac, bias
    return make

def prepare_holonomic_torque_crawl(bio_model_path: str):
    holonomic_constraints = HolonomicConstraintsList()
    holonomic_constraints.add(
        key="middle_pip_dip",
        constraints_fcn=proportional_joint_constraint(pip_idx=10, dip_idx=11, coef=0.849),
    )
    holonomic_constraints.add(
        key="little_pip_dip",
        constraints_fcn=proportional_joint_constraint(pip_idx=13, dip_idx=14, coef=0.849),
    )
    holonomic_constraints.add(
        key="contacts",
        constraints_fcn=_rigid_contacts_holonomic
    )
    bio_model = HolonomicTorqueBiorbdModel(
        bio_model_path,
        holonomic_constraints=holonomic_constraints,
        independent_joint_index=[0,1,3,4,5, 6,8, 10, 13],
        dependent_joint_index=[2,7,9,11,12,14]
    )

    state_mapping = BiMappingList()
    state_mapping.add("q",
                      to_second=[0, 1, None, 2, 3, 4, 5, None, 6, None, 7, None, None, 8, None],
                      to_first=[0, 1, 3, 4, 5, 6, 8, 10, 13])
    state_mapping.add("qdot",
                      to_second=[0, 1, None, 2, 3, 4, 5, None, 6, None, 7, None, None, 8, None],
                      to_first=[0, 1, 3, 4, 5, 6, 8, 10, 13])

    algebraic_mapping = BiMappingList()
    algebraic_mapping.add("q",
                          to_second=[None] * 11 + [0] + [None, None] + [1],
                          to_first=[11,14])

    state_mapping.add("tau",
                    to_second=[None, None, None, None, None, None, 0, None, 1, None, 2, None, None, 3, None],
                    to_first=[6, 8, 10, 13])

    objectives = ObjectiveList()
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=0.001)
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
    q0 = np.zeros(bio_model.nb_independent_joints)
    q0[2] = -0.41
    q0[5] = -0.43
    q0[6] = 1.01
    q0[7] = 0.83
    q0[8] = 0.52

    x_bounds = BoundsList()
    x_bounds.add("q_u", bio_model.bounds_from_ranges("q", mapping=state_mapping))
    x_bounds.add("qdot_u", bio_model.bounds_from_ranges("qdot", mapping=state_mapping))
    x_bounds["q_u"][:, 0] = q0
    x_bounds["qdot_u"][:, 0] = 0
    #x_bounds["qdot_u"][:, -1] = 0

    a_bounds = BoundsList()
    a_bounds.add("q_v", bio_model.bounds_from_ranges("q", mapping=algebraic_mapping))

    x_init = InitialGuessList()
    x_init.add("q_u", q0)
    x_init.add("qdot_u", [0] * bio_model.nb_independent_joints)

    a_init = InitialGuessList()
    a_init.add("q_v", [0.70467, 0.44148])

    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=[-200] * 4, max_bound=[200] * 4)

    return OptimalControlProgram(
        bio_model,
        n_shooting=25,
        phase_time=0.5,
        objective_functions=objectives,
        #constraints=constraints,
        dynamics=DynamicsOptions(ode_solver=OdeSolver.COLLOCATION(polynomial_degree=3)),
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        #a_bounds=a_bounds,
        x_init=x_init,
        #a_init=a_init,
        variable_mappings=state_mapping,
        n_threads=14
    )


def main():
    ocp = prepare_holonomic_torque_crawl("../../models/holonomic_three_finger_crawl.bioMod")
    # ocp.print(to_console=True, to_graph=False)
    ocp.add_plot_penalty(CostType.CONSTRAINTS)
    solver = Solver.IPOPT()
    solver.set_maximum_iterations(1000)
    sol = ocp.solve(solver)
    sol.print_cost()
    sol.animate(n_frames=100)
    sol.graphs()


if __name__ == "__main__":
    main()

