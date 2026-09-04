import signal
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
from bioptim.examples.utils import ExampleUtils, IterationsControllerCallback
from bioptim.gui.online_callback_abstract import OnlineCallbackAbstract
from bioptim.limits.multinode_penalty import MultinodePenaltyFunctions
from bioptim.models.biorbd.model_dynamics import HolonomicTendonBiorbdModel


def displacement_aware_cyclic_phase_transition(controllers: list[PenaltyController, PenaltyController],
                                               stride_dof_index: int = 1, stride: float = -0.05,
                                               states_mapping: list[BiMapping] = None) -> MX:
    states_mapping = MultinodePenaltyFunctions.Functions._prepare_states_mapping(controllers, states_mapping)
    end_states = states_mapping[0].to_second.map(controllers[0].states["all"].cx)
    start_states = states_mapping[0].to_first.map(controllers[1].states["all"].cx)
    constraint_violation = start_states - end_states
    return vertcat(constraint_violation[2:], constraint_violation[stride_dof_index] + stride)

def continuous_cyclic_phase_transition_without_xy(controllers: list[PenaltyController, PenaltyController],
                                                  states_mapping: list[BiMapping] = None) -> MX:
    states_mapping = MultinodePenaltyFunctions.Functions._prepare_states_mapping(controllers, states_mapping)
    end_states = states_mapping[0].to_second.map(controllers[0].states["all"].cx)
    start_states = states_mapping[0].to_first.map(controllers[1].states["all"].cx)
    constraint_violation = start_states - end_states
    return constraint_violation[2:]

def velocity_based_forward_displacement_phase_transition(controllers: list[PenaltyController, PenaltyController],
                                                         target_velocity = -0.05,
                                                         states_mapping: list[BiMapping] = None) -> MX:
    states_mapping = MultinodePenaltyFunctions.Functions._prepare_states_mapping(controllers, states_mapping)
    end_states = states_mapping[0].to_second.map(controllers[0].states["all"].cx)
    start_states = states_mapping[0].to_first.map(controllers[1].states["all"].cx)
    end_y = end_states[1]
    start_y = start_states[1]
    t = controllers[0].tf.cx + controllers[1].tf.cx
    return vertcat(start_states[2:] - end_states[2:], (end_y - start_y) / t - target_velocity)

def proportional_joint_constraint(pip_idx: int, dip_idx: int, coef: float):
    def make(model):
        q = MX.sym("q", model.nb_q)
        qdot = MX.sym("qdot", model.nb_qdot)
        phi = Function("phi", [q], [q[dip_idx] - coef*q[pip_idx]])
        phi_jac = Function("phi_jac", [q], [jacobian(q[dip_idx] - coef*q[pip_idx], q)])
        bias = Function("bias", [q, qdot], [MX.zeros(1,1)])
        return phi, phi_jac, bias
    return make

def marker_position(controller, marker_name: str, axis: Axis):
    q_u = controller.states["q_u"].cx
    q_v_init = getattr(controller.model, "q_v_init_guess", DM.zeros(controller.model.nb_dependent_joints, 1))
    q = controller.model.compute_q()(q_u, q_v_init)
    marker_index = controller.model.marker_index(marker_name)
    return controller.model.marker(marker_index)(q, controller.parameters.cx)[axis]

def marker_velocity(controller, marker_name: str, axis: Axis):
    q_u, qdot_u = controller.states["q_u"].cx, controller.states["qdot_u"].cx
    q_v_init = getattr(controller.model, "q_v_init_guess", DM.zeros(controller.model.nb_dependent_joints, 1))
    q = controller.model.compute_q()(q_u, q_v_init)
    qdot = controller.model.compute_qdot()(q, qdot_u)
    marker_index = controller.model.marker_index(marker_name)
    return controller.model.marker_velocity(marker_index)(q, qdot, controller.parameters.cx)[axis]


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
    objectives.add(marker_position, custom_type=ObjectiveFcn.Mayer, marker_name="base_contact_right_marker",
                   axis=Axis.Y, target=0, quadratic=True, weight=5, phase=0)
    objectives.add(marker_position, custom_type=ObjectiveFcn.Mayer, marker_name="base_contact_right_marker",
                   axis=Axis.Y, target=0, quadratic=True, weight=5, phase=1)
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
    constraints.add(  # Don't penetrate ground (shooting nodes only, see note below)
        marker_position,
        marker_name="middle_endeffector",
        axis=Axis.Z,
        node=Node.ALL,
        min_bound=0,
        max_bound=np.inf,
        phase=1,
    )
    constraints.add(  # Place middle finger on the ground for the contact establishment
        marker_position,
        marker_name="middle_endeffector",
        axis=Axis.Z,
        node=Node.END,
        min_bound=0,
        max_bound=0,
        phase=1,
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

def prepare_three_phase_holonomic_crawl(bio_model_path: str,
                                        no_contact_bio_model_path: str,
                                        n_threads: int = 2):
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
        ),
        HolonomicTendonBiorbdModel(
            bio_model_path,
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
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tendons", weight=0.001, phase=2)
    # objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="non_tendon_tau", weight=0.01, phase=0)
    # objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="non_tendon_tau", weight=0.01, phase=1)
    objectives.add(marker_position, custom_type=ObjectiveFcn.Mayer, marker_name="base_contact_right_marker",
                   axis=Axis.Y, target=0.04, quadratic=True, weight=5, phase=0)
    objectives.add(marker_position, custom_type=ObjectiveFcn.Mayer, marker_name="base_contact_right_marker",
                   axis=Axis.Y, target=0.04, quadratic=True, weight=5, phase=1)
    objectives.add(marker_position, custom_type=ObjectiveFcn.Mayer, marker_name="base_contact_right_marker",
                   axis=Axis.Y, target=0, quadratic=True, weight=5, phase=2)
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
    constraints.add(  # Don't penetrate ground (shooting nodes only, see note below)
        marker_position,
        marker_name="middle_endeffector",
        axis=Axis.Z,
        node=Node.ALL,
        min_bound=0,
        max_bound=np.inf,
        phase=1,
    )
    constraints.add(  # Place middle finger on the ground for the contact establishment
        marker_position,
        marker_name="middle_endeffector",
        axis=Axis.Z,
        node=Node.END,
        min_bound=0,
        max_bound=0,
        phase=1,
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
    constraints.add(  # base_contact_right
        ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES,
        min_bound=0,
        max_bound=np.inf,
        node=Node.ALL,
        contact_index=0,
        phase=2
    )
    constraints.add(  # thumb
        ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES,
        min_bound=0,
        max_bound=np.inf,
        node=Node.ALL,
        contact_index=1,
        phase=2
    )
    constraints.add(  # middle finger
        ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES,
        min_bound=0,
        max_bound=np.inf,
        node=Node.ALL,
        contact_index=4,
        phase=2
    )
    constraints.add(  # little finger
        ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES,
        min_bound=0,
        max_bound=np.inf,
        node=Node.ALL,
        contact_index=5,
        phase=2
    )
    constraints.add( # Minimum distance
        ConstraintFcn.BOUND_STATE,
        key="q_u",
        index=[1],
        min_bound=[-5],
        max_bound=[-0.04],
        phase=2,
        node=Node.ALL,
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
    x_bounds.add("q_u", bio_model[2].bounds_from_ranges("q", mapping=state_mapping), phase=2)
    x_bounds.add("qdot_u", bio_model[0].bounds_from_ranges("qdot", mapping=state_mapping), phase=0)
    x_bounds.add("qdot_u", bio_model[1].bounds_from_ranges("qdot", mapping=state_mapping), phase=1)
    x_bounds.add("qdot_u", bio_model[2].bounds_from_ranges("qdot", mapping=state_mapping), phase=2)
    x_bounds[0]["q_u"][:, 0] = q0_u
    x_bounds[0]["qdot_u"][:, 0] = 0
    x_bounds[0]["qdot_u"][:6, -1] = 0
    x_bounds[1]["qdot_u"][:6, -1] = 0
    x_bounds[2]["qdot_u"][:6, -1] = 0

    x_init = InitialGuessList()
    x_init.add("q_u", q0_u, phase=0)
    x_init.add("qdot_u", [0] * bio_model[0].nb_independent_joints, phase=0)

    u_bounds = BoundsList()
    u_bounds.add("tendons", min_bound=[0] * bio_model[0].nb_tendons, max_bound=[200] * bio_model[0].nb_tendons, phase=0)
    u_bounds.add("tendons", min_bound=[0] * bio_model[1].nb_tendons, max_bound=[200] * bio_model[0].nb_tendons, phase=1)
    u_bounds.add("tendons", min_bound=[0] * bio_model[2].nb_tendons, max_bound=[200] * bio_model[0].nb_tendons, phase=2)
    u_bounds.add("non_tendon_tau", min_bound=[-10] * n_non_tendon, max_bound=[10] * n_non_tendon, phase=0)
    u_bounds.add("non_tendon_tau", min_bound=[-10] * n_non_tendon, max_bound=[10] * n_non_tendon, phase=1)
    u_bounds.add("non_tendon_tau", min_bound=[-10] * n_non_tendon, max_bound=[10] * n_non_tendon, phase=2)

    u_init = InitialGuessList()
    u_init.add("tendons", [5] * bio_model[0].nb_tendons, phase=0)
    u_init.add("non_tendon_tau", [0] * n_non_tendon, phase=0)

    phase_transitions = PhaseTransitionList()
    phase_transitions.add(PhaseTransitionFcn.CONTINUOUS, phase_pre_idx=0)
    phase_transitions.add(PhaseTransitionFcn.CONTINUOUS, phase_pre_idx=1)

    dynamics = DynamicsOptionsList()
    dynamics.add(DynamicsOptions(ode_solver=OdeSolver.COLLOCATION(polynomial_degree=3)), phase=0)
    dynamics.add(DynamicsOptions(ode_solver=OdeSolver.COLLOCATION(polynomial_degree=3)), phase=1)
    dynamics.add(DynamicsOptions(ode_solver=OdeSolver.COLLOCATION(polynomial_degree=3)), phase=2)

    return bio_model, OptimalControlProgram(
        bio_model,
        n_shooting=[26, 26, 26],
        phase_time=(1, 0.5, 1),
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


def prepare_cyclic_holonomic_crawl(bio_model_path: str, no_contact_bio_model_path: str, n_threads=2):
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
    #objectives.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q_u", index=1, target=-0.05, weight=5, phase=0)
    #objectives.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q_u", index=1, target=-0.05, weight=5, phase=1)

    constraints = ConstraintList()
    for name in ("base_contact_right_marker", "thumb_endeffector", "little_endeffector"):
        constraints.add(marker_position, marker_name=name, axis=Axis.Z, node=Node.START, min_bound=0, max_bound=0, phase=0)
    for contact_index in [0,1,4,5]:
        constraints.add(
            ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES,
            min_bound=0,
            max_bound=np.inf,
            node=Node.ALL,
            contact_index=contact_index,
            phase=0
        )
    for contact_marker in ("base_contact_right_marker", "thumb_endeffector", "little_endeffector"):
        constraints.add(
            marker_velocity,
            min_bound=0,
            max_bound=0,
            node=Node.START,
            marker_name=contact_marker,
            axis=Axis.Z,
            phase=0
        )
    for axis in [Axis.X, Axis.Y, Axis.Z]:
        constraints.add(
            marker_velocity,
            marker_name="middle_endeffector",
            axis=axis,
            node=Node.START,
            min_bound=0,
            max_bound=0,
            phase=0,
        )
    for axis in [Axis.X, Axis.Y]:
        constraints.add(
            marker_velocity,
            marker_name="base_contact_right_marker",
            axis=axis,
            node=Node.END,
            min_bound=0,
            max_bound=0,
            phase=0,
        )
    for contact_index in [2,3,4]:
        constraints.add(
            ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES,
            min_bound=0,
            max_bound=np.inf,
            node=Node.ALL,
            contact_index=contact_index,
            phase=1
        )
    constraints.add(  # Don't penetrate ground
        marker_position,
        marker_name="middle_endeffector",
        axis=Axis.Z,
        node=Node.ALL,
        min_bound=0,
        max_bound=np.inf,
        phase=1,
    )
    constraints.add(  # Place middle finger on the ground for the contact establishment
        marker_position,
        marker_name="middle_endeffector",
        axis=Axis.Z,
        node=Node.END,
        min_bound=0,
        max_bound=0,
        phase=1,
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
    #x_bounds[0]["q_u"][2:, 0] = q0_u[2:]
    x_bounds[0]["qdot_u"][:6, 0] = 0
    x_bounds[0]["qdot_u"][:6, -1] = 0
    #x_bounds[1]["qdot_u"][:6, -1] = 0

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
    phase_transitions.add(PhaseTransitionFcn.CYCLIC, custom_function=displacement_aware_cyclic_phase_transition)

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

def prepare_velocity_based_holonomic_cyclic_crawl(bio_model_path: str, no_contact_bio_model_path: str, n_threads=2):
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
    #objectives.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q_u", index=1, target=-0.05, weight=5, phase=0)
    #objectives.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q_u", index=1, target=-0.05, weight=5, phase=1)

    constraints = ConstraintList()
    for phase in range(2):
        constraints.add(
            ConstraintFcn.TIME_CONSTRAINT,
            node=Node.END,
            min_bound=0.5,
            max_bound=2,
            phase=phase
        )
    for name in ("base_contact_right_marker", "thumb_endeffector", "little_endeffector"):
        constraints.add(marker_position, marker_name=name, axis=Axis.Z, node=Node.START, min_bound=0, max_bound=0, phase=0)
    for contact_index in [0,1,4,5]:
        constraints.add(
            ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES,
            min_bound=0,
            max_bound=np.inf,
            node=Node.ALL,
            contact_index=contact_index,
            phase=0
        )
    for contact_marker in ("base_contact_right_marker", "thumb_endeffector", "little_endeffector"):
        constraints.add(
            marker_velocity,
            min_bound=0,
            max_bound=0,
            node=Node.START,
            marker_name=contact_marker,
            axis=Axis.Z,
            phase=0
        )
    for axis in [Axis.X, Axis.Y, Axis.Z]:
        constraints.add(
            marker_velocity,
            marker_name="middle_endeffector",
            axis=axis,
            node=Node.START,
            min_bound=0,
            max_bound=0,
            phase=0,
        )
    for axis in [Axis.X, Axis.Y]:
        constraints.add(
            marker_velocity,
            marker_name="base_contact_right_marker",
            axis=axis,
            node=Node.END,
            min_bound=0,
            max_bound=0,
            phase=0,
        )
    for contact_index in [2,3,4]:
        constraints.add(
            ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES,
            min_bound=0,
            max_bound=np.inf,
            node=Node.ALL,
            contact_index=contact_index,
            phase=1
        )
    constraints.add(  # Don't penetrate ground
        marker_position,
        marker_name="middle_endeffector",
        axis=Axis.Z,
        node=Node.ALL,
        min_bound=0,
        max_bound=np.inf,
        phase=1,
    )
    constraints.add(  # Place middle finger on the ground for the contact establishment
        marker_position,
        marker_name="middle_endeffector",
        axis=Axis.Z,
        node=Node.END,
        min_bound=0,
        max_bound=0,
        phase=1,
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
        0.47, 0.91, 0.77259,
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
    #x_bounds[0]["q_u"][2:, 0] = q0_u[2:]
    x_bounds[0]["qdot_u"][:6, 0] = 0
    x_bounds[0]["qdot_u"][:6, -1] = 0
    #x_bounds[1]["qdot_u"][:6, -1] = 0

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
    phase_transitions.add(PhaseTransitionFcn.CYCLIC, custom_function=velocity_based_forward_displacement_phase_transition)

    dynamics = DynamicsOptionsList()
    dynamics.add(DynamicsOptions(ode_solver=OdeSolver.COLLOCATION(polynomial_degree=3)), phase=0)
    dynamics.add(DynamicsOptions(ode_solver=OdeSolver.COLLOCATION(polynomial_degree=3)), phase=1)

    return bio_model, OptimalControlProgram(
        bio_model,
        n_shooting=[26, 26],
        phase_time=(1, 0.5),
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

def prepare_inverse_velocity_based_holonomic_cyclic_crawl(
        bio_model_path: str,
        no_contact_bio_model_path: str,
        n_threads=2,
        free_end_time=False
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

    bio_model = (
        HolonomicTendonBiorbdModel(
            no_contact_bio_model_path,
            holonomic_constraints=holonomic_constraints,
            independent_joint_index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13],
            dependent_joint_index=[11, 14],
            contact_types=[ContactType.RIGID_EXPLICIT],
            torque_driven_dofs=["thumb_proxy_RotY"]
        ),
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
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tendons", weight=0.001, phase=2)

    constraints = ConstraintList()
    if free_end_time:
        for phase in range(3):
            constraints.add(
                ConstraintFcn.TIME_CONSTRAINT,
                node=Node.END,
                min_bound=0.5,
                max_bound=2,
                phase=phase
            )
    for name in ("base_contact_right_marker", "thumb_endeffector", "little_endeffector"):
        constraints.add(marker_position, marker_name=name, axis=Axis.Z, node=Node.START, min_bound=0, max_bound=0, phase=0)
        constraints.add(
            marker_velocity,
            min_bound=0,
            max_bound=0,
            node=Node.START,
            marker_name=name,
            axis=Axis.Z,
            phase=0
        )
    for phase in [0, 2]:
        for contact_index in [2, 3, 4]:
            constraints.add(
                ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES,
                min_bound=0,
                max_bound=np.inf,
                node=Node.ALL,
                contact_index=contact_index,
                phase=phase
            )
    for contact_index in [0, 1, 4, 5]:
        constraints.add(
            ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES,
            min_bound=0,
            max_bound=np.inf,
            node=Node.ALL,
            contact_index=contact_index,
            phase=1
        )
    for axis in [Axis.X, Axis.Y, Axis.Z]:
        constraints.add(
            marker_velocity,
            marker_name="middle_endeffector",
            axis=axis,
            node=Node.START,
            min_bound=0,
            max_bound=0,
            phase=1,
        )
    for axis in [Axis.X, Axis.Y]:
        constraints.add(
            marker_velocity,
            marker_name="base_contact_right_marker",
            axis=axis,
            node=Node.END,
            min_bound=0,
            max_bound=0,
            phase=1,
        )
    constraints.add(  # Don't penetrate ground
        marker_position,
        marker_name="middle_endeffector",
        axis=Axis.Z,
        node=Node.ALL,
        min_bound=0,
        max_bound=np.inf,
        phase=2,
    )
    constraints.add(  # Place middle finger on the ground for the contact establishment
        marker_position,
        marker_name="middle_endeffector",
        axis=Axis.Z,
        node=Node.END,
        min_bound=0,
        max_bound=0,
        phase=0,
    )

    q0 = [
        0.0, 0.0, 0.01402, -0.24539, 0.0132, 0,
        -0.3, 0, 0,
        0, 0, 0,
        0, 1.4, 1.1886
    ]
    q0_u = q0[:11] + q0[12:14]
    q0_v = [q0[11], q0[14]]
    bio_model[0].q_v_init_guess = DM(q0_v)
    n_non_tendon = len(bio_model[0].non_tendon_tau_indices)

    x_bounds = BoundsList()
    x_bounds.add("q_u", bio_model[0].bounds_from_ranges("q", mapping=state_mapping), phase=0)
    x_bounds.add("q_u", bio_model[1].bounds_from_ranges("q", mapping=state_mapping), phase=1)
    x_bounds.add("q_u", bio_model[2].bounds_from_ranges("q", mapping=state_mapping), phase=2)
    x_bounds.add("qdot_u", bio_model[0].bounds_from_ranges("qdot", mapping=state_mapping), phase=0)
    x_bounds.add("qdot_u", bio_model[1].bounds_from_ranges("qdot", mapping=state_mapping), phase=1)
    x_bounds.add("qdot_u", bio_model[2].bounds_from_ranges("qdot", mapping=state_mapping), phase=2)
    x_bounds[0]["q_u"][9, 0] = q0_u[9]
    x_bounds[0]["q_u"][11, 0] = q0_u[11]
    x_bounds[1]["qdot_u"][:6, 0] = 0
    x_bounds[1]["qdot_u"][:6, -1] = 0

    x_init = InitialGuessList()
    x_init.add("q_u", q0_u, phase=0)
    x_init.add("qdot_u", [0] * bio_model[0].nb_independent_joints, phase=0)

    u_bounds = BoundsList()
    u_bounds.add("tendons", min_bound=[0] * bio_model[0].nb_tendons, max_bound=[200] * bio_model[0].nb_tendons, phase=0)
    u_bounds.add("tendons", min_bound=[0] * bio_model[1].nb_tendons, max_bound=[200] * bio_model[1].nb_tendons, phase=1)
    u_bounds.add("tendons", min_bound=[0] * bio_model[2].nb_tendons, max_bound=[200] * bio_model[2].nb_tendons, phase=2)
    u_bounds.add("non_tendon_tau", min_bound=[-10] * n_non_tendon, max_bound=[10] * n_non_tendon, phase=0)
    u_bounds.add("non_tendon_tau", min_bound=[-10] * n_non_tendon, max_bound=[10] * n_non_tendon, phase=1)
    u_bounds.add("non_tendon_tau", min_bound=[-10] * n_non_tendon, max_bound=[10] * n_non_tendon, phase=2)

    u_init = InitialGuessList()
    u_init.add("tendons", [5] * bio_model[0].nb_tendons, phase=0)
    u_init.add("non_tendon_tau", [0] * n_non_tendon, phase=0)

    phase_transitions = PhaseTransitionList()
    phase_transitions.add(PhaseTransitionFcn.CONTINUOUS, phase_pre_idx=0)
    phase_transitions.add(PhaseTransitionFcn.CONTINUOUS, phase_pre_idx=1)
    phase_transitions.add(PhaseTransitionFcn.CYCLIC, custom_function=velocity_based_forward_displacement_phase_transition)

    dynamics = DynamicsOptionsList()
    dynamics.add(DynamicsOptions(ode_solver=OdeSolver.COLLOCATION(polynomial_degree=3)), phase=0)
    dynamics.add(DynamicsOptions(ode_solver=OdeSolver.COLLOCATION(polynomial_degree=3)), phase=1)
    dynamics.add(DynamicsOptions(ode_solver=OdeSolver.COLLOCATION(polynomial_degree=3)), phase=2)

    return bio_model, OptimalControlProgram(
        bio_model,
        n_shooting=[12, 23, 12],
        phase_time=(0.3, 0.75, 0.3),
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

def prepare_ramp_up_to_cyclic(bio_model_path: str,
                              no_contact_bio_model_path: str,
                              n_threads: int = 2):
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
            no_contact_bio_model_path,
            holonomic_constraints=holonomic_constraints,
            independent_joint_index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13],
            dependent_joint_index=[11, 14],
            contact_types=[ContactType.RIGID_EXPLICIT],
            torque_driven_dofs=["thumb_proxy_RotY"]
        ),
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
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tendons", weight=0.001, phase=2)

    constraints = ConstraintList()
    for phase in range(3):
        constraints.add(
            ConstraintFcn.TIME_CONSTRAINT,
            node=Node.END,
            min_bound=0.2,
            max_bound=2,
            phase=phase
        )
    for contact_index in (2,3,4):
        constraints.add( # Unilateral contacts
            ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES,
            min_bound=0,
            max_bound=np.inf,
            node=Node.ALL,
            contact_index=contact_index,
            phase=0
        )
        constraints.add(  # Unilateral contacts
            ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES,
            min_bound=0,
            max_bound=np.inf,
            node=Node.ALL,
            contact_index=contact_index,
            phase=2
        )
    constraints.add( # Drive middle finger to the ground
        marker_position,
        node=Node.END,
        marker_name="middle_endeffector",
        axis=Axis.Z,
        min_bound=0,
        max_bound=0,
        phase=0
    )
    for axis in [Axis.X, Axis.Y, Axis.Z]:
        constraints.add( # Manual impact handling (by having none) since PhaseTransitionFcn.IMPACT is not usable because of holonomic constraints (no full "q")
            marker_velocity,
            marker_name="middle_endeffector",
            axis=axis,
            node=Node.END,
            min_bound=0,
            max_bound=0,
            phase=0,
        )
    for contact_index in (0,1,4,5):
        constraints.add( # Unilateral contacts
            ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES,
            min_bound=0,
            max_bound=np.inf,
            node=Node.ALL,
            contact_index=contact_index,
            phase=1
        )
    for phase in (0,2):
        constraints.add(  # Don't penetrate ground
            marker_position,
            marker_name="middle_endeffector",
            axis=Axis.Z,
            node=Node.ALL,
            min_bound=0,
            max_bound=np.inf,
            phase=phase,
        )

    q0 = [
        0, 0.06, 0.014921, -0.263417, -0.058610, 0,
        -0.42, 0, 0,
        0, 0, 0,
        0, 1.4, 1.1886
    ]
    q0_u = q0[:11] + q0[12:14]
    q0_v = [q0[11], q0[14]]
    bio_model[0].q_v_init_guess = DM(q0_v)
    n_non_tendon = len(bio_model[0].non_tendon_tau_indices)

    x_bounds = BoundsList()
    x_bounds.add("q_u", bio_model[0].bounds_from_ranges("q", mapping=state_mapping), phase=0)
    x_bounds.add("q_u", bio_model[1].bounds_from_ranges("q", mapping=state_mapping), phase=1)
    x_bounds.add("q_u", bio_model[2].bounds_from_ranges("q", mapping=state_mapping), phase=2)
    x_bounds.add("qdot_u", bio_model[0].bounds_from_ranges("qdot", mapping=state_mapping), phase=0)
    x_bounds.add("qdot_u", bio_model[1].bounds_from_ranges("qdot", mapping=state_mapping), phase=1)
    x_bounds.add("qdot_u", bio_model[2].bounds_from_ranges("qdot", mapping=state_mapping), phase=2)
    x_bounds[0]["q_u"][2:, 0] = q0_u[2:] # keep x,y free
    x_bounds[0]["qdot_u"][:, 0] = 0

    q_f = [
        0, 0.03, 0.020547, -0.376057, 0.012275, -0.01,
        -0.33, 1.12, 0.65,
        0.726109, 0.28, 0.23772,
        0.42, 0.76, 0.64524
    ]
    q_f_u = q_f[:11] + q_f[12:14]
    x_bounds[2]["q_u"][:, -1] = q_f_u
    x_bounds[2]["qdot_u"][:, -1] = 0

    x_init = InitialGuessList()
    x_init.add("q_u", q0_u, phase=0)
    x_init.add("qdot_u", [0] * bio_model[0].nb_independent_joints, phase=0)

    u_bounds = BoundsList()
    u_bounds.add("tendons", min_bound=[0] * bio_model[0].nb_tendons, max_bound=[200] * bio_model[0].nb_tendons, phase=0)
    u_bounds.add("tendons", min_bound=[0] * bio_model[1].nb_tendons, max_bound=[200] * bio_model[0].nb_tendons, phase=1)
    u_bounds.add("tendons", min_bound=[0] * bio_model[2].nb_tendons, max_bound=[200] * bio_model[0].nb_tendons, phase=2)
    u_bounds.add("non_tendon_tau", min_bound=[-10] * n_non_tendon, max_bound=[10] * n_non_tendon, phase=0)
    u_bounds.add("non_tendon_tau", min_bound=[-10] * n_non_tendon, max_bound=[10] * n_non_tendon, phase=1)
    u_bounds.add("non_tendon_tau", min_bound=[-10] * n_non_tendon, max_bound=[10] * n_non_tendon, phase=2)

    u_init = InitialGuessList()
    u_init.add("tendons", [5] * bio_model[0].nb_tendons, phase=0)
    u_init.add("non_tendon_tau", [0] * n_non_tendon, phase=0)

    phase_transitions = PhaseTransitionList()
    phase_transitions.add(PhaseTransitionFcn.CONTINUOUS, phase_pre_idx=0) # IMPACT is not usable because of holonomic constraints (no full "q")
    phase_transitions.add(PhaseTransitionFcn.CONTINUOUS, phase_pre_idx=1)

    dynamics = DynamicsOptionsList()
    dynamics.add(DynamicsOptions(ode_solver=OdeSolver.COLLOCATION(polynomial_degree=3)), phase=0)
    dynamics.add(DynamicsOptions(ode_solver=OdeSolver.COLLOCATION(polynomial_degree=3)), phase=1)
    dynamics.add(DynamicsOptions(ode_solver=OdeSolver.COLLOCATION(polynomial_degree=3)), phase=2)

    return bio_model, OptimalControlProgram(
        bio_model,
        n_shooting=[26, 26, 26],
        phase_time=(0.5, 0.8, 0.5),
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

def prepare_single_phase_ramp_up(bio_model_path: str, n_threads=8):
    ...

def prepare_inchworm_ocp(
        middle_model_path: str,
        little_model_path: str,
        n_threads=2
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

    bio_model = (
        HolonomicTendonBiorbdModel(
            middle_model_path,
            holonomic_constraints=holonomic_constraints,
            independent_joint_index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13],
            dependent_joint_index=[11, 14],
            contact_types=[ContactType.RIGID_EXPLICIT],
            torque_driven_dofs=["thumb_proxy_RotY"]
        ),
        HolonomicTendonBiorbdModel(
            little_model_path,
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
    objectives.add(marker_position, custom_type=ObjectiveFcn.Mayer, marker_name="base_contact_right_marker",
                   axis=Axis.X, quadratic=True, weight=50, phase=0)
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tendons", weight=0.001, phase=1)
    objectives.add(marker_position, custom_type=ObjectiveFcn.Mayer, marker_name="base_contact_right_marker",
                   axis=Axis.X, quadratic=True, weight=50, phase=1)

    constraints = ConstraintList()
    for phase in range(2):
        constraints.add(
            ConstraintFcn.TIME_CONSTRAINT,
            node=Node.END,
            min_bound=0.5,
            max_bound=2,
            phase=phase
        )
    for name in ("base_contact_right_marker", "thumb_endeffector", "middle_endeffector"):
        constraints.add(marker_position, marker_name=name, axis=Axis.Z, node=Node.START, min_bound=0, max_bound=0, phase=0)
    for name in ("base_contact_right_marker", "thumb_endeffector", "little_endeffector"):
        constraints.add(marker_position, marker_name=name, axis=Axis.Z, node=Node.START, min_bound=0, max_bound=0, phase=1)
    for contact_index in [0,1,4]:
        constraints.add(
            ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES,
            min_bound=0,
            max_bound=np.inf,
            node=Node.ALL,
            contact_index=contact_index,
            phase=0
        )
        constraints.add(
            ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES,
            min_bound=0,
            max_bound=np.inf,
            node=Node.ALL,
            contact_index=contact_index,
            phase=1
        )
    for contact_marker in ("base_contact_right_marker", "thumb_endeffector"):
        constraints.add(
            marker_velocity,
            min_bound=0,
            max_bound=0,
            node=Node.START,
            marker_name=contact_marker,
            axis=Axis.Z,
            phase=0
        )
    for axis in [Axis.X, Axis.Y, Axis.Z]:
        constraints.add(
            marker_velocity,
            marker_name="middle_endeffector",
            axis=axis,
            node=Node.START,
            min_bound=0,
            max_bound=0,
            phase=0,
        )
    for axis in [Axis.X, Axis.Y, Axis.Z]:
        constraints.add(
            marker_velocity,
            marker_name="little_endeffector",
            axis=axis,
            node=Node.START,
            min_bound=0,
            max_bound=0,
            phase=1,
        )
    constraints.add(  # Don't penetrate ground
        marker_position,
        marker_name="middle_endeffector",
        axis=Axis.Z,
        node=Node.ALL,
        min_bound=0,
        max_bound=np.inf,
        phase=1,
    )
    constraints.add(  # Don't penetrate ground
        marker_position,
        marker_name="little_endeffector",
        axis=Axis.Z,
        node=Node.ALL,
        min_bound=0,
        max_bound=np.inf,
        phase=0,
    )

    q0 = [
        0.0, 0.0, 0.0271, -0.41, 0.0, 0.0,
        -0.43, 0.86, 1.01,
        0.47, 0.91, 0.77259,
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
    x_bounds[0]["qdot_u"][:6, 0] = 0
    x_bounds[0]["qdot_u"][:6, -1] = 0
    #x_bounds[1]["qdot_u"][:6, -1] = 0

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
    phase_transitions.add(PhaseTransitionFcn.CYCLIC, custom_function=velocity_based_forward_displacement_phase_transition)

    dynamics = DynamicsOptionsList()
    dynamics.add(DynamicsOptions(ode_solver=OdeSolver.COLLOCATION(polynomial_degree=3)), phase=0)
    dynamics.add(DynamicsOptions(ode_solver=OdeSolver.COLLOCATION(polynomial_degree=3)), phase=1)

    return bio_model, OptimalControlProgram(
        bio_model,
        n_shooting=[26, 26],
        phase_time=(1, 1),
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

def prepare_five_finger_inchworm_ocp(
        middle_model_path: str,
        ring_model_path: str,
        n_threads=2
):
    holonomic_constraints = HolonomicConstraintsList()
    holonomic_constraints.add(
        key="index_pip_dip",
        constraints_fcn=proportional_joint_constraint(pip_idx=10, dip_idx=11, coef=0.849),
    )
    holonomic_constraints.add(
        key="middle_pip_dip",
        constraints_fcn=proportional_joint_constraint(pip_idx=13, dip_idx=14, coef=0.849),
    )
    holonomic_constraints.add(
        key="ring_pip_dip",
        constraints_fcn=proportional_joint_constraint(pip_idx=16, dip_idx=17, coef=0.849),
    )
    holonomic_constraints.add(
        key="little_pip_dip",
        constraints_fcn=proportional_joint_constraint(pip_idx=19, dip_idx=20, coef=0.849),
    )

    bio_model = (
        HolonomicTendonBiorbdModel(
            middle_model_path,
            holonomic_constraints=holonomic_constraints,
            independent_joint_index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16, 18, 19],
            dependent_joint_index=[11, 14, 17, 20],
            contact_types=[ContactType.RIGID_EXPLICIT],
            torque_driven_dofs=["thumb_proxy_RotY"]
        ),
        HolonomicTendonBiorbdModel(
            ring_model_path,
            holonomic_constraints=holonomic_constraints,
            independent_joint_index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16, 18, 19],
            dependent_joint_index=[11, 14, 17, 20],
            contact_types=[ContactType.RIGID_EXPLICIT],
            torque_driven_dofs=["thumb_proxy_RotY"]
        )
    )

    state_mapping = BiMappingList()
    state_mapping.add("q",
                      to_second=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None, 11, 12, None, 13, 14, None, 15, 16, None],
                      to_first=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16, 18, 19])
    state_mapping.add("qdot",
                      to_second=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None, 11, 12, None, 13, 14, None, 15, 16, None],
                      to_first=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16, 18, 19])

    objectives = ObjectiveList()
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tendons", weight=0.001, phase=0)
    objectives.add(marker_position, custom_type=ObjectiveFcn.Mayer, marker_name="base_contact_right_marker",
                   axis=Axis.X, quadratic=True, weight=50, phase=0)
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tendons", weight=0.001, phase=1)
    objectives.add(marker_position, custom_type=ObjectiveFcn.Mayer, marker_name="base_contact_right_marker",
                   axis=Axis.X, quadratic=True, weight=50, phase=1)

    constraints = ConstraintList()
    for phase in range(2):
        constraints.add(
            ConstraintFcn.TIME_CONSTRAINT,
            node=Node.END,
            min_bound=0.5,
            max_bound=2,
            phase=phase
        )
    for name in ("base_contact_right_marker", "thumb_endeffector", "middle_endeffector"):
        constraints.add(marker_position, marker_name=name, axis=Axis.Z, node=Node.START, min_bound=0, max_bound=0, phase=0)
    for name in ("base_contact_right_marker", "thumb_endeffector", "little_endeffector"):
        constraints.add(marker_position, marker_name=name, axis=Axis.Z, node=Node.START, min_bound=0, max_bound=0, phase=1)
    for contact_index in [0,1,2,5,6]:
        constraints.add(
            ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES,
            min_bound=0,
            max_bound=np.inf,
            node=Node.ALL,
            contact_index=contact_index,
            phase=0
        )
        constraints.add(
            ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES,
            min_bound=0,
            max_bound=np.inf,
            node=Node.ALL,
            contact_index=contact_index,
            phase=1
        )
    for contact_marker in ("base_contact_right_marker", "thumb_endeffector", "index_endeffector", "little_endeffector"):
        constraints.add(
            marker_velocity,
            min_bound=0,
            max_bound=0,
            node=Node.START,
            marker_name=contact_marker,
            axis=Axis.Z,
            phase=0
        )
    for axis in [Axis.X, Axis.Y, Axis.Z]:
        constraints.add(
            marker_velocity,
            marker_name="middle_endeffector",
            axis=axis,
            node=Node.START,
            min_bound=0,
            max_bound=0,
            phase=0,
        )
    for axis in [Axis.X, Axis.Y, Axis.Z]:
        constraints.add(
            marker_velocity,
            marker_name="ring_endeffector",
            axis=axis,
            node=Node.START,
            min_bound=0,
            max_bound=0,
            phase=1,
        )
    constraints.add(  # Don't penetrate ground
        marker_position,
        marker_name="middle_endeffector",
        axis=Axis.Z,
        node=Node.ALL,
        min_bound=0,
        max_bound=np.inf,
        phase=1,
    )
    constraints.add(  # Don't penetrate ground
        marker_position,
        marker_name="ring_endeffector",
        axis=Axis.Z,
        node=Node.ALL,
        min_bound=0,
        max_bound=np.inf,
        phase=0,
    )

    q0 = [
        0.0, 0.0, 0.0271, -0.41, 0.0, 0.0,
        -0.43, 0.86, 1.01,
        0.69, 0.44, 0.37356,
        0.47, 0.91, 0.77259,
        0.47, 0.91, 0.77259,
        0.69, 0.44, 0.37356
    ]
    q0_u = q0[:11] + q0[12:14] + q0[15:17] + q0[18:20]
    q0_v = [q0[11], q0[14], q0[17], q0[20]]
    bio_model[0].q_v_init_guess = DM(q0_v)
    n_non_tendon = len(bio_model[0].non_tendon_tau_indices)

    x_bounds = BoundsList()
    x_bounds.add("q_u", bio_model[0].bounds_from_ranges("q", mapping=state_mapping), phase=0)
    x_bounds.add("q_u", bio_model[1].bounds_from_ranges("q", mapping=state_mapping), phase=1)
    x_bounds.add("qdot_u", bio_model[0].bounds_from_ranges("qdot", mapping=state_mapping), phase=0)
    x_bounds.add("qdot_u", bio_model[1].bounds_from_ranges("qdot", mapping=state_mapping), phase=1)
    #x_bounds[0]["qdot_u"][:6, 0] = 0
    #x_bounds[0]["qdot_u"][:6, -1] = 0
    #x_bounds[1]["qdot_u"][:6, -1] = 0

    x_init = InitialGuessList()
    x_init.add("q_u", q0_u, phase=0)
    x_init.add("qdot_u", [0] * bio_model[0].nb_independent_joints, phase=0)

    u_bounds = BoundsList()
    u_bounds.add("tendons", min_bound=[0] * bio_model[0].nb_tendons, max_bound=[200] * bio_model[0].nb_tendons, phase=0)
    u_bounds.add("tendons", min_bound=[0] * bio_model[1].nb_tendons, max_bound=[200] * bio_model[0].nb_tendons, phase=1)
    u_bounds.add("non_tendon_tau", min_bound=[-10] * n_non_tendon, max_bound=[10] * n_non_tendon, phase=0)
    u_bounds.add("non_tendon_tau", min_bound=[-10] * n_non_tendon, max_bound=[10] * n_non_tendon, phase=1)

    u_init = InitialGuessList()
    u_init.add("tendons", [2] * bio_model[0].nb_tendons, phase=0)
    u_init.add("non_tendon_tau", [0] * n_non_tendon, phase=0)

    phase_transitions = PhaseTransitionList()
    phase_transitions.add(PhaseTransitionFcn.CONTINUOUS, phase_pre_idx=0)
    phase_transitions.add(PhaseTransitionFcn.CYCLIC, custom_function=velocity_based_forward_displacement_phase_transition)

    dynamics = DynamicsOptionsList()
    dynamics.add(DynamicsOptions(ode_solver=OdeSolver.COLLOCATION(polynomial_degree=3)), phase=0)
    dynamics.add(DynamicsOptions(ode_solver=OdeSolver.COLLOCATION(polynomial_degree=3)), phase=1)

    return bio_model, OptimalControlProgram(
        bio_model,
        n_shooting=[26, 26],
        phase_time=(1, 1),
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
    states = sol.decision_states(to_merge=[SolutionMerge.NODES, SolutionMerge.PHASES])
    q = bio_model[0].compute_q_from_u_iterative(states["q_u"], q_v_init=np.array(bio_model[0].q_v_init_guess))
    viz = bioviz.Viz(model_path)
    viz.load_movement(q)
    viz.exec()
    sol.graphs(automatically_organize=False)

def three_phase_main():
    model_path = ExampleUtils.folder + "/models/holonomic_three_finger_crawl.bioMod"
    model_path_no_contact = ExampleUtils.folder + "/models/holonomic_three_finger_crawl_no_contact.bioMod"
    bio_model, ocp = prepare_three_phase_holonomic_crawl(
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
    solver.set_maximum_iterations(5000)
    sol = ocp.solve(solver)
    sol.print_cost()
    states = sol.decision_states(to_merge=[SolutionMerge.NODES, SolutionMerge.PHASES])
    q = bio_model[0].compute_q_from_u_iterative(states["q_u"], q_v_init=np.array(bio_model[0].q_v_init_guess))
    viz = bioviz.Viz(model_path)
    viz.load_movement(q)
    viz.exec()
    sol.graphs(automatically_organize=False, show_gcom_plot=True, show_interactive_stability_plot=True)

def cyclic_main():
    model_path = ExampleUtils.folder + "/models/holonomic_three_finger_crawl.bioMod"
    model_path_no_contact = ExampleUtils.folder + "/models/holonomic_three_finger_crawl_no_contact.bioMod"
    bio_model, ocp = prepare_cyclic_holonomic_crawl(
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
    solver.set_maximum_iterations(12000)
    sol = ocp.solve(solver)
    sol.print_cost()
    states = sol.decision_states(to_merge=[SolutionMerge.NODES, SolutionMerge.PHASES])
    q = bio_model[0].compute_q_from_u_iterative(states["q_u"], q_v_init=np.array(bio_model[0].q_v_init_guess))
    viz = bioviz.Viz(model_path)
    viz.load_movement(q)
    viz.exec()
    sol.graphs(automatically_organize=False)

def velocity_based_cyclic_main():
    model_path = ExampleUtils.folder + "/models/holonomic_three_finger_crawl.bioMod"
    model_path_no_contact = ExampleUtils.folder + "/models/holonomic_three_finger_crawl_no_contact.bioMod"
    bio_model, ocp = prepare_velocity_based_holonomic_cyclic_crawl(
        model_path,
        model_path_no_contact,
        n_threads=14,
    )
    ocp.add_plot_penalty(CostType.CONSTRAINTS)
    solver = Solver.IPOPT()
    solver.set_maximum_iterations(2000)
    sol = ocp.solve(solver)
    sol.print_cost()
    states = sol.decision_states(to_merge=[SolutionMerge.NODES, SolutionMerge.PHASES])
    q = bio_model[0].compute_q_from_u_iterative(states["q_u"], q_v_init=np.array(bio_model[0].q_v_init_guess))
    viz = bioviz.Viz(model_path)
    viz.load_movement(q)
    viz.exec()
    sol.graphs(automatically_organize=False)

def inverse_velocity_based_cyclic_main():
    model_path = ExampleUtils.folder + "/models/holonomic_three_finger_crawl.bioMod"
    model_path_no_contact = ExampleUtils.folder + "/models/holonomic_three_finger_crawl_no_contact.bioMod"
    bio_model, ocp = prepare_inverse_velocity_based_holonomic_cyclic_crawl(
        model_path,
        model_path_no_contact,
        n_threads=4,
    )
    ocp.add_plot_penalty(CostType.CONSTRAINTS)
    solver = Solver.IPOPT()
    solver.set_maximum_iterations(1_000_000)
    ocp.set_ocp_solver(solver)
    ocp.ocp_solver.options_common["iteration_callback"] = IterationsControllerCallback(ocp, budget=2000, default_extension=500)
    sol = ocp.solve(solver)
    sol.print_cost()
    states = sol.decision_states(to_merge=[SolutionMerge.NODES, SolutionMerge.PHASES])
    q = bio_model[0].compute_q_from_u_iterative(states["q_u"], q_v_init=np.array(bio_model[0].q_v_init_guess))
    viz = bioviz.Viz(model_path)
    viz.load_movement(q)
    viz.exec()
    sol.graphs(automatically_organize=False)

def ramp_up_main():
    model_path = ExampleUtils.folder + "/models/holonomic_three_finger_crawl.bioMod"
    model_path_no_contact = ExampleUtils.folder + "/models/holonomic_three_finger_crawl_no_contact.bioMod"
    bio_model, ocp = prepare_ramp_up_to_cyclic(
        model_path,
        model_path_no_contact,
        n_threads=4,
    )
    ocp.add_plot_penalty(CostType.CONSTRAINTS)
    solver = Solver.IPOPT()
    solver.set_maximum_iterations(1_000_000)
    ocp.set_ocp_solver(solver)
    ocp.ocp_solver.options_common["iteration_callback"] = IterationsControllerCallback(ocp, budget=1000, default_extension=500)
    sol = ocp.solve(solver)
    sol.print_cost()
    states = sol.decision_states(to_merge=[SolutionMerge.NODES, SolutionMerge.PHASES])
    q = bio_model[0].compute_q_from_u_iterative(states["q_u"], q_v_init=np.array(bio_model[0].q_v_init_guess))
    viz = bioviz.Viz(model_path)
    viz.load_movement(q)
    viz.exec()
    sol.graphs(automatically_organize=False, show_gcom_plot=True, show_interactive_stability_plot=True)

def inchworm_main():
    middle_model_path = "three_finger_inchworm_middle.bioMod"
    little_model_path = "three_finger_inchworm_little.bioMod"
    bio_model, ocp = prepare_inchworm_ocp(
        middle_model_path,
        little_model_path,
        n_threads=8,
    )
    ocp.add_plot_penalty(CostType.CONSTRAINTS)
    solver = Solver.IPOPT()
    solver.set_maximum_iterations(1_000_000)
    ocp.set_ocp_solver(solver)
    ocp.ocp_solver.options_common["iteration_callback"] = IterationsControllerCallback(ocp, budget=1000, default_extension=500)
    sol = ocp.solve(solver)
    sol.print_cost()
    states = sol.decision_states(to_merge=[SolutionMerge.NODES, SolutionMerge.PHASES])
    q = bio_model[0].compute_q_from_u_iterative(states["q_u"], q_v_init=np.array(bio_model[0].q_v_init_guess))
    viz = bioviz.Viz(middle_model_path)
    viz.load_movement(q)
    viz.exec()
    sol.graphs(automatically_organize=False)

def five_fingered_inchworm_main():
    middle_model_path = str(Path(__file__).with_name("five_finger_inchworm_middle.bioMod"))
    little_model_path = str(Path(__file__).with_name("five_finger_inchworm_ring.bioMod"))
    bio_model, ocp = prepare_five_finger_inchworm_ocp(
        middle_model_path,
        little_model_path,
        n_threads=8,
    )
    ocp.add_plot_penalty(CostType.CONSTRAINTS)
    solver = Solver.IPOPT()
    solver.set_maximum_iterations(1_000_000)
    ocp.set_ocp_solver(solver)
    ocp.ocp_solver.options_common["iteration_callback"] = IterationsControllerCallback(ocp, budget=1000, default_extension=500)
    sol = ocp.solve(solver)
    sol.print_cost()
    states = sol.decision_states(to_merge=[SolutionMerge.NODES, SolutionMerge.PHASES])
    q = bio_model[0].compute_q_from_u_iterative(states["q_u"], q_v_init=np.array(bio_model[0].q_v_init_guess))
    viz = bioviz.Viz(middle_model_path)
    viz.load_movement(q)
    viz.exec()
    sol.graphs(automatically_organize=False)

if __name__ == "__main__":
    #single_phase_main()
    #two_phase_main()
    #three_phase_main()
    #cyclic_main()
    #velocity_based_cyclic_main()
    #inverse_velocity_based_cyclic_main()
    #ramp_up_main()
    inchworm_main()
    #five_fingered_inchworm_main()
