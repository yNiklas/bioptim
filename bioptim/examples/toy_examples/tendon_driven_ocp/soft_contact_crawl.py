from functools import partial

import bioviz
from casadi import DM

from bioptim import CostType, Solver, InterpolationType, ControlType, PhaseDynamics, HolonomicConstraintsList
import numpy as np
from pathlib import Path

from bioptim import TendonBiorbdModel, OptimalControlProgram, ObjectiveFcn, BoundsList, \
    Solver, ConstraintList, ConstraintFcn, Node, ContactType, ObjectiveList, \
    CostType, Axis, TorqueBiorbdModel, BiMappingList, Solution, \
    SolutionMerge, InitialGuessList, VariableScalingList, DynamicsOptions, OdeSolver, PhaseTransitionList, \
    PhaseTransitionFcn, DynamicsOptionsList
from bioptim.examples.toy_examples.tendon_driven_ocp.holonomic_tendon import marker_position, \
    velocity_based_forward_displacement_phase_transition
from bioptim.examples.toy_examples.torque_driven_ocp.holonomic import proportional_joint_constraint
from bioptim.examples.utils import IterationsControllerCallback, ExampleUtils
from bioptim.models.biorbd.model_dynamics import HolonomicTendonBiorbdModel


def prepare_touchdown_ocp(biorbd_model_path: str):
    bio_model = TendonBiorbdModel(
        biorbd_model_path,
        contact_types=[ContactType.SOFT_EXPLICIT],
        torque_driven_dofs=["thumb_proxy_RotY", "middle_distal_RotX", "little_distal_RotX"],
    )
    objectives = ObjectiveList()
    #objectives.add(ObjectiveFcn.Mayer.MINIMIZE_TIME,)
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tendons")

    constraints = ConstraintList()
    constraints.add(
        ConstraintFcn.TRACK_MARKERS,
        marker_index="middle_endeffector",
        node=Node.END,
        axes=[Axis.Z]
    )

    x_bounds = BoundsList()
    x_bounds.add("q", bio_model.bounds_from_ranges("q"))
    x_bounds.add("qdot", bio_model.bounds_from_ranges("qdot"))
    x_bounds["q"][:, 0] = 0
    x_bounds["q"][2, 0] = 0.025
    x_bounds["q"][3, 0] = -0.41
    x_bounds["q"][6, 0] = -0.43
    x_bounds["q"][7, 0] = 0.86
    x_bounds["q"][8, 0] = 1.01
    x_bounds["q"][11, 0] = 0.52
    x_bounds["q"][12, 0] = 0.63
    x_bounds["qdot"][:, 0] = 1e-10
    x_bounds["qdot"][:, -1] = 0

    x_init = InitialGuessList()
    x_init.add("q", [0, 0, 0.1, -0.41, 0, 0, -0.43, 0.86, 1.01, 0, 0, 0.52, 0.63, 0, 0])
    x_init.add("qdot", [1e-10] * bio_model.nb_qdot)

    u_bounds = BoundsList()
    nb_control = 3#bio_model.nb_tau-6
    u_bounds.add("tendons", min_bound=[0]*nb_control, max_bound=[500]*nb_control)
    nb_non_tau = len(bio_model.non_tendon_tau_indices)
    u_bounds.add("non_tendon_tau", min_bound=[-100]*nb_non_tau, max_bound=[100]*nb_non_tau)

    map = BiMappingList()
    map.add("tau", to_second=[None, None, None, None, None, None, 0, 1,2,3,4,5,6,7,8], to_first=[6,7, 8,9,10,11,12,13,14])

    return OptimalControlProgram(
        bio_model,
        n_shooting=50,
        phase_time=1,
        objective_functions=objectives,
        constraints=constraints,
        #variable_mappings=map,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        n_threads=12
    )



def prepare_torque_crawl_ocp(bio_model_path: str):
    bio_model = TorqueBiorbdModel(
        bio_model_path,
        contact_types=[ContactType.SOFT_IMPLICIT],
    )
    n_shooting = 50
    objectives = ObjectiveList()
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=0.001)
    objectives.add(ObjectiveFcn.Mayer.TRACK_MARKERS, marker_index="base_contact_right_marker", axes=[Axis.Y], weight=5, target=0.065)
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", index=[5], weight=15)

    # Starting posture (fingers pre-flexed and in contact with the ground).
    q0 = [
        0.0, 0.0, 0.0271, -0.41, 0.0, 0.0,
        -0.7024763436, 0.5861794253, 0.9108124402,
        0.4682393892, 0.8335137780, 0.5205543439,
        0.5872591904, 0.5522876575, 0.6356821555,
    ]

    n_non_tendon = len(bio_model.non_tendon_tau_indices)

    x_bounds = BoundsList()
    x_bounds.add("q", bio_model.bounds_from_ranges("q"))
    x_bounds.add("qdot", bio_model.bounds_from_ranges("qdot"))
    x_bounds["q"][:, 0] = q0
    x_bounds["qdot"][:, 0] = 1e-10
    x_bounds["qdot"][:, -1] = 0

    x_init = InitialGuessList()
    x_init.add("q", q0)
    x_init.add("qdot", [1e-10] * bio_model.nb_qdot)

    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=[-200] * (bio_model.nb_tau-6), max_bound=[200] * (bio_model.nb_tau-6))

    a_bounds = BoundsList()
    a_bounds.add(
        "soft_contact_forces",
        min_bound=[-200, -200, -200, -200, -200, 0] * bio_model.nb_soft_contacts,
        max_bound=[200, 200, 200, 200, 200, 200] * bio_model.nb_soft_contacts,
        interpolation=InterpolationType.CONSTANT
    )

    u_init = InitialGuessList()
    u_init.add("tau", [0] * (bio_model.nb_tau-6))

    a_init = InitialGuessList()
    a_init.add(
        "soft_contact_forces",
        np.asarray(bio_model.soft_contact_forces()(q0, np.full(bio_model.nb_qdot, 1e-10), [])).reshape((-1,)),
    )

    mapping = BiMappingList()
    mapping.add("tau", to_second=[None]*6 + [0,1,2,3,4,5,6,7,8], to_first=[6,7,8,9,10,11,12,13,14])

    return OptimalControlProgram(
        bio_model,
        n_shooting=n_shooting,
        phase_time=0.5,
        objective_functions=objectives,
        dynamics=DynamicsOptions(
            phase_dynamics=PhaseDynamics.ONE_PER_NODE,
            ode_solver=OdeSolver.COLLOCATION(polynomial_degree=3),
        ),
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        a_bounds=a_bounds,
        x_init=x_init,
        variable_mappings=mapping,
        u_init=u_init,
        a_init=a_init,
        control_type=ControlType.CONSTANT,
        n_threads=1,
    )


def prepare_crawl_ocp(bio_model_path: str):
    bio_model = TendonBiorbdModel(
        bio_model_path,
        contact_types=[ContactType.SOFT_IMPLICIT],
        torque_driven_dofs=["thumb_proxy_RotY", "middle_distal_RotX", "little_distal_RotX"],
    )
    n_shooting = 50
    objectives = ObjectiveList()
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tendons", weight=0.001)
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="non_tendon_tau", weight=0.01)
    objectives.add(ObjectiveFcn.Mayer.TRACK_MARKERS, marker_index="base_contact_right_marker", axes=[Axis.Y], weight=5)
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", index=[5], weight=20)

    # Starting posture (fingers pre-flexed and in contact with the ground).
    q0 = [
        0.0, 0.0, 0.0271, -0.41, 0.0, 0.0,
        -0.7024763436, 0.5861794253, 0.9108124402,
        0.4682393892, 0.8335137780, 0.5205543439,
        0.5872591904, 0.5522876575, 0.6356821555,
    ]

    n_non_tendon = len(bio_model.non_tendon_tau_indices)

    x_bounds = BoundsList()
    x_bounds.add("q", bio_model.bounds_from_ranges("q"))
    x_bounds.add("qdot", bio_model.bounds_from_ranges("qdot"))
    x_bounds["q"][:, 0] = q0
    x_bounds["qdot"][:, 0] = 1e-10
    x_bounds["qdot"][:, -1] = 0

    x_init = InitialGuessList()
    x_init.add("q", q0)
    x_init.add("qdot", [1e-10] * bio_model.nb_qdot)

    u_bounds = BoundsList()
    u_bounds.add("tendons", min_bound=[0] * bio_model.nb_tendons, max_bound=[200] * bio_model.nb_tendons)
    u_bounds.add("non_tendon_tau", min_bound=[-10] * n_non_tendon, max_bound=[10] * n_non_tendon)

    a_bounds = BoundsList()
    a_bounds.add(
        "soft_contact_forces",
        min_bound=[-200, -200, -200, -200, -200, 0] * bio_model.nb_soft_contacts,
        max_bound=[200, 200, 200, 200, 200, 200] * bio_model.nb_soft_contacts,
        interpolation=InterpolationType.CONSTANT
    )

    u_init = InitialGuessList()
    u_init.add("tendons", [5] * bio_model.nb_tendons)
    u_init.add("non_tendon_tau", [0] * n_non_tendon)

    a_init = InitialGuessList()
    a_init.add(
        "soft_contact_forces",
        np.asarray(bio_model.soft_contact_forces()(q0, np.full(bio_model.nb_qdot, 1e-10), [])).reshape((-1,)),
    )

    u_scaling = VariableScalingList()
    u_scaling.add("tendons", scaling=[100.0] * bio_model.nb_tendons)
    u_scaling.add("non_tendon_tau", scaling=[10.0] * n_non_tendon)

    return OptimalControlProgram(
        bio_model,
        n_shooting=n_shooting,
        phase_time=0.5,
        objective_functions=objectives,
        dynamics=DynamicsOptions(
            phase_dynamics=PhaseDynamics.ONE_PER_NODE,
            ode_solver=OdeSolver.COLLOCATION(polynomial_degree=3),
        ),
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        a_bounds=a_bounds,
        x_init=x_init,
        u_init=u_init,
        a_init=a_init,
        u_scaling=u_scaling,
        control_type=ControlType.CONSTANT,
        n_threads=1,
    )

def prepare_holonomic_soft_crawl_ocp(bio_model_path: str, n_threads=8):
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
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tendons", weight=0.001)
    objectives.add(marker_position, custom_type=ObjectiveFcn.Mayer, marker_name="base_contact_right_marker",
                   axis=Axis.Y, target=0, quadratic=True, weight=40)

    constraints = ConstraintList()
    constraints.add(
        marker_position,
        node=Node.END,
        marker_name="base_contact_right_marker",
        min_bound=-5,
        max_bound=0.07,
        axis=Axis.Y
    )

    # Starting posture (fingers pre-flexed and in contact with the ground).
    q0 = [
        0.0, 0.0, 0.0235, -0.41, 0.0, 0.0,
        -0.43, 0.86, 1.01,
        0.47, 0.91, 0.77259,
        0.69, 0.44, 0.37356
    ]
    #q0 = [
    #    0, 0, 0.014367, -0.274935, 0, 0,
    #    -0.210919, 0, 1.188015,
    #    0, 1.072009, 0.910136,
    #    0, 1.369, 1.162281
    #]
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

def prepare_two_phase_holonomic_soft_crawl_ocp(bio_model_path: str, n_threads=8):
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
            contact_types=[],
            torque_driven_dofs=["thumb_proxy_RotY"]
        ),
        HolonomicTendonBiorbdModel(
            bio_model_path,
            holonomic_constraints=holonomic_constraints,
            independent_joint_index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13],
            dependent_joint_index=[11, 14],
            contact_types=[],
            torque_driven_dofs=["thumb_proxy_RotY"]
        ),
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
    objectives.add(marker_position, custom_type=ObjectiveFcn.Mayer, marker_name="base_contact_right_marker",
                   axis=Axis.Y, target=0, quadratic=True, weight=40, phase=1)

    constraints = ConstraintList()
    #constraints.add(
    #    marker_position,
    #    node=Node.ALL,
    #    marker_name="middle_endeffector",
    #    min_bound=0,
    #    max_bound=0,
    #    axis=Axis.Z,
    #    phase=1
    #)
    constraints.add(
        marker_position,
        node=Node.END,
        marker_name="base_contact_right_marker",
        min_bound=-5,
        max_bound=0.07,
        axis=Axis.Y,
        phase=1
    )

    # Starting posture (fingers pre-flexed and in contact with the ground).
    q0 = [
        0.0, 0.0, 0.01289, -0.41, 0.0, 0.0,
        -0.43, 0.86, 1.01,
        0, 0, 0,
        0.69, 0.44, 0.37356
    ]
    q0_u = q0[:11] + q0[12:14]
    q0_v = [q0[11], q0[14]]
    bio_model[0].q_v_init_guess = DM(q0_v)

    x_bounds = BoundsList()
    x_bounds.add("q_u", bio_model[0].bounds_from_ranges("q", mapping=state_mapping), phase=0)
    x_bounds.add("q_u", bio_model[0].bounds_from_ranges("q", mapping=state_mapping), phase=1)
    x_bounds.add("qdot_u", bio_model[0].bounds_from_ranges("qdot", mapping=state_mapping), phase=0)
    x_bounds.add("qdot_u", bio_model[0].bounds_from_ranges("qdot", mapping=state_mapping), phase=1)
    x_bounds[0]["q_u"][:, 0] = q0_u
    x_bounds[0]["qdot_u"][:, 0] = 1e-10
    x_bounds[1]["qdot_u"][:, -1] = 0

    x_init = InitialGuessList()
    x_init.add("q_u", q0_u)
    x_init.add("qdot_u", [1e-10] * bio_model[0].nb_independent_joints)

    u_bounds = BoundsList()
    u_bounds.add("tendons", min_bound=[0] * bio_model[0].nb_tendons, max_bound=[200] * bio_model[0].nb_tendons, phase=0)
    u_bounds.add("tendons", min_bound=[0] * bio_model[0].nb_tendons, max_bound=[200] * bio_model[0].nb_tendons, phase=1)
    u_bounds.add("non_tendon_tau", min_bound=[-20], max_bound=[20], phase=0)
    u_bounds.add("non_tendon_tau", min_bound=[-20], max_bound=[20], phase=1)

    phase_transitions = PhaseTransitionList()
    phase_transitions.add(PhaseTransitionFcn.CONTINUOUS, phase_pre_idx=0)

    dynamics_options = DynamicsOptionsList()
    dynamics_options.add(DynamicsOptions(ode_solver=OdeSolver.COLLOCATION(polynomial_degree=3)), phase=0)
    dynamics_options.add(DynamicsOptions(ode_solver=OdeSolver.COLLOCATION(polynomial_degree=3)), phase=1)

    return bio_model, OptimalControlProgram(
        bio_model,
        n_shooting=[20, 20],
        phase_time=[0.5, 0.5],
        objective_functions=objectives,
        constraints=constraints,
        dynamics=dynamics_options,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        variable_mappings=state_mapping,
        n_threads=n_threads
    )

def prepare_cyclic_holonomic_soft_crawl_ocp(bio_model_path: str):
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
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tendons", weight=0.001)

    constraints = ConstraintList()
    constraints.add(
        ConstraintFcn.TIME_CONSTRAINT,
        node=Node.END,
        min_bound=0.5,
        max_bound=1.5
    )

    phase_transitions = PhaseTransitionList()
    phase_transitions.add(
        PhaseTransitionFcn.CYCLIC,
        custom_function=partial(velocity_based_forward_displacement_phase_transition, target_velocity=-0.07),
        phase_pre_idx=0,
    )

    # Starting posture (fingers pre-flexed and in contact with the ground).
    q0 = [
        0.0, 0.0, 0.01581, -0.30541,-0.03601,0,#-0.41, 0.0, 0.0,
        -0.38,0,0,#-0.43, 0.86, 1.01,
        0.19, 0.81, 0.81*0.849, #0.47, 0.91, 0.77259,
        0, 1.369, 1.1852#0.69, 0.44, 0.37356
    ]
    q0_u = q0[:11] + q0[12:14]
    q0_v = [q0[11], q0[14]]
    bio_model.q_v_init_guess = DM(q0_v)

    x_bounds = BoundsList()
    x_bounds.add("q_u", bio_model.bounds_from_ranges("q", mapping=state_mapping))
    x_bounds.add("qdot_u", bio_model.bounds_from_ranges("qdot", mapping=state_mapping))
    #x_bounds["q_u"][:, 0] = q0_u
    #x_bounds["qdot_u"][:, 0] = 1e-10
    #x_bounds["qdot_u"][:, -1] = 0

    x_init = InitialGuessList()
    x_init.add("q_u", q0_u)
    x_init.add("qdot_u", [1e-10] * bio_model.nb_independent_joints)

    u_bounds = BoundsList()
    u_bounds.add("tendons", min_bound=[0] * bio_model.nb_tendons, max_bound=[200] * bio_model.nb_tendons)
    u_bounds.add("non_tendon_tau", min_bound=[-20], max_bound=[20])

    return bio_model, OptimalControlProgram(
        bio_model,
        n_shooting=40,
        phase_time=1,
        objective_functions=objectives,
        phase_transitions=phase_transitions,
        dynamics=DynamicsOptions(
            ode_solver=OdeSolver.COLLOCATION(polynomial_degree=3),
            #phase_dynamics=PhaseDynamics.ONE_PER_NODE,
        ),
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        variable_mappings=state_mapping,
        n_threads=10
    )


def main():
    #ocp = prepare_touchdown_ocp(str(Path(__file__).with_name("soft_contact_crawl.bioMod")))
    #ocp = prepare_crawl_ocp(str(Path(__file__).with_name("soft_contact_crawl.bioMod")))
    ocp = prepare_torque_crawl_ocp(str(Path(__file__).with_name("soft_contact_crawl.bioMod")))
    ocp.add_plot_penalty(CostType.CONSTRAINTS)
    solver = Solver.IPOPT()
    solver.set_maximum_iterations(2000)
    sol = ocp.solve(solver)
    sol.print_cost()
    sol.animate(100)
    sol.graphs()

def holonomic_main():
    model_path = str(Path(__file__).with_name("holonomic_soft_contact_three_finger.bioMod"))
    bio_model, ocp = prepare_holonomic_soft_crawl_ocp(
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
    ExampleUtils.save_control_data(ocp, sol, "solutions/soft_contact_single_phase.npz")

def holonomic_two_phase_main():
    model_path = str(Path(__file__).with_name("holonomic_soft_contact_three_finger_2.bioMod"))
    bio_model, ocp = prepare_two_phase_holonomic_soft_crawl_ocp(
        model_path,
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
    q = bio_model[0].compute_q_from_u_iterative(states["q_u"])
    viz = bioviz.Viz(model_path)
    viz.load_movement(q)
    viz.exec()
    sol.graphs(automatically_organize=False)

def cyclic_main():
    model_path = str(Path(__file__).with_name("holonomic_soft_contact_three_finger_2.bioMod"))
    bio_model, ocp = prepare_cyclic_holonomic_soft_crawl_ocp(model_path)
    ocp.add_plot_penalty(CostType.CONSTRAINTS)
    solver = Solver.IPOPT()
    solver.set_maximum_iterations(1_000_000)
    ocp.set_ocp_solver(solver)
    ocp.ocp_solver.options_common["iteration_callback"] = IterationsControllerCallback(ocp, budget=2000, default_extension=500)
    sol = ocp.solve(solver)
    sol.print_cost()
    states = sol.decision_states(to_merge=[SolutionMerge.NODES, SolutionMerge.PHASES])
    q = bio_model.compute_q_from_u_iterative(states["q_u"])
    viz = bioviz.Viz(model_path)
    viz.load_movement(q)
    viz.exec()
    sol.graphs(automatically_organize=False)

if __name__ == "__main__":
    #main()
    holonomic_main()
    #holonomic_two_phase_main()
    #cyclic_main()
