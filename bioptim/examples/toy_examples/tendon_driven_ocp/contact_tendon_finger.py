import faulthandler

import numpy as np
from biorbd import contact_index
from casadi import MX

faulthandler.enable()

from bioptim import TendonBiorbdModel, OptimalControlProgram, Objective, ObjectiveFcn, BoundsList, InitialGuessList, \
    Solver, ConstraintList, ConstraintFcn, Node, ContactType, ObjectiveList, PhaseTransitionList, PhaseTransitionFcn, \
    CostType, TorqueBiorbdModel, PenaltyController, BiMappingList, DynamicsOptions
from bioptim.examples.utils import ExampleUtils

def prepare_single_phase_ocp(biorbd_model_path: str,) -> OptimalControlProgram:
    bio_model = TendonBiorbdModel(biorbd_model_path)

    objective_functions = ObjectiveList()
    objective_functions.add(Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tendons", phase=0))
    #objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME)

    constraints = ConstraintList()
    #constraints.add(
    #    ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES,
    #    min_bound=0,
    #    max_bound=np.inf,
    #    node=Node.ALL,
    #    contact_index=0
    #)
    #constraints.add(
    #    ConstraintFcn.SUPERIMPOSE_MARKERS,
    #    node=Node.END,
    #    first_marker="endeffector",
    #    second_marker="endeffector_final"
    #)
    constraints.add(  # Reach ground with distal at the end of first phase
        ConstraintFcn.TRACK_MARKERS,
        node=Node.END,
        marker_index=bio_model.marker_index("endeffector"),
        index=2,
        min_bound=0,
        #max_bound=0.001
    )
    constraints.add(
        ConstraintFcn.TRACK_MARKERS,
        node=Node.ALL_SHOOTING,
        marker_index=bio_model.marker_index("base_contact_right_marker"),
        index=2,
        min_bound=0
    )

    dof_mapping = BiMappingList()
    #dof_mapping.add("q", to_second=[None, None, None, 0, 1], to_first=[3, 4])
    #dof_mapping.add("qdot", to_second=[None, None, None, 0, 1], to_first=[3, 4])
    #dof_mapping.add("tendons", to_second=[None, None, None, 0, 1], to_first=[3,4])

    x_bounds = BoundsList()
    x_bounds.add("q", bio_model.bounds_from_ranges("q"))
    x_bounds.add("qdot", bio_model.bounds_from_ranges("qdot"))
    x_bounds["q"][:, 0] = 0
    #x_bounds[0]["q"][1, 0] = 0.5
    #x_bounds[1]["q"][0, -1] = 1.2
    #x_bounds[1]["q"][1, -1] = 0.6
    #x_bounds["qdot"][:, 0] = 0
    #x_bounds["qdot"][:, -1] = 0

    u_bounds = BoundsList()
    u_bounds.add("tendons", min_bound=[0], max_bound=[40])

    u_init = InitialGuessList()
    u_init.add("tendons", [0.5])

    return OptimalControlProgram(
        bio_model,
        n_shooting=50,
        phase_time=2,
        objective_functions=objective_functions,
        constraints=constraints,
        #variable_mappings=dof_mapping,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        u_init=u_init
    )

def track_z_coordinate_of_marker(controller: PenaltyController, marker_name: str) -> MX:
    marker_idx = controller.model.marker_index(marker_name)
    markers_pos = controller.model.markers()(controller.states["q"].cx, controller.parameters.cx)
    return markers_pos[2, marker_idx]


def prepare_ocp(biorbd_model_path: str,) -> OptimalControlProgram:
    bio_model = (
        TendonBiorbdModel(biorbd_model_path, contact_types=[ContactType.RIGID_EXPLICIT]),
        TendonBiorbdModel(biorbd_model_path, contact_types=[ContactType.RIGID_EXPLICIT]),
    )

    endeffector_marker_idx = bio_model[0].marker_index("endeffector")

    objective_functions = ObjectiveList()
    objective_functions.add(Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tendons", phase=0))
    objective_functions.add(Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tendons", phase=1))
    #objective_functions.add(Objective(ObjectiveFcn.Mayer.TRACK_MARKERS, marker_index=endeffector_marker_idx, phase=0))
    #objective_functions.add(Objective(ObjectiveFcn.Mayer.MINIMIZE_TIME, phase=0))
    #objective_functions.add(Objective(ObjectiveFcn.Mayer.MINIMIZE_TIME, phase=1))

    constraints = ConstraintList()
    #constraints.add(
    #    track_z_coordinate_of_marker,
    #    node=Node.ALL_SHOOTING,
    #    marker_name="base_contact_right_marker",
    #    min_bound=0,
    #    max_bound=0.01
    #)
    constraints.add( # Keep base toe on the ground
        ConstraintFcn.TRACK_MARKERS,
        node=Node.ALL_SHOOTING,
        marker_index=bio_model[0].marker_index("base_contact_left_marker"),
        index=2,
        min_bound=0,
        max_bound=0.01,
        phase=0
    )
    constraints.add( # Keep base heel on the ground
        ConstraintFcn.TRACK_MARKERS,
        node=Node.ALL_SHOOTING,
        marker_index=bio_model[0].marker_index("base_contact_right_marker"),
        index=2,
        min_bound=0,
        max_bound=0.01,
        phase=0
    )
    #constraints.add( # Unilateral base toe contact (first phase)
    #    ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES,
    #    min_bound=0,
    #    max_bound=np.inf,
    #    node=Node.ALL_SHOOTING,
    #    phase=0,
    #    contact_index=1
    #)
    #constraints.add( # Unilateral base heel contact (first phase)
    #    ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES,
    #    min_bound=0,
    #    max_bound=np.inf,
    #    node=Node.ALL_SHOOTING,
    #    phase=0,
    #    contact_index=2
    #)
    #constraints.add(
    #    ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES,
    #    min_bound=0,
    #    max_bound=np.inf,
    #    node=Node.ALL_SHOOTING,
    #    phase=1,
    #    contact_index=1
    #)
    constraints.add( # Unilateral base heel contact (second phase)
        ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES,
        min_bound=0,
        max_bound=np.inf,
        node=Node.ALL_SHOOTING,
        phase=1,
        contact_index=2
    )
    constraints.add( # Unilateral distal contact (second phase)
        ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES,
        min_bound=0,
        max_bound=np.inf,
        node=Node.ALL_SHOOTING,
        phase=1,
        contact_index=3
    )
    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        node=Node.END,
        first_marker="endeffector",
        second_marker="endeffector_contact_inlet",
        phase=0
    )
    #constraints.add( # Reach ground with distal at the end of first phase
    #    ConstraintFcn.TRACK_MARKERS,
    #    node=Node.END,
    #    marker_index=endeffector_marker_idx,
    #    index=2,
    #    min_bound=0,
    #    phase=0
    #)
    #constraints.add( # Reach final location at the end of second phase
    #    ConstraintFcn.SUPERIMPOSE_MARKERS,
    #    node=Node.END,
    #    first_marker="endeffector",
    #    second_marker="endeffector_final",
    #    phase=1
    #)
    constraints.add(  # Keep base heel on the ground (second phase)
        ConstraintFcn.TRACK_MARKERS,
        node=Node.ALL_SHOOTING,
        marker_index=bio_model[0].marker_index("base_contact_right_marker"),
        index=2,
        min_bound=0,
        max_bound=0.01,
        phase=1
    )

    dof_mapping = BiMappingList()
    #dof_mapping.add("q", to_second=[None, None, None, 0, 1], to_first=[3,4])
    #dof_mapping.add("qdot", to_second=[None, None, None, 0, 1], to_first=[3,4])

    phase_transitions = PhaseTransitionList()
    phase_transitions.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=0)

    x_bounds = BoundsList()
    x_bounds.add("q", bio_model[0].bounds_from_ranges("q", mapping=dof_mapping), phase=0)
    x_bounds.add("q", bio_model[0].bounds_from_ranges("q", mapping=dof_mapping), phase=1)
    x_bounds.add("qdot", bio_model[0].bounds_from_ranges("qdot", mapping=dof_mapping), phase=0)
    x_bounds.add("qdot", bio_model[0].bounds_from_ranges("qdot", mapping=dof_mapping), phase=1)
    x_bounds[0]["q"][:, 0] = 0
    #x_bounds[0]["q"][0, 0] = 0.7
    #x_bounds[0]["q"][1, 0] = 0.5
    #x_bounds[1]["q"][0, -1] = 1.2
    #x_bounds[1]["q"][1, -1] = 0.6
    #x_bounds[0]["qdot"][:, 0] = 0
    #x_bounds["qdot"][:, -1] = 0

    u_bounds = BoundsList()
    u_bounds.add("tendons", min_bound=[0], max_bound=[50], phase=0)
    u_bounds.add("tendons", min_bound=[0], max_bound=[50], phase=1)

    return OptimalControlProgram(
        bio_model,
        n_shooting=[50,50],
        phase_time=(2,2),
        objective_functions=objective_functions,
        constraints=constraints,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        phase_transitions=phase_transitions,
        variable_mappings=dof_mapping,
        n_threads=10
    )

def main():
    ocp = prepare_single_phase_ocp(ExampleUtils.folder + "/models/test_finger.bioMod")
    ocp.print(to_console=True, to_graph=False)
    ocp.add_plot_penalty(CostType.CONSTRAINTS)
    sol = ocp.solve(Solver.IPOPT())
    sol.print_cost()
    sol.animate(n_frames=100)
    sol.graphs()

if __name__ == "__main__":
    main()
