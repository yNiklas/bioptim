import faulthandler

import numpy as np
from biorbd import contact_index

faulthandler.enable()

from bioptim import TendonBiorbdModel, OptimalControlProgram, Objective, ObjectiveFcn, BoundsList, InitialGuessList, \
    Solver, ConstraintList, ConstraintFcn, Node, ContactType, ObjectiveList, PhaseTransitionList, PhaseTransitionFcn, \
    CostType, TorqueBiorbdModel
from bioptim.examples.utils import ExampleUtils

def prepare_single_phase_ocp(biorbd_model_path: str,) -> OptimalControlProgram:
    bio_model = TendonBiorbdModel(biorbd_model_path, contact_types=[ContactType.RIGID_EXPLICIT])

    objective_functions = ObjectiveList()
    #objective_functions.add(Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tendons", phase=0))
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME)

    constraints = ConstraintList()
    constraints.add(
        ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES,
        min_bound=0,
        max_bound=np.inf,
        node=Node.END
    )
    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        node=Node.END,
        first_marker="endeffector",
        second_marker="endeffector_final"
    )

    x_bounds = BoundsList()
    x_bounds.add("q", bio_model.bounds_from_ranges("q"))
    x_bounds.add("qdot", bio_model.bounds_from_ranges("qdot"))
    x_bounds[0]["q"][:, 0] = 0
    #x_bounds[0]["q"][1, 0] = 0.5
    #x_bounds[1]["q"][0, -1] = 1.2
    #x_bounds[1]["q"][1, -1] = 0.63
    x_bounds[0]["qdot"][:, 0] = 0
    #x_bounds["qdot"][:, -1] = 0

    u_bounds = BoundsList()
    u_bounds.add("tendons", min_bound=[0], max_bound=[400])

    return OptimalControlProgram(
        bio_model,
        n_shooting=50,
        phase_time=1,
        objective_functions=objective_functions,
        constraints=constraints,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
    )



def prepare_ocp(biorbd_model_path: str,) -> OptimalControlProgram:
    bio_model = (
        TendonBiorbdModel(biorbd_model_path, contact_types=[ContactType.RIGID_EXPLICIT]),
        TendonBiorbdModel(biorbd_model_path, contact_types=[ContactType.RIGID_EXPLICIT]),
    )

    objective_functions = ObjectiveList()
    #objective_functions.add(Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tendons", phase=0))
    #objective_functions.add(Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tendons", phase=0))
    objective_functions.add(Objective(ObjectiveFcn.Mayer.MINIMIZE_TIME, phase=0))
    objective_functions.add(Objective(ObjectiveFcn.Mayer.MINIMIZE_TIME, phase=1))

    constraints = ConstraintList()
    constraints.add(
        ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES,
        min_bound=0,
        max_bound=np.inf,
        node=Node.ALL_SHOOTING,
        phase=0,
        contact_index=1
    )
    constraints.add(
        ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES,
        min_bound=0,
        max_bound=np.inf,
        node=Node.ALL_SHOOTING,
        phase=0,
        contact_index=2
    )
    constraints.add(
        ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES,
        min_bound=0,
        max_bound=np.inf,
        node=Node.ALL_SHOOTING,
        phase=1,
        contact_index=1
    )
    constraints.add(
        ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES,
        min_bound=0,
        max_bound=np.inf,
        node=Node.ALL_SHOOTING,
        phase=1,
        contact_index=2
    )
    constraints.add(
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
    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        node=Node.END,
        first_marker="endeffector",
        second_marker="endeffector_final",
        phase=1
    )

    phase_transitions = PhaseTransitionList()
    phase_transitions.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=0)

    x_bounds = BoundsList()
    x_bounds.add("q", bio_model[0].bounds_from_ranges("q"), phase=0)
    x_bounds.add("q", bio_model[0].bounds_from_ranges("q"), phase=1)
    x_bounds.add("qdot", bio_model[0].bounds_from_ranges("qdot"), phase=0)
    x_bounds.add("qdot", bio_model[0].bounds_from_ranges("qdot"), phase=1)
    x_bounds[0]["q"][:, 0] = 0
    #x_bounds[0]["q"][0, 0] = 0.7
    #x_bounds[0]["q"][1, 0] = 0.5
    #x_bounds[1]["q"][0, -1] = 1.2
    #x_bounds[1]["q"][1, -1] = 0.6
    #x_bounds[0]["qdot"][:, 0] = 0
    #x_bounds["qdot"][:, -1] = 0

    u_bounds = BoundsList()
    u_bounds.add("tendons", min_bound=[0], max_bound=[40], phase=0)
    u_bounds.add("tendons", min_bound=[0], max_bound=[40], phase=1)

    return OptimalControlProgram(
        bio_model,
        n_shooting=[50,50],
        phase_time=(1,1),
        objective_functions=objective_functions,
        constraints=constraints,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        phase_transitions=phase_transitions
    )

def main():
    ocp = prepare_ocp(ExampleUtils.folder + "/models/tendon_2dof_finger_with_contact.bioMod")
    ocp.print(to_console=False, to_graph=False)
    ocp.add_plot_penalty(CostType.CONSTRAINTS)
    sol = ocp.solve(Solver.IPOPT())
    sol.print_cost()
    sol.animate(n_frames=100)
    sol.graphs()

if __name__ == "__main__":
    main()
