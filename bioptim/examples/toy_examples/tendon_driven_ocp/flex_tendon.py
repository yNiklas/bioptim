from bioptim import TendonBiorbdModel, ObjectiveList, Objective, ObjectiveFcn, ConstraintList, BoundsList, \
    OptimalControlProgram, SolutionMerge, PhaseTransitionList, PhaseTransitionFcn
from bioptim.examples.utils import ExampleUtils


def single_ocp():
    model = TendonBiorbdModel(ExampleUtils.folder + "/models/tendon_test_finger.bioMod")
    obj = ObjectiveList()
    obj.add(Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tendons"))
    x_bounds = BoundsList()
    x_bounds.add("q", model.bounds_from_ranges("q"))
    x_bounds.add("qdot", model.bounds_from_ranges("qdot"))
    x_bounds["q"][:, 0] = 0
    x_bounds["q"][0, -1] = 0.4
    x_bounds["qdot"][:, -1] = 0
    u_bounds = BoundsList()
    u_bounds.add("tendons", min_bound=[0], max_bound=[10])
    return OptimalControlProgram(
        model,
        50,
        2,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=obj,
    )

def prepare_ocp():
    model = (
        TendonBiorbdModel(ExampleUtils.folder + "/models/tendon_test_finger.bioMod"),
        TendonBiorbdModel(ExampleUtils.folder + "/models/tendon_test_finger.bioMod"),
    )

    obj = ObjectiveList()
    obj.add(Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tendons"))

    constraints = ConstraintList()
    phase_transition = PhaseTransitionList()
    phase_transition.add(PhaseTransitionFcn.CONTINUOUS, phase_pre_idx=0)

    x_bounds = BoundsList()
    x_bounds.add("q", model[0].bounds_from_ranges("q"), phase=0)
    x_bounds.add("q", model[0].bounds_from_ranges("q"), phase=1)
    x_bounds.add("qdot", model[0].bounds_from_ranges("qdot"), phase=0)
    x_bounds.add("qdot", model[0].bounds_from_ranges("qdot"), phase=1)
    x_bounds[0]["q"][:, 0] = 0
    x_bounds[0]["q"][0, -1] = 0.5
    #x_bounds[0]["qdot"][:, -1] = 0
    x_bounds[1]["q"][:, -1] = 0
    x_bounds[1]["qdot"][:, -1] = 0
    u_bounds = BoundsList()
    u_bounds.add("tendons", min_bound=[0], max_bound=[100])
    return OptimalControlProgram(
        model,
        [50,50],
        [3,3],
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=obj,
        phase_transitions=phase_transition,
    )

if __name__ == '__main__':
    ocp = prepare_ocp()
    sol = ocp.solve()
    sol.print_cost()
    sol_controls = sol.decision_controls(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["tendons"]
    print(sol_controls)
    sol.animate(n_frames=100)
    sol.graphs()
