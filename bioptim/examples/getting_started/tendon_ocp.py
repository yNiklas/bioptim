import faulthandler
faulthandler.enable()

from bioptim import TendonBiorbdModel, OptimalControlProgram, Objective, ObjectiveFcn, BoundsList, InitialGuessList, \
    Solver
from bioptim.examples.utils import ExampleUtils


def prepare_ocp(biorbd_model_path: str,) -> OptimalControlProgram:
    bio_model = TendonBiorbdModel(biorbd_model_path)
    print(f"nb_tendons: {bio_model.nb_tendons}")
    print(f"tendon_names: {bio_model.tendon_names}")

    #objective_functions = Objective(ObjectiveFcn.Mayer.MINIMIZE_TIME)
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tendons")

    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][0, 0] = 0.2
    x_bounds["q"][0, -1] = 0.4
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"][0, 0] = 0
    x_bounds["qdot"][0, -1] = 0

    u_bounds = BoundsList()
    u_bounds["tendons"] = [0], [100]

    return OptimalControlProgram(
        bio_model,
        n_shooting=50,
        phase_time=2,
        objective_functions=objective_functions,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
    )

def main():
    ocp = prepare_ocp(ExampleUtils.folder + "/models/tendon_manipulator.bioMod")
    print("Prepared OCP")
    ocp.print(to_console=False, to_graph=False)
    sol = ocp.solve(Solver.IPOPT())
    sol.print_cost()
    sol.animate(n_frames=50)
    sol.graphs()

if __name__ == "__main__":
    main()
