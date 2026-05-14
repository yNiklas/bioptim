from bioptim import TendonBiorbdModel, OptimalControlProgram, Objective, ObjectiveFcn


def prepare_ocp(biorbd_model_path: str,) -> OptimalControlProgram:
    bio_model = TendonBiorbdModel(biorbd_model_path)
    print(f"nb_tendons: {bio_model.nb_tendons}")

    objective_functions = Objective(ObjectiveFcn.Mayer.MINIMIZE_TIME)

    return OptimalControlProgram(
        bio_model,
        n_shooting=50,
        phase_time=1,
        objective_functions=objective_functions
    )

def main():
    ocp = prepare_ocp()

if __name__ == "__main__":
    main()
