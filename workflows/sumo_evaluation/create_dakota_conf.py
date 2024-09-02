import pathlib as pl
import sys

script_dir = pl.Path(__file__).parent
sys.path.append(script_dir)
from utils import funs_create_dakota_conf, funs_data_processing

if True:  # VARIABLES_DF
    import numpy as np
    import pandas as pd

    # from adaptive_sampling_19D import VARIABLES_DF  # 19D as default
    GEOMETRIC_VARIABLES = {
        "DEPTH": {
            "initial_point": 1.0,
            "lower_bound": 0.5,
            "upper_bound": 1.5,
        },
        "DIAMETER": {
            "initial_point": 0.3,
            "lower_bound": 0.1,
            "upper_bound": 1.0,
        },
        # "POSITION": {
        #     "initial_point": -14.0,
        #     "lower_bound": -21.0,
        #     "upper_bound": 21.0,
        # },
        "ANGLE": {
            "initial_point": 90.0,
            "lower_bound": 0.0,
            "upper_bound": 90.0,
        },
    }

    THICKNESS_VARIABLES = {
        "THICKNESS_SKIN": {
            "mean": 0.514,
            "std": 0.108,
        },
        "THICKNESS_SCT": {
            "mean": 0.944,
            "std": 0.331,
        },
        "THICKNESS_APONEUROSIS": {
            "mean": 0.247,
            "std": 0.101,
        },
        "THICKNESS_LOOSE_AREOLAR_TISSUE": {
            "mean": 0.655,
            "std": 0.376,
        },
        "THICKNESS_SKULL_OUTER": {
            "mean": 1.83,
            "std": 0.5,  # TODO find real value!!
        },
        "THICKNESS_SKULL_DIPLOE": {
            "mean": 2.84,
            "std": 0.5,  # TODO find real value!!
        },
        "THICKNESS_SKULL_INNER": {
            "mean": 1.44,
            "std": 0.5,  # TODO find real value!!
        },
        # THICKNESS_CSF=3.6,  ## https://www.researchgate.net/figure/Thickness-of-scalp-skull-and-CSF-layer_fig14_51754044
        "THICKNESS_CSF": {
            "mean": 3.6,
            "std": 0.5,  # TODO find real value!!
        },
    }

    ### Or use min-max values in table??
    CONDUCTIVITY_VARIABLES = {
        "CONDUCTIVITY_SKIN": {
            "mean": 0.148297101,
            "std": 0.042145684,
        },
        "CONDUCTIVITY_SCT": {
            "mean": 0.077621273,
            "std": 0.093082005,
        },
        "CONDUCTIVITY_APONEUROSIS": {  ## connective tissue
            "mean": 0.079195857,
            "std": 0.023344569,
        },
        "CONDUCTIVITY_LOOSE_AREOLAR_TISSUE": {  ## connective tissue
            "mean": 0.079195857,
            "std": 0.023344569,
        },
        "CONDUCTIVITY_SKULL_CORTICAL": {
            "mean": 0.00644673,
            "std": 0.002535023,
        },
        "CONDUCTIVITY_SKULL_DIPLOE": {
            "mean": 0.09975,
            "std": 0.121595589,
        },
        "CONDUCTIVITY_CSF": {
            "mean": 1.87899971,
            "std": 0.685074315,
        },
    }

    def convert_mean_std_to_initial_lower_upper(d: dict) -> dict:
        new_d = {}
        for key in d:
            new_d[key] = {
                "initial_point": d[key]["mean"],
                "lower_bound": np.max([d[key]["mean"] - 2 * d[key]["std"], 0.0]),
                "upper_bound": d[key]["mean"] + 2 * d[key]["std"],
            }
        return new_d

    VARIABLES_DICT = GEOMETRIC_VARIABLES.copy()
    VARIABLES_DICT.update(convert_mean_std_to_initial_lower_upper(THICKNESS_VARIABLES))
    VARIABLES_DICT.update(
        convert_mean_std_to_initial_lower_upper(CONDUCTIVITY_VARIABLES)
    )

    ##### convert keys to Cedric's notation ########
    NEW_VARIABLES_DICT = {
        k.title().replace("_", ""): v for k, v in VARIABLES_DICT.items()
    }
    ################################################

    VARIABLES_DF = pd.DataFrame(NEW_VARIABLES_DICT).T


def create_dakota_conf(
    dakota_conf_path,
    TRAINING_SAMPLES_FILE="itrainsample.csv",
    CROSS_VALIDATION_FOLDS=10,
):
    """Creates a GP surrogate model based on some data file.
    RMSE and other metrics are evaluated through cross-validation.
    """

    #

    print("Creating dakota file")
    dakota_conf_path = script_dir / "dakota.in"
    dakota_conf = funs_create_dakota_conf.start_dakota_file()
    TRAINING_SAMPLES_FILE = funs_data_processing.process_input_file(
        script_dir / TRAINING_SAMPLES_FILE,
        columns_to_remove=["Interface"],
    )
    dakota_conf += funs_create_dakota_conf.add_surrogate_model(
        TRAINING_SAMPLES_FILE,
        cross_validation_folds=CROSS_VALIDATION_FOLDS,
    )
    dakota_conf += funs_create_dakota_conf.add_variables(
        variables=VARIABLES_DF.index.tolist(),
        # initial_points=VARIABLES_DF["initial_point"].tolist(),
        # lower_bounds=VARIABLES_DF["lower_bound"].tolist(),
        # upper_bounds=VARIABLES_DF["upper_bound"].tolist(),
        # lower_bounds=[-10] * len(VARIABLES_DF),
        # upper_bounds=[10] * len(VARIABLES_DF),
    )
    dakota_conf += funs_create_dakota_conf.add_responses(descriptors=["AfPeak"])

    ## I believe I will need a method, even if never called upon
    # dakota_conf += funs_create_dakota_conf.add_evaluation_method(
    #     None
    #     # TRAINING_SAMPLES_FILE
    # )  # GIVING ERROR (N VARS?)
    # dakota_conf += funs_creake_dakota_conf.add_sampling_method(
    #     num_samples=10
    # )  # Need lower & upper bound
    # dakota_conf += funs_create_dakota_conf.add_python_interface("evaluator")
    dakota_conf += funs_create_dakota_conf.add_moga_method(
        max_iterations=1,
        population_size=2,
        # minimal, we are not interested, it is just to have a method
    )

    ## print & save, to be able to inspect
    print(dakota_conf)
    funs_create_dakota_conf.write_to_file(dakota_conf, dakota_conf_path)

    #


if __name__ == "__main__":
    create_dakota_conf(script_dir / "dakota.in")
