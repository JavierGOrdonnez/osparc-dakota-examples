import numpy as np
import pandas as pd

GEOMETRIC_VARIABLES = {
    "RELDEPTH": {
        "initial_point": 0.5,
        "lower_bound": 0.0,
        "upper_bound": 1.0,
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
    "ELECTRELDEPTH": {
        "initial_point": 0.5,
        "lower_bound": 0.0,
        "upper_bound": 1.0,
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
            "lower_bound": np.max([d[key]["mean"] - 3 * d[key]["std"], 0.0]),
            "upper_bound": d[key]["mean"] + 3 * d[key]["std"],
        }

        if key == "THICKNESS_LOOSE_AREOLAR_TISSUE":
            new_d[key]["lower_bound"] = 0.2
            ## The electrode's thickness is 0.2mm, and should be within the LAT

        if "THICKNESS" in key:  ## impose minimum 0.1mm thickness
            new_d[key]["lower_bound"] = np.max([new_d[key]["lower_bound"], 0.1])

        if "CONDUCTIVITY" in key:  ## impose minimum 0.001 S/m conductivity
            new_d[key]["lower_bound"] = np.max([new_d[key]["lower_bound"], 0.001])

    return new_d


VARIABLES_DICT = GEOMETRIC_VARIABLES.copy()
VARIABLES_DICT.update(convert_mean_std_to_initial_lower_upper(THICKNESS_VARIABLES))
VARIABLES_DICT.update(convert_mean_std_to_initial_lower_upper(CONDUCTIVITY_VARIABLES))

VARIABLES_DF = pd.DataFrame(VARIABLES_DICT).T
