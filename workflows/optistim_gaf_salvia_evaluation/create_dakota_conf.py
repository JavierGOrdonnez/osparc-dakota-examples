import pathlib as pl
import sys, os
import importlib
import typing

script_dir = pl.Path(__file__).parent
from get_pulse import SEGMENT_PW, DURATION

if typing.TYPE_CHECKING:
    from utils import funs_create_dakota_conf
else:
    utils_dir = script_dir.parent / "utils"
    sys.path.append(str(utils_dir))
    importlib.invalidate_caches()
    import funs_create_dakota_conf


NVARS = int(DURATION / SEGMENT_PW)
INPUT_FILE = script_dir / "COMFORT_1ms_amplitudesweep.csv"


def create_dakota_conf(
    dakota_conf_path: pl.Path,
) -> None:
    """Evaluate a list of pulse amplitudes"""

    #

    print("Creating dakota file")
    dakota_conf_path = script_dir / "dakota.in"
    dakota_conf = funs_create_dakota_conf.start_dakota_file()
    dakota_conf += funs_create_dakota_conf.add_evaluation_method(
        INPUT_FILE, model_pointer="GAF_EVALUATOR", includes_eval_id=False
    )
    dakota_conf += funs_create_dakota_conf.add_evaluator_model(id_model="GAF_EVALUATOR")
    dakota_conf += funs_create_dakota_conf.add_variables(
        variables=[f"p{i+1}" for i in range(NVARS)],
        # lower_bounds=[-1.0 for _ in range(NVARS)],
        # upper_bounds=[1.0 for _ in range(NVARS)],
        ## TODO check max amplitudes provided by Salvia
    )
    dakota_conf += funs_create_dakota_conf.add_python_interface(
        evaluation_function="batch_evaluator", batch_mode=True
    )
    dakota_conf += funs_create_dakota_conf.add_responses(
        [
            "activation",
            "energy",
        ]
    )

    ## print & save, to be able to inspect
    print(dakota_conf)
    funs_create_dakota_conf.write_to_file(dakota_conf, dakota_conf_path)


if __name__ == "__main__":
    create_dakota_conf(script_dir / "dakota.in")
