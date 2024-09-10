import pathlib as pl
import sys

script_dir = pl.Path(__file__).parent
sys.path.append(script_dir)
import utils
from utils import funs_create_dakota_conf
from create_20D_variables import VARIABLES_DF


def create_dakota_conf(dakota_conf_path):
    """Creates a minimalistic dakota configuration file that samples an LHS"""

    #

    print("Creating dakota file")
    dakota_conf_path = script_dir / "dakota.in"
    dakota_conf = funs_create_dakota_conf.start_dakota_file()
    dakota_conf += funs_create_dakota_conf.add_sampling_method(num_samples=1000)
    dakota_conf += funs_create_dakota_conf.add_evaluator_model()
    # default, but always need an evaluator
    dakota_conf += funs_create_dakota_conf.add_variables(
        variables=VARIABLES_DF.index.tolist(),
        initial_points=VARIABLES_DF["initial_point"].tolist(),
        lower_bounds=VARIABLES_DF["lower_bound"].tolist(),
        upper_bounds=VARIABLES_DF["upper_bound"].tolist(),
    )
    dakota_conf += funs_create_dakota_conf.add_responses(["AFmax_4um", "GAFmax_4um"])
    dakota_conf += funs_create_dakota_conf.add_python_interface("INTERFACE")
    ## print & save, to be able to inspect
    print(dakota_conf)
    funs_create_dakota_conf.write_to_file(dakota_conf, dakota_conf_path)

    #


if __name__ == "__main__":
    create_dakota_conf(script_dir / "dakota.in")
