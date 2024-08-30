import pathlib as pl
import sys

script_dir = pl.Path(__file__).parent
sys.path.append(script_dir)
from utils import funs_creake_dakota_conf


def create_dakota_conf(dakota_conf_path):
    """Creates a minimalistic dakota configuration file that samples an LHS"""

    #

    print("Creating dakota file")
    dakota_conf_path = script_dir / "dakota.in"
    dakota_conf = funs_creake_dakota_conf.start_dakota_file()
    dakota_conf += funs_creake_dakota_conf.add_sampling_method()
    dakota_conf += funs_creake_dakota_conf.add_evaluator_model()
    # default, but always need an evaluator
    dakota_conf += funs_creake_dakota_conf.add_variables(
        ["X0", "X1", "X2"],
        lower_bounds=[0, 0, 0],
        upper_bounds=[1, 1, 1],
    )
    dakota_conf += funs_creake_dakota_conf.add_responses(["SUM", "MULT"])
    dakota_conf += funs_creake_dakota_conf.add_python_interface("evaluator")
    ## print & save, to be able to inspect
    print(dakota_conf)
    funs_creake_dakota_conf.write_to_file(dakota_conf, dakota_conf_path)

    #


if __name__ == "__main__":
    create_dakota_conf(script_dir / "dakota.in")
