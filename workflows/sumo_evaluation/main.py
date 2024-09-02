### This file is the access point for "make python_example"
## It will sample for a simple function - to demonstrate feasibility of the approach
## When this is uploaded to OSPARC, the same dakota.in will be executed inside DakotaService
## and model evaluations will be executed within the ParallelRunner

import pathlib as pl
import dakota.environment as dakenv
import sys

script_dir = pl.Path(__file__).parent
print(script_dir)
sys.path.append(script_dir)

#


def main(dakota_conf_path=None):
    if dakota_conf_path is None:
        dakota_conf_path = script_dir / "dakota.in"

    print("Starting dakota")
    dakota_conf = dakota_conf_path.read_text()
    study = dakenv.study(
        # callbacks={"evaluator": model.evaluate},
        callbacks={},  ## DONT NEED ANY MODEL EVALUATIONS
        input_string=dakota_conf,
    )

    study.execute()


if __name__ == "__main__":
    from create_dakota_conf import create_dakota_conf

    dakota_conf_path = script_dir / "dakota.in"
    create_dakota_conf(dakota_conf_path)

    main()
