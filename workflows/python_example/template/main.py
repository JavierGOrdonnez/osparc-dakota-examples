import os
import json
import pathlib as pl
import numpy as np
from typing import Dict, Union, Optional

INPUT_VARIABLES = ["X0", "X1"]
OUTPUT_RESPONSES = ["SUM", "MULT"]


def main():
    print(list(os.environ.keys()))
    input_path = pl.Path(os.environ["INPUT_FOLDER"])
    output_path = pl.Path(os.environ["OUTPUT_FOLDER"])

    input_file_path = input_path / "input.json"
    output_file_path = output_path / "output.json"

    input = json.loads(input_file_path.read_text())

    output = model(input)

    output_file_path.write_text(json.dumps(output))


def model(input_dict: Dict[str: float]):
    assert len(input_dict)==len(INPUT_VARIABLES), "Wrong number of input variables"
    assert np.all([invar in input_dict for invar in INPUT_VARIABLES]), "Wrong name of input variables"
    
    output = {}
    output["SUM"] = 0.
    output["MULT"] = 1.
    
    for k, v in input_dict.items():
        output["SUM"] += v
        output["MULT"] *= v

    return output


if __name__ == "__main__":
    main()
