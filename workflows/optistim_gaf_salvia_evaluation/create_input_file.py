"""
Create a sweep of comfort pulses, 
varying either the amplitude or the duration of the pulse.
"""

from get_pulse import SEGMENT_PW, DURATION
import numpy as np
import pandas as pd
import pathlib as pl

script_dir = pl.Path(__file__).parent

NVARS = int(DURATION / SEGMENT_PW)


def create_comfort_amplitude_sweep(pw=1.0):
    var_names = [f"p{i+1}" for i in range(NVARS)]
    series = []
    nseg_per_pw = round(pw / SEGMENT_PW)
    for amp in np.arange(-1, 1.01, 0.1):
        amp = np.round(amp, 1)
        vars = []
        for _ in range(5):
            for _ in range(nseg_per_pw):
                vars.append(amp)
            for _ in range(nseg_per_pw):
                vars.append(0)
        for _ in range(5):
            for _ in range(nseg_per_pw):
                vars.append(-amp)
        series.append(pd.Series(vars, index=var_names))
        ## NB: if pw is not 1.0, will need to fill up with 0s
    df = pd.concat(series, axis=1).T
    df.to_csv(script_dir / "COMFORT_1ms_amplitudesweep.csv", index=False, sep=" ")


if __name__ == "__main__":
    create_comfort_amplitude_sweep()
