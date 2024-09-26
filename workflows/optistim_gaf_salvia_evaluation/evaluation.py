# import s4l_neurofunctions as nf
from get_pulse import get_pulse
from get_pulse import SEGMENT_PW, DURATION  ## overwrite if necessary

import numpy as np


def deactivate_tqdm():
    import os

    os.environ["TQDM_DISABLE"] = "1"

    from tqdm import tqdm
    from functools import partialmethod

    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)  # type: ignore


deactivate_tqdm()


def evaluator(inputs):
    params = inputs["cv"]
    print(f"Evaluating Free Pulse for {params}")
    return {
        # "fns": {
        #     "activation": evaluate_activation(params),
        #     "energy": evaluate_energy(params),
        # }
        "fns": [
            evaluate_activation(params),
            evaluate_energy(params),
        ]
    }


def evaluate_activation(x) -> float:
    ## Can not be evaluated without the model
    # pulse = get_pulse(*x, segment_pw=SEGMENT_PW, duration=DURATION)
    # activation = nf.evaluate_activation(pulse)

    print(type(x))
    activation = np.sqrt(np.sum(np.array(x) ** 2))  ## mockup
    print(f"Activation: {activation}")
    return activation


def evaluate_energy(x) -> float:
    pulse = get_pulse(*x, segment_pw=SEGMENT_PW, duration=DURATION)
    R = 1.2e3  # 1.2 kOhm -- from the model, +-1V generates 1.65mA
    ## total work = sum I^2 * R * dt
    energy = [(i * 1e-3) ** 2 * R * SEGMENT_PW for i in pulse.amplitude_list]
    energy = sum(energy)
    print(f"Energy: {energy}")
    return energy


if __name__ == "__main__":
    ## mockup
    x = [0.1 for _ in range(150)]
    evaluate_activation(x)
    evaluate_energy(x)
    print("Done")
