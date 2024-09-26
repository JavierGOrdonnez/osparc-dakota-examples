# import s4l_neurofunctions as nf
# from s4l_neurofunctions.neuron import StimulationPulse
from stimulation_pulse import StimulationPulse
import numpy as np


SEGMENT_PW = 0.1  # Width, in ms, of 1 pulse segment
DURATION = 15.0
## eg 150 free parameters!! Let's see


def get_pulse(
    *args, segment_pw: float = SEGMENT_PW, duration=DURATION, stds=None
) -> StimulationPulse:
    assert len(args) == int(
        duration / segment_pw
    ), "Number of arguments must match the duration of the pulse"

    pulse_object = StimulationPulse(None)
    pulse_object.name = "Free Pulse"

    args = np.array(args)
    mean = np.round(np.sum(args), 4)
    print("mean: ", mean)  # we will substract this from the pulse
    current_balanced_amplitudes = args - mean

    # Create the pulse
    for amp in current_balanced_amplitudes:
        pulse_object._insert_time_interval(amp, segment_pw)

    pulse_object.finish_pulse(DURATION)

    ## TODO include stds (for plotting) - previously called "errors"

    return pulse_object
