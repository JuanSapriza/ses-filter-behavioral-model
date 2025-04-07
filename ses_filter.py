import numpy as np
from timeseries import *

def ses_filter(series, window_n, gain_b ):
    """
    Calculate pseudo low pass filter (LPF) for the input time series as a simplification
    of the Simple Exponential Smoothing Smoothing (SES) aka a cooler moving mean.

    Args:
        series (Timeseries): Input time series: a structure with .data and .time
        (two numpy arrays of the same length) and containing the input signal (data)
        and the timestamp of each sample (time).
        window_n: The number of samples that will form the window of the moving mean.
        gain_b (int): Number of bits for scaling. If the input is two small, i.e. less bits
        than log2(window_n), then you will have major degradation every time you do the
        division. It is preferable to scale the input by adding some bits as LSB. How many bits
        are added is gain_b. E.g. if the input is 2-bit wide, you might take it to a 16 bits
        word by left-shifting 14-bits => gain_b is 14.

    Returns:
        Timeseries: Time series with pseudo mean calculated.
    """
    o = Timeseries(f"Pseudo-LPFd ({window_n:0.0f} n/{gain_b:0.0f} b)", time = series.time )
    o.params.update(series.params)
    o.params[ TSP_SAMPLE_B] = gain_b
    m           = int((2**gain_b)/2) # The mean starts at the middle of the range
    bits        = int(np.log2(window_n))
    mb          = int(m << bits)
    data        = []

    for i in range( len(series.data) ):
        # This approach is MUCH cheaper than doing the following:
        # m = np.average(sar.conversion.data[i-lpf_win_n:i])
        # and does not require keeping values stored in registers :)
        # If data does not come at the desired resolution, an extra left bit shift is required (see ADC-emu's amplify val)
        # In the case where the SAR ADC is 4 bits and the final resolution is 16 bits, then it's as simple
        # as putting the 4 bits as the 4 MSB of a 16 bit word, no computation needed.
        mb  = mb - m + (int(series.data[i])<<int(gain_b))   # m[i]xb = m[i-1]xb - m[i-1] + s[i]
        m   = mb >> bits                                    # m[i] = m[i]xb /b
        data.append(m)
    o.data = np.array(data)
    return o.copy()


