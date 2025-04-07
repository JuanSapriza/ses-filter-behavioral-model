from timeseries import *
from ts_params import *

import matplotlib.pyplot as plt

def pseudo_lpf(series, window_n, gain_b ):
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
        mb  = mb - m + (int(series.data[i])<<int(gain_b))   # m[i]xb = m[i-1]xb - m[i-1] + s[i]]
        m   = mb >> bits                                    # m[i] = m[i]xb /b
        data.append(m)
    o.data = np.array(data)
    return o.copy()

import itertools
def generate_combinations(V, D):
    if D == 1: return [(v,) for v in V]  # Return a list of tuples with one element each
    else: return list(itertools.combinations_with_replacement(V, D))



# Plot the inputs
def plot_series( series, format='step', figsize=(15,3) ):
    plt.figure( figsize=figsize )
    for s in series:
        if format == 'step':
            plt.step(s.time, s.data, label=s, alpha=0.5, where='post' )
        else:
            plt.plot(s.time, s.data, label=s, alpha=0.5 )
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
    plt.show()

def normalize_data( data ):
    data -= min(data)
    data = data/( max(data))
    return data

def align_signal_rmse( signal, reference, attempts=30, step_s=30e-6, percentile=99 ):
    if signal is None or len(signal.time) < 2: return None
    dx_s        = step_s
    best_mse    = -1000
    best_dx     = 0

    for i in range( attempts ):
        ref = reference.copy()
        sig = signal.copy()

        sig.time -= dx_s*i

        sig = interpolate_series_to_match( sig, ref )

        ref = crop_series_time_percentile(ref, percentile)
        sig = crop_series_time_percentile(sig, percentile)

        ref = zero_mean_unit_variance(ref)
        sig = zero_mean_unit_variance(sig)

        mse = compute_rmse(sig, ref)
        if mse > best_mse:
            best_mse    = mse
            best_dx     = dx_s*i
            best_ref    = ref
            best_sig    = sig

    signal.time = np.array(signal.time, dtype=np.float32 ) - np.float32(best_dx)
    signal.params[TSP_T_OFFSET_S]   = best_dx
    signal.params[TSP_SCORE_MSE_DB] = best_mse

    return best_mse


def lc_reconstruct( series, rec_type, order=1, ):
    if series is None or len(series.time) == 0: return None
    x           = series.copy()
    # Reconstruct the LC signal into a piece-wise polynomial function (i.e. linear if order=1)
    lc_rec      = rec_type(x) # This produces a sparse signal!
    lc_rec_fr   = rec_piecewise_poly_fmin(lc_rec, order=order)

    lc_rec_fr.name += f"rec P({order})"
    lc_rec_fr.params.update(x.params)
    lc_rec_fr.params[TSP_LC_REC_METHOD]= f"{type.__name__}"
    lc_rec_fr.params[TSP_LC_REC_ORDER] = order


    start=1
    for ignore_xings,d in enumerate(series.data[start:]):
        if np.sign(d) != np.sign(series.data[start+ignore_xings-1]): break

    lc_rec_fr.time = np.array(lc_rec_fr.time[start+ignore_xings:], dtype=np.float32)
    lc_rec_fr.data = np.array(lc_rec_fr.data[start+ignore_xings:], dtype=np.float32)

    return lc_rec_fr.copy()

def lc_compute_datarate( series ):
    if series is None or len(series.time) == 0: return None
    x = series

    start=1
    for ignore_xings,d in enumerate(x.data[start:]):
        if np.sign(d) != np.sign(x.data[start+ignore_xings-1]): break

    # Compute the data width required to acquire or transmit samples (in transmission we assume live transmission and therefore no time is sent)

    # First compute the maximum number of xings or skipped sampled for any LC acquisition
    max_lvl_change      = int(np.mean(abs(x.data[ignore_xings:])) + 3*np.std(abs(x.data[ignore_xings:])))
    max_smpl_skip       = max(x.time[ignore_xings:])

    # One bit for direction
    format_dirs_b       = 1
    format_lvls_b       = np.ceil(np.log2( max_lvl_change +1 ))
    # The time is simply the number of samples skipped. Two subsequent samples are at a distance of 1 (because we leave time=0 to send overflow data)
    format_time_b       = np.log2(max_smpl_skip)

    # We define the two data types: for acquisition and for transmission
    # # The data-rate (bps) sneding only crossings time information is implicit, considers the min number of bits of both Xings and time +1 (because of the sign)
    x_b = format_dirs_b + format_lvls_b
    t_b = format_time_b

    x.params[TSP_LC_SAMPLE_X_B]         = x_b
    x.params[TSP_LC_SAMPLE_T_B]         = t_b

    x.params[TSP_SCORE_AVG_DR_ACQ_BPS]  = (t_b+x_b) *x.params[TSP_LC_AVG_ACQ_F_HZ]
    x.params[TSP_SCORE_AVG_DR_TX_BPS]   = x_b       *x.params[TSP_LC_AVG_ACQ_F_HZ]

    x.params[TSP_SCORE_MAX_DR_ACQ_BPS]  = (t_b+x_b) *x.params[TSP_LC_MAX_ACQ_F_HZ]
    x.params[TSP_SCORE_MAX_DR_TX_BPS]   = x_b       *x.params[TSP_LC_MAX_ACQ_F_HZ]

    x.params[TSP_SCORE_STD_DR_ACQ_BPS]  = (t_b+x_b) *x.params[TSP_LC_STD_ACQ_F_HZ]
    x.params[TSP_SCORE_STD_DR_TX_BPS]   = x_b       *x.params[TSP_LC_STD_ACQ_F_HZ]

    # print(f"{format_dirs_b:0.0f}\t{format_lvls_b:0.0f}\t{format_time_b:0.0f}\t{x_b:0.0f} ({lcs.params[TSP_SCORE_AVG_DR_TX_BPS]:0.0f} bps)\t{xt_b:0.0f} ({lcs.params[TSP_SCORE_AVG_DR_ACQ_BPS]:0.0f} bps)")
    return




def lc_power_model( s, ref, hw_blocks ):

    report = []
    # Moving mean stages
    for window_n in s.params[TSP_MEAN_WIN_N]:
        Wg = int(np.log2(s.params[TSP_FITLER_GAIN]))
        report.append(instance(hw_blocks['subs'], Wg,ref.params[TSP_F_HZ],1))
        report.append(instance(hw_blocks['addr'], Wg,ref.params[TSP_F_HZ],1))
        report.append(instance(hw_blocks['addr'], Wg,ref.params[TSP_F_HZ],2))

    # Downsampling
    counter_bits = np.ceil(np.log2(ref.params[TSP_F_HZ]/s.params[TSP_F_HZ]))
    report.append(instance(hw_blocks['cntr'], counter_bits, ref.params[TSP_F_HZ], 1))

    # LC comparison
    report.append(instance(hw_blocks['regs'], s.params[TSP_LC_LVL_W_B],s.params[TSP_F_HZ], 2))
    report.append(instance(hw_blocks['subs'], s.params[TSP_LC_LVL_W_B],s.params[TSP_F_HZ], 2))
    report.append(instance(hw_blocks['mux4'], s.params[TSP_LC_LVL_W_B],s.params[TSP_F_HZ], 1))
    report.append(instance(hw_blocks['regs'], s.params[TSP_LC_LVL_W_B],s.params[TSP_LC_AVG_ACQ_F_HZ], 3))
    report.append(instance(hw_blocks['incr'], s.params[TSP_LC_LVL_W_B],s.params[TSP_LC_AVG_ACQ_F_HZ], 2))

    # Accumulating results
    trans_count, complexity = 0,0
    for r in report:
        trans_count  += r['Transistors']
        complexity   += r['Power (complexity)']

    s.params[TSP_COST_BREAKDOWN]    = report
    s.params[TSP_SCORE_TRANS_N]     = trans_count
    s.params[TSP_SCORE_COMPLEXITY]  = complexity
    return


def cic_power_model( s, ref, hw_blocks ):
    report = []

    # Integrator stages
    for wl in s.params[TSP_CIC_STAGE_REG_B][:s.params[TSP_CIC_STAGES]]:
        report.append(instance(hw_blocks['addr'],wl,ref.params[TSP_F_HZ],1))
        report.append(instance(hw_blocks['regs'],wl,ref.params[TSP_F_HZ],1))

    # Subsampling
    report.append(instance(hw_blocks['cntr'],np.ceil(np.log2(s.params[TSP_CIC_RATE_CHANGE])), ref.params[TSP_F_HZ], 1))
    report.append(instance(hw_blocks['regs'],s.params[TSP_CIC_STAGE_REG_B][s.params[TSP_CIC_STAGES]],s.params[TSP_F_HZ], 2))

    # Comb stages
    for wl in s.params[TSP_CIC_STAGE_REG_B][s.params[TSP_CIC_STAGES]+1:]:
        report.append(instance(hw_blocks['subs'],wl,s.params[TSP_F_HZ],1))
        report.append(instance(hw_blocks['regs'],wl,s.params[TSP_F_HZ],1))
        report.append(instance(hw_blocks['regs'],wl,s.params[TSP_F_HZ],s.params[TSP_CIC_DIFF_DELAY]))

    # Optional extra subtractor to only send differences
    report.append(instance(hw_blocks['subs'], wl, s.params[TSP_F_HZ],1))
    report.append(instance(hw_blocks['regs'], wl, s.params[TSP_F_HZ],1))

    # Accumulate results
    trans_count, complexity = 0,0
    for r in report:
        trans_count  += r['Transistors']
        complexity   += r['Power (complexity)']

    s.params[TSP_COST_BREAKDOWN]    = report
    s.params[TSP_SCORE_TRANS_N]     = trans_count
    s.params[TSP_SCORE_COMPLEXITY]  = complexity
    return



def diff_series( s ):
    o_data = []
    for i in range(1, len(s.data)):
        o_data.append(int(s.data[i] - s.data[i-1]))
    o_data = np.array(o_data)
    o_time = s.time[1:]
    o = Timeseries("Diff'd", time=o_time, data=o_data)
    o.params.update(s.params)
    sample_b = np.ceil(np.log2(max(o_data) - min(o_data)))

    o.params[TSP_SAMPLE_B]      = sample_b
    o.params[TSP_SCORE_DR_BPS]  = sample_b*o.params[TSP_F_HZ]

    rec = o.copy()
    for i in range(1,len(rec.data)):
        rec.data[i] = rec.data[i-1] + rec.data[i]

    o.params[TSP_REC_SERIES] = rec
    s.params[TSP_DIFF_SERIES] = o.copy()
    return




def find_trade_off_points(points, lower_better):
    n = len(points)
    is_trade_off_point = np.ones(n, dtype=bool)  # Assume all points are trade-off points initially

    # Bounding boxes for minimum and maximum values found so far
    min_bounds = np.full((len(lower_better)), np.inf)
    max_bounds = np.full((len(lower_better)), -np.inf)

    for i in range(n):
        for d in range(len(lower_better)):
            if lower_better[d]:
                min_bounds[d] = min(min_bounds[d], points[i][d])
            else:
                max_bounds[d] = max(max_bounds[d], points[i][d])

        for j in range(n):
            if i != j and is_trade_off_point[j]:
                if all((points[j][d] <=points[i][d] if lower_better[d] else points[j][d] > points[i][d])
                       for d in range(len(points[i]))) and \
                   any((points[j][d] < points[i][d] if lower_better[d] else points[j][d] > points[i][d])
                       for d in range(len(points[i]))):
                    is_trade_off_point[i] = False
                    break

    # Filter points that are still marked as possible trade-off points
    strict_trade_off_points = np.array(points)[is_trade_off_point]

    # Sort the points by the x values (first column)
    sorted_points = strict_trade_off_points[strict_trade_off_points[:, 0].argsort()]

    # Split into two arrays, one for x and one for y
    x_values = sorted_points[:, 0]
    y_values = sorted_points[:, 1]

    return x_values, y_values



def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def NEO(x):
    neo_signal = np.zeros(len(x))
    neo_signal[1:-1] = x[1:-1]**2 - x[2:] * x[:-2]
    return neo_signal

def calculate_f1_score(true_timestamps, predicted_timestamps, tolerance):
    true_timestamps = np.array(true_timestamps)
    predicted_timestamps = np.array(predicted_timestamps)

    # Calculate matches based on tolerance
    true_positive = 0
    for predicted_time in predicted_timestamps:
        # Check if any true timestamp matches the predicted timestamp within the tolerance
        if np.any(np.abs(true_timestamps - predicted_time) <= tolerance):
            true_positive += 1

    # Calculate precision and recall
    precision = true_positive / len(predicted_timestamps) if predicted_timestamps.size > 0 else 0
    recall = true_positive / len(true_timestamps) if true_timestamps.size > 0 else 0

    # Calculate F-1 score
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0

    return f1_score


def spike_detection_neo_f1( series, ann, tolerance_s=0.15e-3, fi_Hz=500, ff_Hz=5000, thrh=5, plot=False  ):

    fs = series.params[TSP_F_HZ]
    time_axis = series.time
    signal = series.data
    ymax = max(signal)
    ymin = min(signal)

    filtered_signal = bandpass_filter(signal, fi_Hz, ff_Hz, fs, order=4)
    neo_signal = NEO(filtered_signal)

    mean_neo = np.mean(neo_signal)
    std_neo = np.std(neo_signal)
    threshold = mean_neo + thrh * std_neo  # Adjust multiplier based on data characteristics

    potential_spikes = np.where(neo_signal > threshold)[0]

    min_distance = int(0.001 * fs)  # 1 ms refractory period
    spike_indices = []
    last_spike = -np.inf

    for idx in potential_spikes:
        if idx - last_spike > min_distance:
            spike_indices.append(idx)
            last_spike = idx

    spike_times = time_axis[0] + np.array(spike_indices) / fs  # Convert indices to time stamps

    ann_spike_times = [ t for t in ann if t >= time_axis[0] and t <= time_axis[-1] ]


    f1 = calculate_f1_score( ann_spike_times, spike_times, tolerance_s )

    if plot:
        plt.figure(figsize=(10,3))
        plt.plot(time_axis, neo_signal)
        plt.vlines(x=ann_spike_times, ymin=min(neo_signal), ymax=max(neo_signal), color='lightgreen')
        plt.scatter(spike_times, neo_signal[spike_indices], color='red', label='Detected Spikes')
        plt.hlines(y=[threshold], xmin=time_axis[0], xmax=time_axis[-1], color='k', alpha=0.5)
        plt.show()

        plt.figure(figsize=(10,3))
        plt.plot(time_axis, signal, label='Original Signal')
        plt.vlines(x=ann_spike_times, ymin=ymin, ymax=ymax, color='lightgreen')


        plt.scatter(spike_times, signal[spike_indices], color='red', label='Detected Spikes')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.title(f'F1:{f1*100:0.0f}%')
        plt.show()

    return f1
