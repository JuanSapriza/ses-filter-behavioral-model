
from timeseries import *
from utils import *

def load_data(sin):
    time, raw_input, ds_output, ilpfd_output  = [], [], [], []

    if sin: filename = "sin_500Hz_85OSR_3dBFS"
    else:   filename = "iEEG_10000Hz_85OSR_65dBFS"

    with open( filename + ".csv", 'r') as f:
        import csv
        reader = csv.reader(f)
        for row in reader:
            # Append data from each row to the respective list
            time.           append(float(row[0]))
            raw_input.      append(float(row[1]))
            ds_output.      append(float(row[2]))
            ilpfd_output.   append(float(row[3]))


    # # Select the ranges of information you want to plot
    # start_s = 0.25
    # end_s   = 0.5

    # # Compute the parameters to be stored with the Timeseries
    # fs_Hz   = 1/time[1]
    # start_n = int( start_s   * fs_Hz )
    # end_n   = int( end_s     * fs_Hz )

    # # Crop the Timeseries
    # time            = time          [start_n : end_n]
    # raw_input       = raw_input     [start_n:end_n]
    # ds_output       = ds_output     [start_n:end_n]
    # ilpfd_output    = ilpfd_output  [start_n:end_n]

    # Create the Timeseries variables
    raw         = Timeseries( filename,             time = time, data = np.array(raw_input) )
    raw_norm    = Timeseries( filename,             time = time, data = normalize_data(np.array(raw_input)) )
    ds_out      = Timeseries( f"DS - {filename}",   time = time, data = normalize_data(np.array(ds_output)) )
    ilpfd       = Timeseries( f"ilpfd - {filename}",time = time, data = normalize_data(np.array(ilpfd_output)) )

    import re
    match = re.search(r'(\w+)_(\d+)Hz_(\d+)OSR_(\d+)dBFS', filename)

    if match:
        for x in [raw, ds_out, ilpfd]:
            x.params[TSP_TYPE]              = match.group(1)
            x.params[TSP_F_NYQUIST_HZ]      = float(match.group(2))  # Convert to float for numerical operations
            x.params[TSP_OSR]               = f"{int(match.group(3))}x"  # Convert to integer
            x.params[TSP_INPUT_LEVEL_DBFS]  = f"{-int(match.group(4))} dBFS"  # Convert to integer
            x.params[TSP_OG_AMPL_RANGE_V]   = [-1,1]

    ds_out.params[TSP_SAMPLE_B]     = 1
    ilpfd.params[TSP_SAMPLE_B]      = np.ceil(np.log2(max(ilpfd.data)-min(ilpfd.data)))
    ilpfd.params[TSP_SCORE_DR_BPS]  = ilpfd.params[TSP_SAMPLE_B]*ilpfd.params[TSP_F_HZ]
    align_signal_rmse(ilpfd, raw)
    diff_series(ilpfd)
    rmse = align_signal_rmse(ilpfd, raw)
    print(f"Ideal RMSE: {rmse:0.1f} dB")

    return raw_norm, ds_out, ilpfd