#In[]:
# Load the data
%reload_ext autoreload
%autoreload 2

from load import *
from ses_filter import *

SIN = False
raw, ds_out, _ = load_data(SIN)

#In[]:
# Run the SES filter
%matplotlib widget
%reload_ext autoreload
%autoreload 2

# Copy the delta sigma output not to modify the original values
ses_filtered = ds_out.copy()

if SIN:
    ses_gain_b  = 15
    ses_stages  = 4
    ses_w_n     = 16
else:
    ses_gain_b  = 17
    ses_stages  = 5     #
    ses_w_n     = 16    # log(Wo)

# Downsample (decimate) the data with a downsample ratio of <ses_downsample_r>
ses_downsample_r = 32

# pass the data through <ses_stages> SES filters
for ses_stage in range(ses_stages):
    if ses_stage == 0:  gain_b = ses_gain_b
    else:               gain_b = 0
    ses_filtered = ses_filter( ses_filtered, ses_w_n, gain_b )  # 16 samples window,


# Make a copy of the timeseries not to modify the originals
ses_filtered_normalized = ses_filtered.copy()
raw_normalized          = raw.copy()
ds_out_normalized       = ds_out.copy()

# Ignore the first few values and then normalize (just for visualization)
ses_filtered_normalized.data = ses_filtered_normalized.data [ses_w_n*20:]
raw_normalized         .data = raw_normalized         .data [ses_w_n*20:]
ds_out_normalized      .data = ds_out_normalized      .data [ses_w_n*20:]
ses_filtered_normalized.time = ses_filtered_normalized.time [ses_w_n*20:]
raw_normalized         .time = raw_normalized         .time [ses_w_n*20:]
ds_out_normalized      .time = ds_out_normalized      .time [ses_w_n*20:]

# Normalize the value of the SES filter output (just for visualization)
ses_filtered_normalized.data -= min(ses_filtered_normalized.data)
ses_filtered_normalized.data = ses_filtered_normalized.data/max(ses_filtered_normalized.data)

# Do the decimation --keep every other value--
ses_filtered_normalized.time = ses_filtered_normalized.time[::ses_downsample_r]
ses_filtered_normalized.data = ses_filtered_normalized.data[::ses_downsample_r]

import matplotlib.pyplot as plt

plt.figure(figsize=(10,3))
plt.step(ds_out_normalized.time, ds_out_normalized.data, alpha=0.3, linewidth=0.5, label=ds_out_normalized.name)
plt.plot(raw_normalized.time, raw_normalized.data, label=raw_normalized.name)
plt.plot(ses_filtered_normalized.time, ses_filtered_normalized.data, label=ses_filtered_normalized.name, alpha=0.5)

if SIN: plt.xlim(-0.0001, 0.001)
else:  plt.xlim(0.438, 0.443)

plt.ylim(-0.1,1.1)
plt.legend(loc='upper right')

plt.show()

