# Copyright (C) 2025 by
# Tobias Braun

#------------------ PATH ---------------------------#
# working directory
import sys
WDPATH = "/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/paper/scripts/ARnetlab"
sys.path.insert(0, WDPATH)
# input and output
INPUT_PATH = '/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/paper/data/'
INDICES_PATH = '/Users/tbraun/Desktop/data/Global_Climate_Indices/'
OUTPUT_PATH = '/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/paper/suppl_figures/'


# %% IMPORT MODULES
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt


# %% PLOT PARAMETERS
plt.style.use('default')
# Update Matplotlib parameters
colorbar_dir = 'horizontal'

# Change default tick direction
params = {'xtick.direction': 'in',
          'ytick.direction': 'in'}
plt.rcParams.update(params)
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['font.size'] = 16

# %% FUNCTIONS

# Function to classify phases with sustained conditions
def classify_sustained_phases(index_series, threshold, min_periods, phase_value, zscore = False):
    if zscore:
        index_series = (index_series - np.nanmean(index_series))/np.std(index_series)
    # Find consecutive periods meeting the threshold
    condition_met = index_series > threshold if phase_value == 1 else index_series < threshold
    condition_met = condition_met.astype(int)
    
    # Identify consecutive runs
    sustained_periods = (condition_met.groupby((condition_met != condition_met.shift()).cumsum())
                         .cumsum() >= min_periods)
    return sustained_periods * phase_value

# %% LOAD DATA

# PIKART
d_ars_pikart = pd.read_pickle(INPUT_PATH + 'PIKART' + '_hex.pkl')
# tARget v4
d_ars_target = pd.read_pickle(INPUT_PATH + 'target' + '_hex.pkl')
# we also import the untransformed one as it contains the lf_lons needed here (only)
d_ars_target_nohex = pd.read_pickle(INPUT_PATH + 'tARget_globalARcatalog_ERA5_1940-2023_v4.0_converted.pkl')
d_ars_target['lf_lon'] = d_ars_target_nohex['lf_lon']

# ENSO
d_enso = pd.read_csv(INDICES_PATH + 'ONI.txt', delim_whitespace=True, header=None, dtype=str, skiprows=1, engine='python')
d_enso = d_enso[d_enso.apply(lambda row: not row.str.contains('-99.90').any(), axis=1)]


# %% ENSO pre-processing

# We set ENSO as the oscillation index dataframe
d_oscidx = d_enso.copy()

# Parameters for event detection
WIN = 3 #months
thresh1, thresh2 = 0.5, -0.5 # amplitude threshold
min_consecutive_periods = 5 #months
zscore = True

# TRANSFORM IT TO MAKE IT COMPATIBLE WITH CATALOG FORMAT
# Convert remaining valid data to float
d_oscidx = d_oscidx.astype(float)
# Assign column names for readability
d_oscidx.columns = ['Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
# Reshape data to long format with a Date column for easier time-based processing
d_oscidx_long = d_oscidx.melt(id_vars='Year', var_name='Month', value_name='COidx')
d_oscidx_long['Month'] = pd.to_datetime(d_oscidx_long['Month'], format='%b').dt.month
d_oscidx_long['Date'] = pd.to_datetime(d_oscidx_long[['Year', 'Month']].assign(Day=1))
d_oscidx_long = d_oscidx_long.set_index('Date').sort_index()

# Calculate 3-month rolling averages to simulate overlapping three-month periods
d_oscidx_long['COidx_3month_avg'] = d_oscidx_long['COidx'].rolling(window=WIN, min_periods=3).mean()

# Initialize the 'teleconnection' column with NaNs
d_oscidx_long['teleconnection'] = 0  # Start with Neutral (0) for all periods

# Classify two regimes: 1: El Nino, -1: La Nina
d_oscidx_long['teleconnection'] = classify_sustained_phases(d_oscidx_long['COidx_3month_avg'], thresh1, min_consecutive_periods, 1, zscore=zscore)
d_oscidx_long['teleconnection'] += classify_sustained_phases(d_oscidx_long['COidx_3month_avg'], thresh2, min_consecutive_periods, -1, zscore=zscore)

# Set any remaining periods as Neutral (0)
d_oscidx_long['teleconnection'] = d_oscidx_long['teleconnection'].replace({np.nan: 0}).astype(int)
d_oscidx = d_oscidx_long.copy()



# %% FIGURE

# Merge with ARcat DataFrame based on time
ARcat = d_ars_pikart.copy()
# Resample to daily res
d_oscidx_daily = d_oscidx['teleconnection'].resample('D').ffill()
# Merge
ARcat['time'] = pd.to_datetime(ARcat['time']).dt.floor('D')
ARcat = ARcat.merge(d_oscidx_daily.rename('teleconnection'), left_on='time', right_index=True, how='left')


fig, ax = plt.subplots(figsize=(6, 3))
plt.axhline(color='grey')
# Plot the 3-month average index
ax.plot(d_oscidx_long.index, d_oscidx_long['COidx_3month_avg'], 
        label='3-Month Avg', color='black', linewidth=2)

# Classify points
el_nino = d_oscidx_long[d_oscidx_long['teleconnection'] == 1]
la_nina = d_oscidx_long[d_oscidx_long['teleconnection'] == -1]
neutral = d_oscidx_long[d_oscidx_long['teleconnection'] == 0]

# Add classified scatter points
ax.scatter(el_nino.index, el_nino['COidx_3month_avg'], 
           color='indianred', label='El Niño', marker='o', s=50, linewidth=0.5)
ax.scatter(la_nina.index, la_nina['COidx_3month_avg'], 
           color='royalblue', label='La Niña', marker='o', s=50, linewidth=0.5)
ax.scatter(neutral.index, neutral['COidx_3month_avg'], 
           color='grey', label='Neutral', marker='o', s=50, linewidth=0.5)

# Axis labels and title
ax.set_xlabel('year', fontsize=12)
ax.set_ylabel('ONI (3-month average)', fontsize=12)

# Ticks
ax.tick_params(axis='both', labelsize=10)

# Legend
ax.legend(loc='upper left', fontsize=10, frameon=True, edgecolor='gray')

# Tight layout
plt.tight_layout()
plt.savefig(OUTPUT_PATH + "Fig4S4_ONI.pdf", dpi=300, bbox_inches='tight')


