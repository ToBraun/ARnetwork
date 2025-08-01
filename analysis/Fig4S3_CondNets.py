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

# standard packages
import numpy as np
import pandas as pd
import matplotlib as mpl
#mpl.use('Agg')
from matplotlib import pyplot as plt

# specific packages
import cartopy.crs as ccrs

# my packages
import ARnet_sub as artn
import NETplots_sub as nplot


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



# %% PARAMETERS

# Merge with ARcat DataFrame based on time
ARcat = d_ars_pikart.copy()
# Resample to daily res
d_oscidx_daily = d_oscidx['teleconnection'].resample('D').ffill()
# Merge
ARcat['time'] = pd.to_datetime(ARcat['time']).dt.floor('D')
ARcat = ARcat.merge(d_oscidx_daily.rename('teleconnection'), left_on='time', right_index=True, how='left')



# %% CONDITIONAL AR NETWORKS

"""
SEASONS order: winter, summer
ENSO order: La Nina, neutral, El Nino
"""

# PICK THE INDEX
cond = 'teleconnection'

if cond == 'teleconnection':
    l_eps = np.array([3, 9, 3])
elif cond == 'season':
    l_eps = np.array([5,5,5,5])

## Network parameters
# spatiotemporal extent
T = None # no clipping
X = 'global'
# nodes
res = 2 # h3 system, corresponds to closest resolution to 2 degrees
grid_type = 'hexagonal'
loc = 'centroid'
# edges
weighing = 'absolute'
self_links = False
weighted = True
directed = True
ndec = 8.4 # number of decades
eps = l_eps
thresh = np.array(1.25*eps, dtype=int)
# conditioning
LC_cond = None # lifecycle conditioning


# PIKART
ARcat = d_ars_pikart.copy()
ARcat['time'] = pd.to_datetime(ARcat['time']).dt.floor('D')
if cond != 'season':
    d_oscidx_daily = d_oscidx['teleconnection'].resample('D').ffill()
    ARcat = ARcat.merge(d_oscidx_daily.rename('teleconnection'), left_on='time', right_index=True, how='left')
ARcat['lf_lon'] = ARcat['lf_lon'].replace(0, np.nan)
# Convert landfall latitudes and longitudes to hexagon index
l_arcats_pikart, d_coord_dict = artn.preprocess_catalog(ARcat, T, loc, grid_type, X, res, cond, LC_cond)
Apik, t_idx_pikart, t_hexidx_pikart, t_ivt_pikart, t_grid_pikart = artn.generate_transport_matrix(l_arcats_pikart, grid_type, d_coord_dict, LC_cond)
Gpikart = artn.generate_network(Apik, t_grid_pikart, weighted, directed, eps, self_links, weighing)
# SEASONAL: only look at DJF and JJA
if cond == 'season':
    ns1, ns2 = 2, 3
    Gpikart, t_idx_pikart, t_hexidx_pikart, t_ivt_pikart = [Gpikart[ns1], Gpikart[ns2]], [t_idx_pikart[ns1], t_idx_pikart[ns2]], [t_hexidx_pikart[ns1], t_hexidx_pikart[ns2]], [t_ivt_pikart[ns1], t_ivt_pikart[ns2]]

# tARget
ARcat = d_ars_target.copy()
ARcat['time'] = pd.to_datetime(ARcat['time']).dt.floor('D')
if cond != 'season':
    d_oscidx_daily = d_oscidx['teleconnection'].resample('D').ffill()
    ARcat = ARcat.merge(d_oscidx_daily.rename('teleconnection'), left_on='time', right_index=True, how='left')
ARcat['lf_lon'] = ARcat['lf_lon'].replace(0, np.nan)
# Convert landfall latitudes and longitudes to hexagon index
l_arcats_target, d_coord_dict = artn.preprocess_catalog(ARcat, T, loc, grid_type, X, res, cond, LC_cond)
Atarget, t_idx_target, t_hexidx_target, t_ivt_target, t_grid_target = artn.generate_transport_matrix(l_arcats_target, grid_type, d_coord_dict, LC_cond)
Gtarget = artn.generate_network(Atarget, t_grid_target, weighted, directed, eps, self_links, weighing)
# SEASONAL: only look at DJF and JJA
if cond == 'season':
    Gtarget, t_idx_target, t_hexidx_target, t_ivt_target = [Gtarget[ns1], Gtarget[ns2]], [t_idx_target[ns1], t_idx_target[ns2]], [t_hexidx_target[ns1], t_hexidx_target[ns2]], [t_ivt_target[ns1], t_ivt_target[ns2]]
Lp = len(Gtarget)

# Check: did we set the thresholds right?
print(Gpikart[0].number_of_edges())
print(Gpikart[1].number_of_edges())
print(Gpikart[2].number_of_edges())


# %% SEASONS

# Consensus networks for seasons 
Gcons_djf = artn.consensus_network([Gpikart[0], Gtarget[0]], thresh[0], eps[0])
Gcons_jja = artn.consensus_network([Gpikart[1], Gtarget[1]], thresh[0], eps[0])

### DJF
nplot.plot_network(Gcons_djf, widths='weights', colours='weights', layout='default', ndec=ndec, log=False,
                  arrowsize=0, linewidth=3, curvature=0.4, fontsize=14, ncolors=20, discard=180,
                  alpha=.5, show_nodes=True, proj = ccrs.EqualEarth(), show_axes=False)
plt.savefig(OUTPUT_PATH + "Fig4S3a.png", dpi=300, bbox_inches='tight')


### JJA
nplot.plot_network(Gcons_jja, widths='weights', colours='weights', layout='default', ndec=ndec, log=False,
                  arrowsize=0, linewidth=3, curvature=0.4, fontsize=14, ncolors=20, discard=180,
                  alpha=.5, show_nodes=True, proj = ccrs.EqualEarth(), show_axes=False)
plt.savefig(OUTPUT_PATH + "Fig4S3b.png", dpi=300, bbox_inches='tight')


# %% ENSO

# Consensus networks for seasons 
Gcons_nina = artn.consensus_network([Gpikart[0], Gtarget[0]], thresh[0], eps[0])
Gcons_nino = artn.consensus_network([Gpikart[2], Gtarget[2]], thresh[2], eps[2])

### La Niña
nplot.plot_network(Gcons_nina, widths='weights', colours='weights', layout='default', ndec=ndec, log=False,
                  arrowsize=0, linewidth=3, curvature=0.4, fontsize=14, ncolors=20, discard=180,
                  alpha=.5, show_nodes=True, proj = ccrs.EqualEarth(), show_axes=False)
plt.savefig(OUTPUT_PATH + "Fig4S3c.png", dpi=300, bbox_inches='tight')


### El Niño
nplot.plot_network(Gcons_nino, widths='weights', colours='weights', layout='default', ndec=ndec, log=False,
                  arrowsize=0, linewidth=3, curvature=0.4, fontsize=14, ncolors=20, discard=180,
                  alpha=.5, show_nodes=True, proj = ccrs.EqualEarth(), show_axes=False)
plt.savefig(OUTPUT_PATH + "Fig4S3d.png", dpi=300, bbox_inches='tight')
