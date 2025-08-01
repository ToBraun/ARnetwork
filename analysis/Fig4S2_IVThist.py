# Copyright (C) 2025 by
# Tobias Braun

#------------------ PATH ---------------------------#
# working directory
import sys
WDPATH = "/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/paper/scripts/ARnetlab"
sys.path.insert(0, WDPATH)
# input and output
INPUT_PATH = '/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/paper/data/'
OUTPUT_PATH = '/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/paper/suppl_figures/'


# %% IMPORT MODULES

# standard packages
import numpy as np
import pandas as pd
import matplotlib as mpl
#mpl.use('Agg')
from matplotlib import pyplot as plt
from scipy import stats


# specific packages


# my packages
import ARnet_sub as artn


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

# %% LOAD DATA

# PIKART
d_ars_pikart = pd.read_pickle(INPUT_PATH + 'PIKART' + '_hex.pkl')
# tARget v4
d_ars_target = pd.read_pickle(INPUT_PATH + 'target' + '_hex.pkl')

# we also import the untransformed one as it contains the lf_lons needed here (only)
d_ars_target_nohex = pd.read_pickle(INPUT_PATH + 'tARget_globalARcatalog_ERA5_1940-2023_v4.0_converted.pkl')
d_ars_target['lf_lon'] = d_ars_target_nohex['lf_lon']


# %% AR NETWORK


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
eps = int(2*ndec) # threshold: at least 2ARs/decade
thresh = 1.25*eps
# conditioning
cond = None # any network conditioning
LC_cond = None # lifecycle conditioning


# PIKART
ARcat = d_ars_pikart.copy()
ARcat['time'] = pd.to_datetime(ARcat['time']).dt.floor('D')
ARcat['lf_lon'] = ARcat['lf_lon'].replace(0, np.nan)
# Convert landfall latitudes and longitudes to hexagon index
l_arcats_pikart, d_coord_dict = artn.preprocess_catalog(ARcat, T, loc, grid_type, X, res, cond, LC_cond)
Apik, t_idx_pikart, t_hexidx_pikart, t_ivt_pikart, t_grid_pikart = artn.generate_transport_matrix(l_arcats_pikart, grid_type, d_coord_dict, LC_cond)
Gpikart = artn.generate_network(Apik, t_grid_pikart, weighted, directed, eps, self_links, weighing)

# tARget
ARcat = d_ars_target.copy()
ARcat['time'] = pd.to_datetime(ARcat['time']).dt.floor('D')
ARcat['lf_lon'] = ARcat['lf_lon'].replace(0, np.nan)
# Convert landfall latitudes and longitudes to hexagon index
l_arcats_target, d_coord_dict = artn.preprocess_catalog(ARcat, T, loc, grid_type, X, res, cond, LC_cond)
Atarget, t_idx_target, t_hexidx_target, t_ivt_target, t_grid_target = artn.generate_transport_matrix(l_arcats_target, grid_type, d_coord_dict, LC_cond)
Gtarget = artn.generate_network(Atarget, t_grid_target, weighted, directed, eps, self_links, weighing)


# %% MOISTURE TRANSPORT
# Assign quantiles of IVT changes to edges and nodes.


# Quantiles
qh1, qh2, qh3 = np.array([.6, .75, .90])
ql1, ql2, ql3 = 1 - qh1, 1 - qh2, 1 - qh3

# IVT differences
a_ivt_diffs1 = t_ivt_pikart[0][1] - t_ivt_pikart[0][0]
a_ivt_diffs2 = t_ivt_target[0][1] - t_ivt_target[0][0]
a_all_ivtdiffs = np.hstack([a_ivt_diffs1, a_ivt_diffs2])


# Thresholds
a_IVTthresholds = np.hstack([np.nanquantile(a_all_ivtdiffs, ql3),
                          np.nanquantile(a_all_ivtdiffs, ql2),
                          np.nanquantile(a_all_ivtdiffs, ql1),
                          np.nanquantile(a_all_ivtdiffs, qh1),
                          np.nanquantile(a_all_ivtdiffs, qh2),
                          np.nanquantile(a_all_ivtdiffs, qh3)])


# %% HISTOGRAM



# Derive gaussian kernel density estimates
# PIKART
kde_pik = stats.gaussian_kde(a_ivt_diffs1, bw_method='scott')  
xpik = np.linspace(min(a_ivt_diffs1), max(a_ivt_diffs1), 2000)
a_kde_pik = kde_pik(xpik)
# TARGET
kde_target = stats.gaussian_kde(a_ivt_diffs2, bw_method='scott')  
xtarget = np.linspace(min(a_ivt_diffs2), max(a_ivt_diffs2), 2000)
a_kde_target = kde_target(xtarget)

# colors
CMAP = ['#B22222', '#E66100', '#FDB863', 'grey', 'deepskyblue', 'dodgerblue', 'navy']

#%matplotlib

# Start plotting
fig = plt.figure(figsize=(6,6))
mpl.rcParams['font.size'] = 20
plt.xlabel('IVT differences (kg/ms)')
plt.ylabel('PDF')

# x-axis and color setup
xmin, xmax = -100, 100
thresholds = np.concatenate(([xmin], a_IVTthresholds, [xmax]))

# Plot KDE lines first to get the ylim
#plt.plot(xpik, a_kde_pik, label='PIKART', color='black', linewidth=2)
#plt.plot(xtarget, a_kde_target, label='tARget', color='slategrey', linewidth=2)

# Set log y-scale and limits
plt.yscale('log')
plt.xlim(xmin, xmax)
plt.ylim(1e-4, 5e-2)  # set this manually as needed

# Get the full y-limits from the axis now
ymin, ymax = plt.gca().get_ylim()

# Fill color bands behind the lines
for i in range(len(thresholds) - 1):
    plt.fill_betweenx([ymin, ymax],
                      thresholds[i], thresholds[i+1],
                      color=CMAP[i], alpha=0.3, zorder=0)

# Bring KDE lines to front again (optional but ensures visibility)
plt.plot(xpik, a_kde_pik, label='PIKART', color='black', linewidth=2, zorder=2)
plt.plot(xtarget, a_kde_target, label='tARget', color='slategrey', linewidth=2, zorder=2)

# Legend and save
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=16)
plt.savefig(OUTPUT_PATH + 'Fig4S2_IVThist.pdf', dpi=300, bbox_inches='tight')

