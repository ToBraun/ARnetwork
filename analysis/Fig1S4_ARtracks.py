# Copyright (C) 2025 by
# Tobias Braun

#------------------ PATHS ---------------------------#

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
from numpy.polynomial.polynomial import Polynomial
import pandas as pd
import matplotlib as mpl
#mpl.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as mplcm
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors
from matplotlib.colors import to_rgba, ListedColormap, Normalize, LogNorm
from matplotlib.collections import PolyCollection, LineCollection
import time
from scipy.stats import ks_2samp, mannwhitneyu, wasserstein_distance
import seaborn as sns
from scipy.interpolate import splprep, splev
from scipy.stats import linregress

# specific packages
import networkx as nx
from cmcrameri import cm
import cartopy.feature as cfeature
from tqdm import tqdm
import cartopy.crs as ccrs
import geopandas as gpd
from collections import defaultdict
import random
import itertools
import h3 as h3
from sklearn.metrics import mean_squared_error

# my packages
import ARnet_sub as artn
import NETanalysis_sub as ana
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

def remove_consecutive_duplicates(lons, lats):
    filtered = [(lons[0], lats[0])]
    for lon, lat in zip(lons[1:], lats[1:]):
        if (lon, lat) != filtered[-1]:
            filtered.append((lon, lat))
    return zip(*filtered)



def is_in_hemisphere(path, hemisphere='north'):
    coords = [h3.h3_to_geo(h) for h in path]
    lats, _ = zip(*coords)
    mean_lat = np.mean(lats)
    return mean_lat >= 15 if hemisphere == 'north' else mean_lat < -15


# %% LOAD DATA

# PIKART
d_ars_pikart = pd.read_pickle(INPUT_PATH + 'PIKART' + '_hex.pkl')
# tARget v4
d_ars_target = pd.read_pickle(INPUT_PATH + 'target' + '_hex.pkl')
# we also import the untransformed one as it contains the lf_lons needed here (only)
d_ars_target_nohex = pd.read_pickle(INPUT_PATH + 'tARget_globalARcatalog_ERA5_1940-2023_v4.0_converted.pkl')
d_ars_target['lf_lon'] = d_ars_target_nohex['lf_lon']



# %% CONDITIONAL AR NETWORK: SEASONAL

## Parameters
# spatiotemporal extent
T = None # no clipping
X = 'global'
# nodes
res = 2 # h3 system, corresponds to closest resolution to 2 degrees
grid_type = 'hexagonal'
loc = 'head'
# conditioning
cond = None # any network conditioning
LC_cond = None # lifecycle conditioning

# PIKART
ARcat = d_ars_pikart.copy()
ARcat['time'] = pd.to_datetime(ARcat['time']).dt.floor('D')
ARcat['lf_lon'] = ARcat['lf_lon'].replace(0, np.nan)
# Convert landfall latitudes and longitudes to hexagon index
l_arcats_pikart, d_coord_dict = artn.preprocess_catalog(ARcat, T, loc, grid_type, X, res, cond, LC_cond)

# tARget
ARcat = d_ars_target.copy()
ARcat['time'] = pd.to_datetime(ARcat['time']).dt.floor('D')
ARcat['lf_lon'] = ARcat['lf_lon'].replace(0, np.nan)
# Convert landfall latitudes and longitudes to hexagon index
l_arcats_target, d_coord_dict = artn.preprocess_catalog(ARcat, T, loc, grid_type, X, res, cond, LC_cond)

## EXTRACT tracks
# PIKART
l_artracks_pik = [group for name, group in l_arcats_pikart[0].groupby('trackid')]
observed_paths_pik = [l_artracks_pik[i].coord_idx.values for i in range(len(l_artracks_pik))]
# tARget
l_artracks_target = [group for name, group in l_arcats_target[0].groupby('trackid')]
observed_paths_target = [l_artracks_target[i].coord_idx.values for i in range(len(l_artracks_target))]


# %% MAP PLOT


# PIK random seed: 123
# tARget random seed: 321

# Parameters
l_alltracks = observed_paths_pik.copy() # select catalog
Nsmpl = 20
Lmin = 4 # minimum track length
SEEED = 123#123
l_cols = ["#003b76", "#ffab2f", "#ec48fa", "#6b9700", "#ac79ff", "#c89900", "#76006f",
          "#01864e", "#f80034", "#84b1ff", "#ff8f42", "#005757", "#de0053", "#eeb8c4",
          "#4b1303", "#fcade4", "#450f3c", "#ffa86d", "#910057", "#8b5f58"]

# Filter and sample valid indices
valid_indices = [i for i, path in enumerate(l_alltracks) if len(path) >= Lmin]
np.random.seed(SEEED)
a_indices = np.random.choice(valid_indices, Nsmpl, replace=False)

# FIGURE
%matplotlib 
proj = ccrs.EqualEarth(central_longitude=0)
fig, ax = plt.subplots(subplot_kw={'projection': proj}, figsize=(10, 10))
ax.set_global()
ax.add_feature(cfeature.COASTLINE, color='black')

# TRAJECTORIES
for plotted, idx in enumerate(a_indices):
    path = l_alltracks[idx]
    coords = [h3.h3_to_geo(h) for h in path]
    lats, lons = zip(*coords)

    if np.abs(lats[0]) >= 0:
        lons, lats = remove_consecutive_duplicates(lons, lats)

        if len(lons) < 3:
            continue  # Still skip if insufficient points after cleaning

        lons = np.unwrap(np.radians(lons))
        lons = np.degrees(lons)

        # No smoothing
        points = np.array([lons, lats]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        n_segments = len(segments)
        alphas = np.linspace(0.2, 1.0, n_segments)

        base_rgba = to_rgba(l_cols[plotted % len(l_cols)])
        colors = [(base_rgba[0], base_rgba[1], base_rgba[2], a) for a in alphas]

        lc = LineCollection(
            segments,
            colors=colors,
            linewidths=5,
            transform=ccrs.Geodetic(), zorder=5
        )
        ax.add_collection(lc)

plt.tight_layout()
plt.show()
plt.savefig(OUTPUT_PATH + "Fig1S4c.png", dpi=500, bbox_inches='tight')
