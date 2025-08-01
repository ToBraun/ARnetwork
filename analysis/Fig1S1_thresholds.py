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
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import time

# specific packages
import networkx as nx
from cmcrameri import cm
import cartopy.feature as cfeature
from tqdm import tqdm
import cartopy.crs as ccrs
import geopandas as gpd


# my packages
import ARnet_sub as artn
import NETplots_sub as nplot


# %% PLOT PARAMETERS
plt.style.use('dark_background')
# Update Matplotlib parameters
colorbar_dir = 'horizontal'

# Change default tick direction
params = {'xtick.direction': 'in',
          'ytick.direction': 'in'}
plt.rcParams.update(params)
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['font.size'] = 18

# %% LOAD DATA

# PIKART
d_ars_pikart = pd.read_pickle(INPUT_PATH + 'PIKART' + '_hex.pkl')
# tARget v4
d_ars_target = pd.read_pickle(INPUT_PATH + 'target' + '_hex.pkl')


# %% PARAMETERS

"""
Figure 1 S1: generate networks for different thresholds.
"""

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
ndec = 8.4
eps = 0#84#0
# conditioning
cond = None # any network conditioning
LC_cond = None # lifecycle conditioning


# %% NETWORKS

# PIKART 
ARcat = d_ars_pikart.copy()

## Generate AR network with given threshold 
ARcat = d_ars_pikart.copy()
l_arcats_pikart, d_coord_dict = artn.preprocess_catalog(ARcat, T, loc, grid_type, X, res, cond, LC_cond)
Apik, t_idx, t_hexidx, t_ivt, t_gridpik = artn.generate_transport_matrix(l_arcats_pikart, grid_type, d_coord_dict, LC_cond)
Gpikart = artn.generate_network(Apik, t_gridpik, weighted, directed, eps, self_links, weighing)
Gplot_pikart = artn.complete_nodes(Gpikart, res)

### PLOT
nplot.plot_network(Gplot_pikart, widths='weights', colours='weights', layout='default', ndec=ndec, log=False,
                  arrowsize=0, linewidth=3, curvature=0.4, fontsize=14, ncolors=20, discard=180,
                  alpha=.5, show_nodes=True, proj = ccrs.EqualEarth(), show_axes=False, vmax=68*ndec)
plt.savefig(OUTPUT_PATH + "Fig1S1a.png", dpi=300, bbox_inches='tight')



# tARget
ARcat = d_ars_target.copy()

## Generate AR network with given threshold 
loc = 'centroid'
l_arcats_target, d_coord_dict = artn.preprocess_catalog(ARcat, T, loc, grid_type, X, res, cond, LC_cond)
Atarget, t_idx, t_hexidx, t_ivt, t_gridtarget = artn.generate_transport_matrix(l_arcats_target, grid_type, d_coord_dict, LC_cond)
Gtarget = artn.generate_network(Atarget, t_gridtarget, weighted, directed, eps, self_links, weighing)
Gplot_target = artn.complete_nodes(Gtarget, res)

### PLOT
%matplotlib
nplot.plot_network(Gplot_target, widths='weights', colours='weights', layout='default', ndec=ndec, log=False,
                  arrowsize=0, linewidth=3, curvature=0.4, fontsize=14, ncolors=20, discard=180,
                  alpha=.5, show_nodes=True, proj = ccrs.EqualEarth(), show_axes=False, vmax=37*ndec)
plt.savefig(OUTPUT_PATH + "Fig1S1c.png", dpi=300, bbox_inches='tight')


