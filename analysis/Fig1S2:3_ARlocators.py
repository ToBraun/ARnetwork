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
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt

# specific packages
import cartopy.crs as ccrs

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
d_ars_pikart = pd.read_pickle(INPUT_PATH + 'PIKART_hex.pkl')
d_ars_pikart['time'] = pd.to_datetime(d_ars_pikart['time'])

# tARget v4
d_ars_target = pd.read_pickle(INPUT_PATH + 'target_hex.pkl')
d_ars_target['time'] = pd.to_datetime(d_ars_target['time'])


# %% PARAMETERS

"""
Figure 1 S3: generate networks for AR heads.
"""

## Network parameters
# spatiotemporal extent
T = None # no clipping
X = 'global'
# nodes
res = 2 # h3 system, corresponds to closest resolution to 2 degrees
grid_type = 'hexagonal'
# edges
weighing = 'absolute'
self_links = False
weighted = True
directed = True
ndec = 8.4 # number of decades
eps = int(2*ndec) # threshold: at least 2ARs/decade
# conditioning
cond = None # any network conditioning
LC_cond = None # lifecycle conditioning


# %% PIKART

# Select catalog 
ARcat = d_ars_pikart.copy()


## HEAD 
loc = 'head'
# pre-processing
l_arcats_pikart, d_coord_dict = artn.preprocess_catalog(ARcat, T, loc, grid_type, X, res, cond, LC_cond)
# transport matrix
Apik, t_idx, t_hexidx, t_ivt, t_gridpik = artn.generate_transport_matrix(l_arcats_pikart, grid_type, d_coord_dict, LC_cond)
# generate network
Gpikart_head = artn.generate_network(Apik, t_gridpik, weighted, directed, eps, self_links, weighing)
# complete nodes, even those that are never visited (degree-0)
Gplot_pikart_head = artn.complete_nodes(Gpikart_head, res)

## CORE 
loc = 'core'
# pre-processing
l_arcats_pikart, d_coord_dict = artn.preprocess_catalog(ARcat, T, loc, grid_type, X, res, cond, LC_cond)
# transport matrix
Apik, t_idx, t_hexidx, t_ivt, t_gridpik = artn.generate_transport_matrix(l_arcats_pikart, grid_type, d_coord_dict, LC_cond)
# generate network
Gpikart_core = artn.generate_network(Apik, t_gridpik, weighted, directed, eps, self_links, weighing)
# complete nodes, even those that are never visited (degree-0)
Gplot_pikart_core = artn.complete_nodes(Gpikart_core, res)

# %% tARget-4

# Select catalog
ARcat = d_ars_target.copy()


## HEAD 
loc = 'head'
# pre-processing
l_arcats_target, d_coord_dict = artn.preprocess_catalog(ARcat, T, loc, grid_type, X, res, cond, LC_cond)
# transport matrix
Atarget, t_idx, t_hexidx, t_ivt, t_gridtarget = artn.generate_transport_matrix(l_arcats_target, grid_type, d_coord_dict, LC_cond)
# generate network
Gtarget_head = artn.generate_network(Atarget, t_gridtarget, weighted, directed, eps, self_links, weighing)
# complete nodes, even those that are never visited (degree-0)
Gplot_target_head = artn.complete_nodes(Gtarget_head, res)


# %% CONSENSUS NETWORKS

# HEAD 
thresh = int(1.25*eps)
Gcons_head = artn.consensus_network([Gplot_pikart_head, Gplot_target_head], thresh, eps)


# %% Fig 1 S2 - core


## PIKART core
nplot.plot_network(Gplot_pikart_core, widths='weights', colours='weights', layout='default', ndec=ndec, log=False,
                  arrowsize=0, linewidth=3, curvature=0.4, fontsize=14, ncolors=20, discard=180,
                  alpha=.5, show_nodes=True, proj = ccrs.EqualEarth(), show_axes=False, vmax=60*ndec)
plt.savefig(OUTPUT_PATH + "Fig1S2a.png", dpi=500, bbox_inches='tight')


# %% Fig 1 S3 - head

"""
Figure 1 S2: generate networks for AR core (only PIKART).
"""


## PIKART heads
nplot.plot_network(Gplot_pikart_head, widths='weights', colours='weights', layout='default', ndec=ndec, log=False,
                  arrowsize=0, linewidth=3, curvature=0.4, fontsize=14, ncolors=20, discard=180,
                  alpha=.5, show_nodes=True, proj = ccrs.EqualEarth(), show_axes=False)
plt.savefig(OUTPUT_PATH + "Fig1S3a.png", dpi=500, bbox_inches='tight')


### tARget heads
nplot.plot_network(Gplot_target_head, widths='weights', colours='weights', layout='default', ndec=ndec, log=False,
                  arrowsize=0, linewidth=3, curvature=0.4, fontsize=14, ncolors=20, discard=180,
                  alpha=.5, show_nodes=True, proj = ccrs.EqualEarth(), show_axes=False)
plt.savefig(OUTPUT_PATH + "Fig1S3b.png", dpi=500, bbox_inches='tight')


### Consensus heads
nplot.plot_network(Gcons_head, widths='weights', colours='weights', layout='default', ndec=ndec, log=False,
                  arrowsize=0, linewidth=3, curvature=0.4, fontsize=18, ncolors=20, discard=180,
                  alpha=.5, show_nodes=True, proj = ccrs.EqualEarth(),
                  show_axes=False)
plt.savefig(OUTPUT_PATH + "Fig1S3c.png", dpi=500, bbox_inches='tight')


