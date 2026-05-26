# Copyright (C) 2025 by
# Tobias Braun

#------------------ PATHS ---------------------------#

# working directory
import sys
WDPATH = "/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/paper/scripts/ARnetlab"
sys.path.insert(0, WDPATH)
# input and output
INPUT_PATH = '/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/paper/data/'
OUTPUT_PATH = '/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/paper/data/'


# %% IMPORT MODULES

# standard packages
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt

# specific packages
import networkx as nx

# my packages
import ARnet_sub as artn


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

# Open pkl files with hexagonal coordinates
# PIKART
d_ars_pikart = pd.read_pickle(INPUT_PATH + 'PIKART_hex.pkl')
d_ars_pikart['time'] = pd.to_datetime(d_ars_pikart['time'])

# tARget v4
d_ars_target = pd.read_pickle(INPUT_PATH + 'target_hex.pkl')
d_ars_target['time'] = pd.to_datetime(d_ars_target['time'])


# %% PARAMETERS

"""
Figure 1: generate networks for different AR catalogs.
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

## CENTROID 
loc = 'centroid'
ARcat = d_ars_pikart.copy()
# pre-processing
l_arcats_pikart, d_coord_dict = artn.preprocess_catalog(ARcat, T, loc, grid_type, X, res, cond, LC_cond)
# transport matrix
Apik, t_idx, t_hexidx, t_ivt, t_gridpik = artn.generate_transport_matrix(l_arcats_pikart, grid_type, d_coord_dict, LC_cond)
# generate network
Gpikart = artn.generate_network(Apik, t_gridpik, weighted, directed, eps, self_links, weighing)
# complete nodes, even those that are never visited (degree-0)
Gplot_pikart = artn.complete_nodes(Gpikart, res)


# %% tARget-4

# Select catalog
ARcat = d_ars_target.copy()


## CENTROID 
loc = 'centroid'
# pre-processing
l_arcats_target, d_coord_dict = artn.preprocess_catalog(ARcat, T, loc, grid_type, X, res, cond, LC_cond)
# transport matrix
Atarget, t_idx, t_hexidx, t_ivt, t_gridtarget = artn.generate_transport_matrix(l_arcats_target, grid_type, d_coord_dict, LC_cond)
# generate network
Gtarget = artn.generate_network(Atarget, t_gridtarget, weighted, directed, eps, self_links, weighing)
# complete nodes, even those that are never visited (degree-0)
Gplot_target = artn.complete_nodes(Gtarget, res)


# %% CONSENSUS NETWORK

# Compute consensus network
thresh = int(1.25*eps)
Gplot_cons = artn.consensus_network([Gplot_pikart, Gplot_target], thresh, eps)



# %% SAVE OUTPUT

nx.write_gexf(Gplot_pikart, OUTPUT_PATH + "arnet_pikart_centroid.gexf")
nx.write_gexf(Gplot_target, OUTPUT_PATH + "arnet_target_centroid.gexf")
nx.write_gexf(Gplot_cons, OUTPUT_PATH + "arnet_consensus_centroid.gexf")
