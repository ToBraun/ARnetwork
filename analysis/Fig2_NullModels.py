# Copyright (C) 2025 by
# Tobias Braun

#------------------ PATH ---------------------------#
# working directory
import sys
WDPATH = "/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/paper/scripts/ARnetlab"
sys.path.insert(0, WDPATH)
# input and output
INPUT_PATH = '/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/paper/data/'
OUTPUT_PATH = '/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/paper/data/random_graphs/'


# %% IMPORT MODULES

# standard packages
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt

# specific packages
import networkx as nx
from tqdm import tqdm
import cartopy.crs as ccrs

# my packages
import ARnet_sub as artn
import nullmodels as model

# %% PLOT PARAMETERS
plt.style.use('dark_background')
# Update Matplotlib parameters
colorbar_dir = 'horizontal'

# Change default tick direction
params = {'xtick.direction': 'in',
          'ytick.direction': 'in'}
plt.rcParams.update(params)
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['font.size'] = 16

# %% LOAD DATA

# Open pkl files with hexagonal coordinates
# PIKART
d_ars_pikart = pd.read_pickle(INPUT_PATH + 'PIKART_hex.pkl')
d_ars_pikart['time'] = pd.to_datetime(d_ars_pikart['time'])

# tARget v4
d_ars_target = pd.read_pickle(INPUT_PATH + 'target_hex.pkl')
d_ars_target['time'] = pd.to_datetime(d_ars_target['time'])


# %% REAL CATALOG

"""
Figure 2: random networks/null models of varying complexity.
"""

## Network parameters
# spatiotemporal extent
T = None # no clipping
X = 'global'
# nodes
res = 2 # h3 system, corresponds to closest resolution to 2 degrees
grid_type = 'hexagonal'
loc = 'head'
# edges
weighing = 'absolute'
self_links = False
weighted = True
directed = True
ndec = 8.4 # number of decades
eps = int(2*ndec) # threshold: at least 2ARs/decade
thresh = 1.25*eps # threshold for consensus network
# conditioning
cond = None # any network conditioning
LC_cond = None # lifecycle conditioning

# CATALOGs
## Pre-processing...
l_arcats_pikart, d_coord_dict = artn.preprocess_catalog(d_ars_pikart.copy(), T, loc, grid_type, X, res, cond)
l_arcats_target, d_coord_dict = artn.preprocess_catalog(d_ars_target.copy(), T, loc, grid_type, X, res, cond)
d_pik, d_target = l_arcats_pikart[0], l_arcats_target[0]


# Networks
Gpikart = nx.read_gexf(INPUT_PATH + "arnet_pikart_centroid.gexf")
Gtarget = nx.read_gexf(INPUT_PATH + "arnet_target_centroid.gexf")
Gcons = artn.consensus_network([Gpikart, Gtarget], thresh, eps)


# %% RANDOM NETWORKS

# ETA
##(200*100*2)/3600 + (19*100*2)/3600 + (19*100*2)/3600 + (4*100*2)/3600
## 27h for two hundred realizations

# Number of realizations
Nrealiz = 2
maxdist = 3

# Blank network: generate a fully connected network with real nodes and edge weights = 1
Gblank = model.build_hex_graph(res, dem=None, elevation_scaling=0.001)
Gblank = artn.complete_nodes(Gblank, res)

# Input parameters: properties that should be conserved by random networks 
a_traj_lengths_pikart = d_pik.groupby('trackid').size().values
a_traj_lengths_target = d_target.groupby('trackid').size().values
a_start_nodes_pikart = d_pik.groupby('trackid').coord_idx.first().values
a_start_nodes_target = d_target.groupby('trackid').coord_idx.first().values
a_term_nodes_pikart = d_pik.groupby('trackid').coord_idx.last().values
a_term_nodes_target = d_target.groupby('trackid').coord_idx.last().values


# Generate random walks & networks
## 1) RANDOM
l_Gwalk_rndm_pikart = model.random_walker_ensemble(Gblank, a_traj_lengths_pikart, start_nodes=None, term_nodes=None, Nrealiz=Nrealiz)
l_Gwalk_rndm_target = model.random_walker_ensemble(Gblank, a_traj_lengths_target, start_nodes=None, term_nodes=None, Nrealiz=Nrealiz)
l_Gwalk_rndm_cons = [artn.consensus_network([l_Gwalk_rndm_pikart[n], l_Gwalk_rndm_target[n]], 0, 0) for n in tqdm(range(Nrealiz))]

## 2) GENESIS-CONSTRAINED
l_Gwalk_genesis_pikart = model.random_walker_ensemble(Gblank, a_traj_lengths_pikart, start_nodes=a_start_nodes_pikart, term_nodes=None, Nrealiz=Nrealiz)
l_Gwalk_genesis_target = model.random_walker_ensemble(Gblank, a_traj_lengths_target, start_nodes=a_start_nodes_target, term_nodes=None, Nrealiz=Nrealiz)
l_Gwalk_genesis_cons = [artn.consensus_network([l_Gwalk_genesis_pikart[n], l_Gwalk_genesis_target[n]], 0, 0) for n in tqdm(range(Nrealiz))]

## 3) TERMINATION-CONSTRAINED
l_Gwalk_term_pikart = model.random_walker_ensemble(Gblank, a_traj_lengths_pikart, start_nodes=None, term_nodes=a_term_nodes_pikart, Nrealiz=Nrealiz)
l_Gwalk_term_target = model.random_walker_ensemble(Gblank, a_traj_lengths_target, start_nodes=None, term_nodes=a_term_nodes_target, Nrealiz=Nrealiz)
l_Gwalk_term_cons = [artn.consensus_network([l_Gwalk_term_pikart[n], l_Gwalk_term_target[n]], 0, 0) for n in tqdm(range(Nrealiz))]

## 4) REWIRED
l_Grewired_pikart = model.rewire_edges(Gpikart, Nrealiz, maxdist)
l_Grewired_target = model.rewire_edges(Gtarget, Nrealiz, maxdist)
l_Grewired_cons = model.rewire_edges(Gcons, Nrealiz, maxdist)


import numpy as np

len(list(nx.get_node_attributes(l_Gwalk_rndm_pikart[0], "coordID").values()))
len(np.unique(list(nx.get_node_attributes(l_Gwalk_rndm_pikart[0], "coordID").values())))





plot.plot_network(l_Grewired_cons[0], widths='weights', colours='weights', layout='default', ndec=ndec, log=False,
                  arrowsize=0, linewidth=4, curvature=0.4, fontsize=14, ncolors=20, discard=180,
                  alpha=.5, show_nodes=True, proj = ccrs.EqualEarth(),
                  show_axes=False)


# %% SAVE


# PIKART
for n in range(Nrealiz):
   nx.write_gml(l_Gwalk_rndm_pikart[n], OUTPUT_PATH + 'l_Gwalk_rndm_' + str(n) + '_pik_' + loc + '.gml')
   nx.write_gml(l_Gwalk_genesis_pikart[n], OUTPUT_PATH + 'l_Gwalk_genesis_' + str(n) + '_pik_' + loc + '.gml')
   nx.write_gml(l_Gwalk_term_pikart[n], OUTPUT_PATH + 'l_Gwalk_term_' + str(n) + '_pik_' + loc + '.gml')
   rewired_graph = nx.relabel_nodes(l_Grewired_pikart[n], str)
   nx.write_gml(rewired_graph, OUTPUT_PATH + 'l_G_rewired_' + str(n) + '_pik_' + loc + '.gml')


# tARget
for n in range(Nrealiz):
   nx.write_gml(l_Gwalk_rndm_target[n], OUTPUT_PATH + 'l_Gwalk_rndm_' + str(n) + '_target_' + loc + '.gml')
   nx.write_gml(l_Gwalk_genesis_target[n], OUTPUT_PATH + 'l_Gwalk_genesis_' + str(n) + '_target_' + loc + '.gml')
   nx.write_gml(l_Gwalk_term_target[n], OUTPUT_PATH + 'l_Gwalk_term_' + str(n) + '_target_' + loc + '.gml')
   rewired_graph = nx.relabel_nodes(l_Grewired_target[n], str)
   nx.write_gml(rewired_graph, OUTPUT_PATH + 'l_G_rewired_' + str(n) + '_target_' + loc + '.gml')


# Consensus
for n in range(Nrealiz):
    nx.write_gml(l_Gwalk_rndm_cons[n], OUTPUT_PATH + 'l_Gwalk_rndm_' + str(n) + '._cons_' + loc + '.gml')
    nx.write_gml(l_Gwalk_genesis_cons[n], OUTPUT_PATH + 'l_Gwalk_genesis_' + str(n) + '_cons_' + loc + '.gml')
    nx.write_gml(l_Gwalk_term_cons[n], OUTPUT_PATH + 'l_Gwalk_term_' + str(n) + '_cons_' + loc + '.gml')
    rewired_graph = nx.relabel_nodes(l_Grewired_cons[n], str)
    nx.write_gml(rewired_graph, OUTPUT_PATH + 'l_G_rewired_' + str(n) + '_cons_' + loc + '.gml')

