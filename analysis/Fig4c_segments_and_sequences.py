# Copyright (C) 2026 by
# Tobias Braun

#------------------ PATH ---------------------------#
# working directory
import sys
from pathlib import Path
import os

# Set root directory
REPO_ROOT = Path.cwd()
# Insert path to be able to find subroutines
sys.path.insert(0, str(REPO_ROOT))

# Set paths
INPUT_PATH  = Path(os.environ.get("ARNET_DATA",   REPO_ROOT / "data"))
OUTPUT_PATH = Path(os.environ.get("ARNET_FIGURES", REPO_ROOT / "figures"))


# %% IMPORT MODULES

# standard packages
import numpy as np
import pandas as pd
import matplotlib as mpl
#mpl.use('Agg')
from matplotlib import pyplot as plt

# specific packages
import networkx as nx
from tqdm import tqdm

# local subroutines
import ARnet_sub as artn
import NETanalysis_sub as ana
import NULLmodels_sub as model


# %% PLOT PARAMETERS
plt.style.use('default')
# Update Matplotlib parameters
colorbar_dir = 'horizontal'

# Change default tick direction
params = {'xtick.direction': 'in',
          'ytick.direction': 'in'}
plt.rcParams.update(params)
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['font.size'] = 20

# %% FUNCTIONS

# %% LOAD DATA

# PIKART
d_ars_pikart = pd.read_pickle(INPUT_PATH + 'PIKART' + '_hex.pkl')
# tARget v4
d_ars_target = pd.read_pickle(INPUT_PATH + 'target' + '_hex.pkl')
# we also import the untransformed one as it contains the lf_lons needed here (only)
d_ars_target_nohex = pd.read_pickle(INPUT_PATH + 'tARget_globalARcatalog_ERA5_1940-2023_v4.0_converted.pkl')
d_ars_target['lf_lon'] = d_ars_target_nohex['lf_lon']


# %% AR NETWORKS

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


   
# %% EDGE BETWEENNESS: EBC CONSENSUS
# Compute edge betweenness centrality to all edges & average for consensus network

# LOOP OVER CATALOGS
l_Gs = [Gpikart, Gtarget]
l_Gbetw_phases = []
for n in range(2):
    # Invert weights so that shortest paths correspond to maximum weight paths:
    G = ana.invert_weights(l_Gs[n])
    # EBC
    d_ebetw = nx.edge_betweenness_centrality(G, weight='weight')
    nx.set_edge_attributes(G, d_ebetw, "edge_betweenness")
    l_Gbetw_phases.append(G)

# Averaging of edge betweenness, edge weights and IVT classes:
Gcons0 = artn.average_networks_by_attributes(l_Gbetw_phases[0], l_Gbetw_phases[1], attr_name= "IVTdiff")
# Complete nodes for plotting
Gcons = artn.complete_nodes(Gcons0, res)



# %% PREDICTABILITY


# Parameters
Nrealiz = 200
ebcthresh = 0.9
Lmin = 4


# Real network & paths
Greal = Gcons.copy()
l_artracks_pik = [group for name, group in l_arcats_pikart[0].groupby('trackid')]
observed_paths_pik = [l_artracks_pik[i].coord_idx.values for i in range(len(l_artracks_pik))]

# Random walks
d_pik, d_target = l_arcats_pikart[0], l_arcats_target[0]
# Input parameters: properties that should be conserved by random networks 
a_traj_lengths_pikart = d_pik.groupby('trackid').size().values
a_traj_lengths_target = d_target.groupby('trackid').size().values
# Blank network: generate a fully connected network with real nodes and edge weights = 1
Gblank = model.build_hex_graph(res, dem=None, elevation_scaling=0.001)
Gblank = artn.complete_nodes(Gblank, res)
# Generate random walks
rndm_paths_pik = model.random_walker_paths(Gblank, a_traj_lengths_pikart, 
                        start_nodes=None, term_nodes=None, 
                        Nrealiz=Nrealiz)



# HIGHWAY STATS - REAL
n_segments_pik, frac_segments_pik, n_seq_pik, frac_seq_pik = ana.highway_stats(
    observed_paths_pik, Greal, 
    ebc_attr='edge_betweenness',
    threshold_quantile=ebcthresh,
    min_length=Lmin
)


# HIGHWAY STATS - RANDOM
l_n_segments_rndm, l_frac_segments_rndm, l_n_seq_rndm, l_frac_seq_rndm = [], [], [], []
for n in tqdm(range(Nrealiz)):
    n_segments_rndm, frac_segments_rndm, n_seq_rndm, frac_seq_rndm = ana.highway_stats(
        rndm_paths_pik[0], Greal, 
        ebc_attr='edge_betweenness',
        threshold_quantile=ebcthresh,
        min_length=Lmin
    )
    l_n_segments_rndm.append(n_segments_rndm)
    l_frac_segments_rndm.append(frac_segments_rndm)
    l_n_seq_rndm.append(n_seq_rndm)
    l_frac_seq_rndm.append(frac_seq_rndm)
    

# QUANTILES
alpha = .9999
## Convert to arrays
a_n_segments_rndm, a_frac_segments_rndm, a_n_seq_rndm, a_frac_seq_rndm = np.vstack(l_n_segments_rndm), np.vstack(l_frac_segments_rndm), np.vstack(l_n_seq_rndm), np.vstack(l_frac_seq_rndm)
## Compute quantiles
a_conf_seg = np.nanquantile(a_frac_segments_rndm, alpha, axis=0)
a_conf_seq = np.nanquantile(a_frac_seq_rndm, alpha, axis=0)


# %% FIGURE


fig, axes = plt.subplots(1, 2, figsize=(6, 3.3), sharey=True)

# --- Left panel: segments ---
seg_line, = axes[0].semilogy(n_segments_pik, frac_segments_pik, 
                             color='mediumpurple', marker='o', linewidth=3)
seg_conf, = axes[0].semilogy(a_n_segments_rndm[0,], a_conf_seg, 
                             color='plum', linestyle='dashed', linewidth=3)
axes[0].set_xlabel('segment length $n$', fontsize=16)
axes[0].set_ylabel('fraction of segments', fontsize=16)
axes[0].tick_params(axis='both', labelsize=14)

# --- Right panel: sequences ---
seq_line, = axes[1].semilogy(n_seq_pik, frac_seq_pik,  
                             color='darkcyan', marker='o', linewidth=3)
seq_conf, = axes[1].semilogy(a_n_seq_rndm[0,], a_conf_seq,  
                             color='skyblue', linestyle='dashed', linewidth=3)
axes[1].set_xlabel('sequence length $k$', fontsize=16)
axes[1].tick_params(axis='both', labelsize=14)

# --- Shared legend above both panels ---
fig.legend([seg_line, seq_line, seg_conf], 
           [r'$n$-segments', r'$k$-sequences', r'$\alpha=0.99$'],
           loc='lower center', bbox_to_anchor=(0.53, 0.95), 
           fontsize=15, ncol=3, frameon=False)

plt.tight_layout()
plt.savefig(OUTPUT_PATH + "Fig2c.png", dpi=600, bbox_inches='tight')
plt.show()



