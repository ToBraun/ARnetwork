# Copyright (C) 2025 by
# Tobias Braun

#------------------ PATHS ---------------------------#

# working directory
import sys
WDPATH = "/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/paper/scripts/ARnetlab"
sys.path.insert(0, WDPATH)
# input and output
INPUT_PATH = '/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/paper/data/'
OUTPUT_PATH = '/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/paper/figures/'


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
import h3 as h3


# my packages
import ARnet_sub as artn
import NETanalysis_sub as ana
import nullmodels_sub as model


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

def random_walker_paths(G_blank, traj_lengths, 
                        start_nodes=None, term_nodes=None, 
                        Nrealiz=1):
    """
    Generate an ensemble of random walks on a directed graph.

    Parameters
    ----------
    G_blank : networkx.DiGraph
        Base graph for transitions.
    traj_lengths : list[int]
        Desired trajectory lengths.
    start_nodes : list[str], optional
        Starting node IDs for each trajectory.
    term_nodes : list[str], optional
        Terminal node IDs for each trajectory.
    Nrealiz : int
        Number of realizations.

    Returns
    -------
    walks_ensemble : list[list[list[str]]]
        Outer list over realizations, then trajectories, then nodes in the trajectory.
    """
    walks_ensemble = []
    for r in range(Nrealiz):
        np.random.seed(r)
        walks = []
        for i, L in tqdm(enumerate(traj_lengths)):
            if start_nodes is not None:
                current_node = str(start_nodes[i])
            else:
                current_node = np.random.choice(list(G_blank.nodes()))

            walk = [current_node]
            for _ in range(L - 1):
                neighbors = list(G_blank.successors(current_node))
                if not neighbors:
                    break
                next_node = np.random.choice(neighbors)
                walk.append(next_node)
                current_node = next_node
            walks.append(walk)
        walks_ensemble.append(walks)
    return walks_ensemble


def highway_stats(
    observed_paths, G, 
    ebc_attr='edge_betweenness',
    threshold_quantile=0.9,
    min_length=4
):
    """
    Compute statistics of AR trajectories relative to high-betweenness 'highway' edges:
    (1) fraction of trajectories with at least n highway segments,
    (2) fraction of trajectories with at least n consecutive highway segments.

    Parameters
    ----------
    observed_paths : list of lists/arrays
        Each trajectory as a sequence of node IDs.
    G : networkx.Graph or DiGraph
        Graph with edge betweenness stored as an attribute.
    ebc_attr : str
        Name of edge betweenness attribute.
    threshold_quantile : float
        Quantile threshold to define 'highway' edges (top X%).
    min_length : int
        Minimum trajectory length (number of nodes) to include.

    Returns
    -------
    n_segments : np.ndarray
        Number of highway segments.
    frac_segments : np.ndarray
        Fraction of trajectories with at least that many highway segments.
    n_seq : np.ndarray
        Sequence length of consecutive highway segments.
    frac_seq : np.ndarray
        Fraction of trajectories with at least that many consecutive highway segments.
    """
    # 1. Determine highway edge threshold
    all_ebc = np.array([edata[ebc_attr] for _, _, edata in G.edges(data=True) if ebc_attr in edata])
    if all_ebc.size == 0:
        raise ValueError(f"Graph edges must have '{ebc_attr}' attribute.")
    highway_thresh = np.quantile(all_ebc, threshold_quantile)

    counts = []
    max_run_lengths = []
    n_traj = 0

    for path in tqdm(observed_paths):
        path = path.tolist() if isinstance(path, np.ndarray) else path
        # remove consecutive duplicates
        filtered_path = [path[0]]
        for u, v in zip(path[:-1], path[1:]):
            if u != v:
                filtered_path.append(v)

        if len(filtered_path) < min_length:
            continue

        n_traj += 1
        highway_count = 0
        run_length = 0
        max_run = 0

        for u, v in zip(filtered_path[:-1], filtered_path[1:]):
            try:
                ebc = G[u][v][ebc_attr]
                if ebc >= highway_thresh:
                    highway_count += 1
                    run_length += 1
                    max_run = max(max_run, run_length)
                else:
                    run_length = 0
            except KeyError:
                run_length = 0

        counts.append(highway_count)
        max_run_lengths.append(max_run)

    if n_traj == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # segment stats
    counts = np.array(counts)
    max_count = np.max(counts)
    n_segments = np.arange(1, max_count + 1)
    frac_segments = np.array([(counts >= n).sum() / len(counts) for n in n_segments])

    # sequence stats
    max_seq_len = max(max_run_lengths) if max_run_lengths else 0
    n_seq = np.arange(1, max_seq_len + 1)
    frac_seq = np.array([sum(run >= N for run in max_run_lengths) / n_traj for N in n_seq])

    return n_segments, frac_segments, n_seq, frac_seq

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
rndm_paths_pik = random_walker_paths(Gblank, a_traj_lengths_pikart, 
                        start_nodes=None, term_nodes=None, 
                        Nrealiz=Nrealiz)



# HIGHWAY STATS - REAL
n_segments_pik, frac_segments_pik, n_seq_pik, frac_seq_pik = highway_stats(
    observed_paths_pik, Greal, 
    ebc_attr='edge_betweenness',
    threshold_quantile=ebcthresh,
    min_length=Lmin
)


# HIGHWAY STATS - RANDOM
l_n_segments_rndm, l_frac_segments_rndm, l_n_seq_rndm, l_frac_seq_rndm = [], [], [], []
for n in tqdm(range(Nrealiz)):
    n_segments_rndm, frac_segments_rndm, n_seq_rndm, frac_seq_rndm = highway_stats(
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


# How high is the threshold?
all_ebc = np.array([edata['edge_betweenness'] for _, _, edata in Greal.edges(data=True) if 'edge_betweenness' in edata])
print(np.quantile(all_ebc, .9))


