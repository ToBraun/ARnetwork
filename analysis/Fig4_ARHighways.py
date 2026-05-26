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
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize, LogNorm, ListedColormap
from matplotlib.cm import ScalarMappable
from tqdm import tqdm

import networkx as nx
import cartopy.crs as ccrs

# local subroutines
import ARnet_sub as artn
import NETanalysis_sub as ana
import NETplots_sub as nplot

# %% PLOT PARAMETERS
plt.style.use('default')
plt.rcParams.update({'xtick.direction': 'in', 'ytick.direction': 'in'})
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['font.size'] = 16


# %%FUNCTIONS

def build_network(ARcat):
    ARcat = ARcat.copy()
    ARcat['time'] = pd.to_datetime(ARcat['time']).dt.floor('D')
    ARcat['lf_lon'] = ARcat['lf_lon'].replace(0, np.nan)
    l_arcats, d_coord_dict = artn.preprocess_catalog(
        ARcat, T, loc, grid_type, X, res, cond, LC_cond
    )
    A, t_idx, t_hexidx, t_ivt, t_grid = artn.generate_transport_matrix(
        l_arcats, grid_type, d_coord_dict, LC_cond
    )
    G = artn.generate_network(A, t_grid, weighted, directed, eps, self_links, weighing)
    return G, t_hexidx, t_ivt

# %% LOAD DATA

d_ars_pikart = pd.read_pickle(INPUT_PATH / 'PIKART_hex.pkl')
d_ars_target = pd.read_pickle(INPUT_PATH / 'target_hex.pkl')
d_ars_target_nohex = pd.read_pickle(
    '/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/data/'
    'tARget_globalARcatalog_ERA5_1940-2023_v4.0_converted_lf.pkl'
)
d_ars_target['lf_lon'] = d_ars_target_nohex['lf_lon']
d_ars_target['lf_lat'] = d_ars_target_nohex['lf_lat']


# %% AR NETWORK

# Network parameters
T = None
X = 'global'
res = 2                  # H3 resolution ~ 2 degrees
grid_type = 'hexagonal'
loc = 'centroid'
weighing = 'absolute'
self_links = False
weighted = True
directed = True
ndec = 8.4
eps = int(2 * ndec)      # ≥ 2 ARs / decade
thresh = 1.25 * eps      # kept for parity with original (unused below)
cond = None
LC_cond = None

# ARTNs
Gpikart, t_hexidx_pikart, t_ivt_pikart = build_network(d_ars_pikart)
Gtarget, t_hexidx_target, t_ivt_target = build_network(d_ars_target)


# %% MOISTURE TRANSPORT — assign IVTdiff classes to EDGES and NODES

qh1, qh2, qh3 = 0.60, 0.75, 0.90
ql1, ql2, ql3 = 1 - qh1, 1 - qh2, 1 - qh3

a_ivt_diffs_pikart = t_ivt_pikart[0][1] - t_ivt_pikart[0][0]
a_ivt_diffs_target = t_ivt_target[0][1] - t_ivt_target[0][0]
a_all_ivtdiffs = np.hstack([a_ivt_diffs_pikart, a_ivt_diffs_target])

a_IVTthresholds = np.hstack([
    np.nanquantile(a_all_ivtdiffs, ql3),
    np.nanquantile(a_all_ivtdiffs, ql2),
    np.nanquantile(a_all_ivtdiffs, ql1),
    np.nanquantile(a_all_ivtdiffs, qh1),
    np.nanquantile(a_all_ivtdiffs, qh2),
    np.nanquantile(a_all_ivtdiffs, qh3),
])

# EDGES
tmp_edgesigns_pikart = ana.compute_edge_moisture_transport(
    t_hexidx_pikart[0], t_ivt_pikart[0],
    output='manual', thresholds=a_IVTthresholds,
)
G_esigned_pikart = artn.add_edge_attr_to_graph(
    Gpikart, tmp_edgesigns_pikart, attr_name='IVTdiff'
)

tmp_edgesigns_target = ana.compute_edge_moisture_transport(
    t_hexidx_target[0], t_ivt_target[0],
    output='manual', thresholds=a_IVTthresholds,
)
G_esigned_target = artn.add_edge_attr_to_graph(
    Gtarget, tmp_edgesigns_target, attr_name='IVTdiff'
)

# NODES (still needed for the consensus averaging step)
tmp_nodesigns_pikart = ana.compute_node_moisture_transport(
    t_hexidx_pikart[0], t_ivt_pikart[0],
    output='manual', thresholds=a_IVTthresholds,
)
Gsigned_pikart = artn.add_node_attr_to_graph(
    G_esigned_pikart, tmp_nodesigns_pikart, attr_name='IVTdiff'
)

tmp_nodesigns_target = ana.compute_node_moisture_transport(
    t_hexidx_target[0], t_ivt_target[0],
    output='manual', thresholds=a_IVTthresholds,
)
Gsigned_target = artn.add_node_attr_to_graph(
    G_esigned_target, tmp_nodesigns_target, attr_name='IVTdiff'
)


# %% EDGE BETWEENNESS: EBC CONSENSUS
# Edge betweenness on each catalog, then average for the consensus network.

l_Gs = [Gsigned_pikart, Gsigned_target]
l_Gbetw_phases = []
for n in range(2):
    # Invert weights so that shortest paths correspond to maximum weight paths
    G = ana.invert_weights(l_Gs[n])
    d_ebetw = nx.edge_betweenness_centrality(G, weight='weight')
    nx.set_edge_attributes(G, d_ebetw, "edge_betweenness")
    l_Gbetw_phases.append(G)

# Average EBC, edge weights, and IVT classes
Gcons0 = artn.average_networks_by_attributes(
    l_Gbetw_phases[0], l_Gbetw_phases[1], attr_name="IVTdiff"
)
Gcons = artn.complete_nodes(Gcons0, res)

# %% HIGHWAYS — panel a
 
# Threshold for edge display (set to 0 to show all; use ~0.0058 for the
# supplementary "significant edges only" version)
EBCTHRESH = 0
 
# Log-scaled colour norm across ALL EBC values (consensus + both catalogs)
l_allweights = [[data['edge_betweenness'] for _, _, data in Gcons.edges(data=True)]]
for nph in range(2):
    l_allweights.extend(
        [data['edge_betweenness'] for _, _, data in l_Gbetw_phases[nph].edges(data=True)]
    )
a_allweights = np.hstack(l_allweights)
wmax = np.nanmax(a_allweights)
norm = LogNorm(vmin=np.nanmin(a_allweights[a_allweights > 0]), vmax=wmax)
 
# Plot settings
proj = ccrs.EqualEarth(central_longitude=0)
d_position = {
    i: proj.transform_point(Gcons.nodes[i]['Longitude'], Gcons.nodes[i]['Latitude'],
                            src_crs=ccrs.PlateCarree())
    for i in Gcons.nodes
}
l_colmaps = [plt.get_cmap('Purples'), plt.get_cmap('Greens')]
l_alphas = [0.6, 0.6]
 
fig, ax = plt.subplots(subplot_kw={'projection': proj}, figsize=(10, 10))
ax.set_global()
ax.coastlines(color='black', linewidth=0.5)
nplot.plot_nodes(ax, Gcons, d_position)
 
# Consensus edges (greys)
CMAP_cons = plt.get_cmap('Greys')
for node1, node2 in tqdm(Gcons.edges()):
    edge_weight = Gcons.edges[node1, node2]['edge_betweenness']
    if edge_weight < EBCTHRESH:
        continue
    width = edge_weight / wmax
    color = CMAP_cons(norm(edge_weight))
    lon1, lat1 = Gcons.nodes[node1]['Longitude'], Gcons.nodes[node1]['Latitude']
    lon2, lat2 = Gcons.nodes[node2]['Longitude'], Gcons.nodes[node2]['Latitude']
    for (lon1, lat1), (lon2, lat2) in nplot.split_edges_at_meridian(lon1, lat1, lon2, lat2):
        nplot.draw_curved_edge_with_arrow(
            ax, lon1, lat1, lon2, lat2, color, width, ax.projection,
            False, l0=10, curvature=0.3, alpha=1, arrow_size=0,
        )
 
# PIKART (purples) and tARget (greens) overlays
for k, nph in enumerate([0, 1]):
    Gplot = l_Gbetw_phases[nph]
    CMAP = l_colmaps[k]
    for node1, node2 in tqdm(Gplot.edges()):
        edge_weight = Gplot.edges[node1, node2]['edge_betweenness']
        if edge_weight < EBCTHRESH:
            continue
        width = edge_weight / wmax
        color = CMAP(norm(edge_weight))
        lon1, lat1 = Gplot.nodes[node1]['Longitude'], Gplot.nodes[node1]['Latitude']
        lon2, lat2 = Gplot.nodes[node2]['Longitude'], Gplot.nodes[node2]['Latitude']
        for (lon1, lat1), (lon2, lat2) in nplot.split_edges_at_meridian(lon1, lat1, lon2, lat2):
            nplot.draw_curved_edge_with_arrow(
                ax, lon1, lat1, lon2, lat2, color, width, ax.projection,
                False, l0=10, curvature=0.3, alpha=l_alphas[k], arrow_size=0,
            )
 
plt.show()
plt.savefig(OUTPUT_PATH + "Fig4a.png", dpi=500, bbox_inches='tight')
 
 
# Separate colourbar figure
cbar_fig, cbar_axs = plt.subplots(1, 3, figsize=(25, 0.4))
fs = 20
cbar0 = plt.colorbar(plt.cm.ScalarMappable(cmap=plt.get_cmap('Purples'), norm=norm),
                     cax=cbar_axs[0], orientation='horizontal')
cbar0.set_label('EBC (PIKART)', color='black', fontsize=fs)
cbar0.ax.tick_params(labelsize=fs)
 
cbar1 = plt.colorbar(plt.cm.ScalarMappable(cmap=plt.get_cmap('Greys'), norm=norm),
                     cax=cbar_axs[1], orientation='horizontal')
cbar1.set_label('EBC (consensus)', color='black', fontsize=fs)
cbar1.ax.tick_params(labelsize=fs)
 
cbar2 = plt.colorbar(plt.cm.ScalarMappable(cmap=plt.get_cmap('Greens'), norm=norm),
                     cax=cbar_axs[2], orientation='horizontal')
cbar2.set_label('EBC (tARget-4)', color='black', fontsize=fs)
cbar2.ax.tick_params(labelsize=fs)
 
plt.subplots_adjust(wspace=0.1)
plt.show()
plt.savefig(OUTPUT_PATH + "Fig4a_cbar.png", dpi=500, bbox_inches='tight')
 
 
# %% MOISTURE along EDGES — panel b
 
Gplot = Gcons.copy()
 
proj = ccrs.EqualEarth(central_longitude=0)
d_position = {
    i: proj.transform_point(Gplot.nodes[i]['Longitude'], Gplot.nodes[i]['Latitude'],
                            src_crs=ccrs.PlateCarree())
    for i in Gplot.nodes
}
 
# Edge widths from EBC
a_weights = np.array([data['edge_betweenness'] for _, _, data in Gplot.edges(data=True)])
wmax = np.nanmax(a_weights)
linewidth = 5
 
# Colours from moisture transport class
a_ecolours, a_ewidths = nplot.get_edge_signs(Gplot, attr='IVTdiff', linewidth=linewidth)
CMAP = ListedColormap(['#B22222', '#E66100', '#FDB863', '#999999',
                       'deepskyblue', 'dodgerblue', 'navy'])
norm = Normalize(vmin=-3, vmax=3)
 
fig, ax = plt.subplots(subplot_kw={'projection': proj}, figsize=(10, 10))
ax.set_global()
ax.coastlines(color='black', linewidth=0.5)
nplot.plot_nodes(ax, Gplot, d_position)
 
for k, (node1, node2) in enumerate(tqdm(Gplot.edges())):
    edgecol = a_ecolours[k]
    edge_weight = Gplot.edges[node1, node2]['edge_betweenness']
    width = edge_weight / wmax
    color = CMAP(norm(edgecol))
    lon1, lat1 = Gplot.nodes[node1]['Longitude'], Gplot.nodes[node1]['Latitude']
    lon2, lat2 = Gplot.nodes[node2]['Longitude'], Gplot.nodes[node2]['Latitude']
    for (lon1, lat1), (lon2, lat2) in nplot.split_edges_at_meridian(lon1, lat1, lon2, lat2):
        nplot.draw_curved_edge_with_arrow(
            ax, lon1, lat1, lon2, lat2, color, width, ax.projection,
            False, l0=10, curvature=0.3, alpha=0.75, arrow_size=0,
        )
 
# Colourbar with IVT-threshold tick labels
sm = ScalarMappable(norm=norm, cmap=CMAP)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.04, aspect=30, shrink=0.8)
cbar.set_label('Net IVT change (kg/ms)', fontsize=18)
cbar.ax.tick_params(labelsize=14)
bin_edges = np.linspace(-3, 3, 8)
tick_positions = 0.5 * (bin_edges[:-1] + bin_edges[1:])
cbar.set_ticks(tick_positions)
tick_labels = (
    [f"< {a_IVTthresholds[0]:.0f}"]
    + [f"({lo:.0f},{hi:.0f})" for lo, hi in zip(a_IVTthresholds[:-1], a_IVTthresholds[1:])]
    + [f"{a_IVTthresholds[-1]:.0f} <"]
)
cbar.set_ticklabels(tick_labels)
plt.show()
plt.savefig(OUTPUT_PATH + "Fig4b.png", dpi=600, bbox_inches='tight', transparent=True)


