# Copyright (C) 2025 by
# Tobias Braun

#------------------ PATH ---------------------------#
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
from numpy.polynomial.polynomial import Polynomial
import pandas as pd
import matplotlib as mpl
#mpl.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize, LogNorm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import time

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


""" EDGES """
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
edges = np.concatenate(([-np.inf], a_IVTthresholds, [np.inf]))


# MOISTURE TRANSPORT
# PIKART
tmp_edgesigns1 = ana.compute_edge_moisture_transport(t_hexidx_pikart[0],
                                                t_ivt_pikart[0],
                                                output = 'manual',
                                                thresholds = a_IVTthresholds) 
G_esigned_pikart = artn.add_edge_attr_to_graph(Gpikart, tmp_edgesigns1, attr_name = 'IVTdiff')
# tARget
tmp_edgesigns2 = ana.compute_edge_moisture_transport(t_hexidx_target[0],
                                                t_ivt_target[0],
                                                output = 'manual',
                                                thresholds = a_IVTthresholds) 
G_esigned_target = artn.add_edge_attr_to_graph(Gtarget, tmp_edgesigns2, attr_name = 'IVTdiff')

    
    
""" NODES """
# PIKART
tmp_nodesigns1 = ana.compute_node_moisture_transport(t_hexidx_pikart[0],
                                                    t_ivt_pikart[0],
                                                    output = 'manual',
                                                    thresholds = a_IVTthresholds)
Gsigned_pikart = artn.add_node_attr_to_graph(G_esigned_pikart, tmp_nodesigns1, attr_name = 'IVTdiff')

# tARget
tmp_nodesigns2 = ana.compute_node_moisture_transport(t_hexidx_target[0],
                                                    t_ivt_target[0],
                                                    output = 'manual',
                                                    thresholds = a_IVTthresholds)
Gsigned_target = artn.add_node_attr_to_graph(G_esigned_target, tmp_nodesigns2, attr_name = 'IVTdiff')



# %% EDGE BETWEENNESS: EBC CONSENSUS
# Compute edge betweenness centrality to all edges & average for consensus network

# LOOP OVER CATALOGS
l_Gs = [Gsigned_pikart, Gsigned_target]
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

# %% HIGHWAYS - panel a

Gcons = Greal.copy()

# CAUTION: only show edges that are significant for suppl. figure, OR SET ZERO!!!!
EBCTHRESH = 0 #0.0058

# Define colornorm based on ALL EBC values with log-scaling
l_allweights = [[data['edge_betweenness'] for _, _, data in Gcons.edges(data=True)]]
for nph in range(2):
    l_allweights.extend([data['edge_betweenness'] for _, _, data in l_Gbetw_phases[nph].edges(data=True)])
a_allweights = np.hstack(l_allweights)
wmax = np.nanmax(a_allweights)
# discard zeros in color scaling
norm = LogNorm(vmin=np.nanmin(a_allweights[a_allweights > 0]), vmax=wmax)


# Plot settings
proj = ccrs.EqualEarth(central_longitude=0)
d_position = {i: proj.transform_point(Gcons.nodes[i]['Longitude'], Gcons.nodes[i]['Latitude'],
                                      src_crs=ccrs.PlateCarree()) for i in Gcons.nodes}
l_colmaps = [plt.get_cmap('Purples'), plt.get_cmap('Greens')]
l_alphas = [.6, .6]

# FIGURE
%matplotlib 
fig, ax = plt.subplots(subplot_kw={'projection': proj}, figsize=(10, 10))
ax.set_global()
ax.coastlines(color='black', linewidth=0.5)
nplot.plot_nodes(ax, Gcons, d_position)

# Plot CONSENSUS
for node1, node2 in tqdm(Gcons.edges()):
    edge_weight = Gcons.edges[node1, node2]['edge_betweenness']
    if edge_weight < EBCTHRESH:
        continue  # Skip edges below threshold

    width = edge_weight / wmax
    CMAP = plt.get_cmap('Greys')
    color = CMAP(norm(edge_weight))
    lon1, lat1 = Gcons.nodes[node1]['Longitude'], Gcons.nodes[node1]['Latitude']
    lon2, lat2 = Gcons.nodes[node2]['Longitude'], Gcons.nodes[node2]['Latitude']
    segments = nplot.split_edges_at_meridian(lon1, lat1, lon2, lat2)
    for segment in segments:
        (lon1, lat1), (lon2, lat2) = segment
        nplot.draw_curved_edge_with_arrow(
            ax, lon1, lat1, lon2, lat2, color, width, ax.projection, 
            False, l0=10, curvature=0.3, alpha=1, arrow_size=0
        )


k=0
for nph in [0,1]:
    Gplot = l_Gbetw_phases[nph]
    CMAP = l_colmaps[k] 

    # Plot edges with color mapping
    for node1, node2 in tqdm(Gplot.edges()):
        edge_weight = Gplot.edges[node1, node2]['edge_betweenness']
        if edge_weight < EBCTHRESH:
            continue  # Skip edges below threshold


        width = edge_weight / wmax
        
        # Map the edge weight to a color in the colormap
        color = CMAP(norm(edge_weight))
        
        # Get node coordinates
        lon1, lat1 = Gplot.nodes[node1]['Longitude'], Gplot.nodes[node1]['Latitude']
        lon2, lat2 = Gplot.nodes[node2]['Longitude'], Gplot.nodes[node2]['Latitude']
        
        # Split and draw edge segments
        segments = nplot.split_edges_at_meridian(lon1, lat1, lon2, lat2)
        for segment in segments:
            (lon1, lat1), (lon2, lat2) = segment
            nplot.draw_curved_edge_with_arrow(
                ax, lon1, lat1, lon2, lat2, color, width, ax.projection, 
                False, l0=10, curvature=0.3, alpha=l_alphas[k], arrow_size=0
            )
    k+=1


plt.show()
plt.savefig(OUTPUT_PATH + "Fig4a.png", dpi=500, bbox_inches='tight')



# SEPARATE COLORBAR PLOT
cbar_fig, cbar_axs = plt.subplots(1, 3, figsize=(25, 0.4))
fs = 20
# Create a colorbar for each colormap
cbar0 = plt.colorbar(plt.cm.ScalarMappable(cmap=plt.get_cmap('Purples'), norm=norm), cax=cbar_axs[0], orientation='horizontal')
cbar0.set_label('EBC (PIKART)', color='black', fontsize=fs)
cbar0.ax.tick_params(labelsize=fs)  

cbar1 = plt.colorbar(plt.cm.ScalarMappable(cmap=plt.get_cmap('Greys'), norm=norm), cax=cbar_axs[1], orientation='horizontal')
cbar1.set_label('EBC (consensus)', color='black', fontsize=fs)
cbar1.ax.tick_params(labelsize=fs)  

cbar2 = plt.colorbar(plt.cm.ScalarMappable(cmap=plt.get_cmap('Greens'), norm=norm), cax=cbar_axs[2], orientation='horizontal')
cbar2.set_label('EBC (tARget-4)', color='black', fontsize=fs)
cbar2.ax.tick_params(labelsize=fs)  
# Adjust layout to avoid overlap
plt.subplots_adjust(wspace=0.1)
plt.show()
plt.savefig(OUTPUT_PATH + "Fig4a_cbar.png", dpi=500, bbox_inches='tight')



# %% MOISTURE along EDGES - panel b

# Input graph
Gplot = Gcons.copy()


# Plot settings
proj = ccrs.EqualEarth(central_longitude=0)
d_position = {i: proj.transform_point(Gplot.nodes[i]['Longitude'], Gplot.nodes[i]['Latitude'],
                                      src_crs=ccrs.PlateCarree()) for i in Gplot.nodes}

# EDGE WIDTHS: EBC
a_weights = np.array([data['edge_betweenness'] for _, _, data in Gplot.edges(data=True)])
wmax = np.nanmax(a_weights)
linewidth = 5
# COLOURS: moisture transport
a_ecolours, a_ewidths = nplot.get_edge_signs(Gplot, attr='IVTdiff', linewidth=linewidth)
CMAP = ListedColormap(['#B22222', '#E66100', '#FDB863', '#999999', 'deepskyblue', 'dodgerblue', 'navy'])
norm = Normalize(vmin=-3, vmax=3)#


# FIGURE
%matplotlib 
fig, ax = plt.subplots(subplot_kw={'projection': proj}, figsize=(10, 10))
ax.set_global()
ax.coastlines(color='black', linewidth=0.5)
nplot.plot_nodes(ax, Gplot, d_position)
k=0
for node1, node2 in tqdm(Gplot.edges()):
    edgecol = a_ecolours[k]
    edge_weight = Gplot.edges[node1, node2]['edge_betweenness']
    width = edge_weight / wmax
    # Map the edge weight to a color in the colormap
    color = CMAP(norm(edgecol))
    # Get node coordinates
    lon1, lat1 = Gplot.nodes[node1]['Longitude'], Gplot.nodes[node1]['Latitude']
    lon2, lat2 = Gplot.nodes[node2]['Longitude'], Gplot.nodes[node2]['Latitude']
    # Split and draw edge segments
    segments = nplot.split_edges_at_meridian(lon1, lat1, lon2, lat2)
    for segment in segments:
        (lon1, lat1), (lon2, lat2) = segment
        nplot.draw_curved_edge_with_arrow(
            ax, lon1, lat1, lon2, lat2, color, width, ax.projection, 
            False, l0=10, curvature=0.3, alpha=0.75, arrow_size=0
        )
    k += 1


# Add colorbar
sm = ScalarMappable(norm=norm, cmap=CMAP)
sm.set_array([])  
cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.04, aspect=30, shrink=0.8)
cbar.set_label('Net IVT change (kg/ms)', fontsize=18)
cbar.ax.tick_params(labelsize=14)
bin_edges = np.linspace(-3, 3, 8)
tick_positions = 0.5 * (bin_edges[:-1] + bin_edges[1:])
cbar.set_ticks(tick_positions)
tick_labels = (
    [f"< {a_IVTthresholds[0]:.0f}"] +
    [f"({lo:.0f},{hi:.0f})" for lo, hi in zip(a_IVTthresholds[:-1], a_IVTthresholds[1:])] +
    [f"{a_IVTthresholds[-1]:.0f} <"]
)
cbar.set_ticklabels(tick_labels)
plt.show()
plt.savefig('/Users/tbraun/Desktop/' + "Fig4b.png", dpi=600, bbox_inches='tight', transparent=True)
#plt.savefig(OUTPUT_PATH + "Fig4b.png", dpi=500, bbox_inches='tight')




# %% MOISTURE along NODES - panel c

# Input graph
Gplot = Gcons.copy()

# Project node positions
proj = ccrs.EqualEarth(central_longitude=0)
d_position = {
    i: proj.transform_point(Gplot.nodes[i]['Longitude'], Gplot.nodes[i]['Latitude'],
                            src_crs=ccrs.PlateCarree()) for i in Gplot.nodes
}

# Extract data
signs = np.array([
    int(round(s)) if pd.notnull(s) else 0
    for s in (Gplot.nodes[i].get('sign', 0) for i in Gplot.nodes)
])
abs_signs = np.abs(signs)
max_sign = np.nanmax(abs_signs)

# Parameters
size_scale = 50  # bubble size scaling
coords = list(d_position.values())
x_coords, y_coords = zip(*coords)

# Define color map for each sign
sign_color_map = {
    -3: 'darkred',
    -2: 'peru',
    -1: 'gold',
     0: '#aaaaaa',
     1: 'deepskyblue',
     2: 'dodgerblue',
     3: 'darkblue',
}

# Assign colors and sizes
colors = [sign_color_map.get(s, '#aaaaaa') for s in signs]
sizes = [
    size_scale * (abs(s) / max_sign) if not np.isclose(s, 0) else size_scale * 0.2
    for s in signs
]

# PLOT
fig, ax = plt.subplots(subplot_kw={'projection': proj}, figsize=(10, 10))
ax.set_global()
ax.coastlines(color='black', linewidth=0.5)

# Scatter nodes
sc = ax.scatter(x_coords, y_coords,
                s=sizes,
                c=colors,
                alpha=[0.3 if s == 0 else 0.7 for s in signs],
                linewidths=0.3,
                zorder=10)

# === Custom Legend ===
# legend_elements = [
#     Line2D([0], [0], marker='o', color='w', label='Strong Loss',
#            markerfacecolor='darkred', markersize=12),
#     Line2D([0], [0], marker='o', color='w', label='Moderate Loss',
#            markerfacecolor='peru', markersize=10),
#     Line2D([0], [0], marker='o', color='w', label='Weak Loss',
#            markerfacecolor='gold', markersize=8),
#     Line2D([0], [0], marker='o', color='w', label='Neutral',
#            markerfacecolor='#aaaaaa', markersize=7),
#     Line2D([0], [0], marker='o', color='w', label='Weak Gain',
#            markerfacecolor='deepskyblue', markersize=8),
#     Line2D([0], [0], marker='o', color='w', label='Moderate Gain',
#            markerfacecolor='dodgerblue', markersize=10),
#     Line2D([0], [0], marker='o', color='w', label='Strong Gain',
#            markerfacecolor='darkblue', markersize=12),
# ]
tick_labels = (
    [f"< {a_IVTthresholds[0]:.0f}"] +
    [f"({lo:.0f},{hi:.0f})" for lo, hi in zip(a_IVTthresholds[:-1], a_IVTthresholds[1:])] +
    [f"{a_IVTthresholds[-1]:.0f} <"]
)
legend_labels = tick_labels  # use the same tick labels
legend_colors = ['#B22222', '#E66100', '#FDB863', '#999999', 'deepskyblue', 'dodgerblue', 'navy']
legend_sizes = [12, 11, 10, 9, 10, 11, 12]  # optional: scale by importance

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=label,
           markerfacecolor=color, markersize=size)
    for label, color, size in zip(legend_labels, legend_colors, legend_sizes)
]


# Place legend ABOVE the plot
fig.legend(handles=legend_elements, title='Net IVT change (kg/ms)',
           loc='lower center', bbox_to_anchor=(0.5, 0.12),
           ncol=4, fontsize=14, title_fontsize=16, frameon=False)

# Show or save
plt.subplots_adjust(top=0.9)  # make space for legend
plt.show()
plt.savefig(OUTPUT_PATH + "Fig4c.png", dpi=500, bbox_inches='tight')




