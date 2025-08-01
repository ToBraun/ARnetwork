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
#mpl.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm as CMAP
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize, LogNorm
import random
import time
from scipy.stats import linregress, t
from collections import defaultdict, Counter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import json
from itertools import combinations


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
import netCDF4 as nc
import h3 as h3
from shapely.geometry import shape, mapping, Polygon, MultiPolygon, Point
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
import seaborn as sns


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
mpl.rcParams['font.size'] = 20


# %% FUNCTIONS

# %% LOAD DATA

# PIKART
d_ars_pikart = pd.read_pickle(INPUT_PATH + 'PIKART' + '_hex.pkl')
# tARget v4
d_ars_target = pd.read_pickle(INPUT_PATH + 'target' + '_hex.pkl')
# we also import the untransformed one as it contains the lf_lons needed here (only)
#d_ars_target_nohex = pd.read_pickle(PATH + 'tARget_globalARcatalog_ERA5_1940-2023_v4.0_converted.pkl')
#d_ars_target['lf_lon'] = d_ars_target_nohex['lf_lon']

# COLOURS: 
l_cols = ['#D3D3D3', "#e4ffdc","#ffb8c1","#e89595","#c0d2b8","#b6ffde","#ffdf9c","#81afe7",
"#9bc783","#87b4ad","#4fcdff","#bbab6e","#d7d084","#81d3a1","#ccffd0",
"#c3cd81","#cafcff","#afaf78","#ffe0aa","#deffb6","#f2de92","#ffd4ff",
"#29c1b8","#fffbc1","#35d4d9","#83e6bd","#d8d0ff","#ffa6b3","#aedbdf",
"#9ee4ff","#7ffffc","#bbda91","#cce3cb","#b4a2e3","#7cfceb","#7ef4ff","#a2b273","#d09cc0",
"#d9a2e3","#ffbdf0","#54b8f4","#cf9fa9","#bbbcff","#eab278","#c0d8ff","#ffd0e4",
"#90b49e","#cdb7ff","#87b789","#14bcda","#54b8d6","#c7ffe0","#b9a4c3","#ffc4f2",
"#a6c1ab","#71b9a2","#7eb4c4","#65e1cb","#33d2f4","#f4c383","#ffcfbf"]


# %% REAL CATALOG


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
eps = 8 #4 threshold: low value here
thresh = 1.25*eps
# conditioning
cond = None # any network conditioning
LC_cond = None # lifecycle conditioning


# Generate real networks 
l_arcats1, d_coord_dict1 = artn.preprocess_catalog(d_ars_pikart, T, loc, grid_type, X, res, cond, LC_cond)
l_arcats2, d_coord_dict2 = artn.preprocess_catalog(d_ars_target, T, loc, grid_type, X, res, cond, LC_cond)
tmp_arcat1, tmp_arcat2 = l_arcats1[0], l_arcats2[0]
A1, t_idx1, t_hexidx1, t_ivt1, t_grid1 = artn.generate_transport_matrix([tmp_arcat1], grid_type, d_coord_dict1, LC_cond)
G01 = artn.generate_network(A1, t_grid1, weighted, directed, eps, self_links, weighing)
A2, t_idx2, t_hexidx2, t_ivt2, t_grid2 = artn.generate_transport_matrix([tmp_arcat2], grid_type, d_coord_dict2, LC_cond)
G02 = artn.generate_network(A2, t_grid2, weighted, directed, eps, self_links, weighing)
# consensus
Gcons = artn.consensus_network([G01, G02], thresh, eps)
# complete the network
G = artn.complete_nodes(Gcons, 2)
## Relabel the nodes, starting from 0
Greal = nx.relabel_nodes(G, dict(zip(list(G.nodes()), range(len(list(G.nodes()))))))



# %% RUN THROUGH SEEDS 

# PARAMETERS
use_node_weights_as_flow = True
Nmin = 15
n_seeds = 100
seeds = np.random.uniform(0, 1000, n_seeds)

# Select the consensus network
G0 = Greal.copy()
# Complete nodes
Gc = artn.complete_nodes(G0, 2)
# Relabel nodes with integer indices starting from 0
G = nx.relabel_nodes(Gc, dict(zip(list(Gc.nodes()), range(len(Gc.nodes())))))

# Get coordinate IDs and map them to new relabeled node IDs
d_coordID = nx.get_node_attributes(G, "coordID")
d_node_comm = pd.DataFrame()
d_node_comm['hex_id'] = [d_coordID[node] for node in G.nodes()]


# Loop over seeds and compute community assignments
for i, seed in tqdm(enumerate(seeds), total=n_seeds):
    communities, _ = ana.detect_non_hierarchical_communities(
        G,
        use_node_weights_as_flow=use_node_weights_as_flow,
        filename=None,
        return_flows=True,
        seed=seed
    )

    # Relabel community indices consistently
    communities = ana.relabel_communities(communities)

    # Filter out small communities
    community_sizes = Counter(communities.values())
    filtered_communities = {
        node: (comm if community_sizes[comm] >= Nmin else -999)
        for node, comm in communities.items()
    }

    # Save community column
    colname = f"community_{i+1}"
    d_node_comm[colname] = [filtered_communities[node] for node in G.nodes()]

# Add final assignment column for the last iteration (can be overwritten later by 'central_community')
d_node_comm['community'] = d_node_comm[f'community_{n_seeds}']


# Grab the list of nodes
node_list = d_node_comm.index.values


# Add hex IDs as a column
d_node_comm['hex_id'] = [d_coordID[node] for node in node_list]
# Add community assignments - we just take the last iteration from the loop!!!!
d_node_comm['community'] = [filtered_communities[node] for node in node_list]
## add hexagon geometries
d_node_comm["geometry"] = [nplot.boundary_geom(hex_id) for hex_id in d_node_comm.hex_id]
# split hexagons
d_node_comm = nplot.split_hexagons(d_node_comm)
# set geometry
d_node_comm = d_node_comm.set_geometry('geometry');



# %% Mean

# Compute a 'consensus, i.e., what is the set of nodes a node becomes most consistently grouped with?
d_node_comm_mean = ana.compute_consensus_communities_setwise(d_node_comm, seeds)

# Prepare discrete color mapping
unique_values = sorted(d_node_comm_mean['consensus_community'].unique())
n_colors = len(l_cols)
print('Number of consensus communities (incl -999):', n_colors)
# Too small communities are gray
value_to_color = {-999: '#D3D3D3'}
for idx, val in enumerate(unique_values):
    if val == -999:
        continue
    value_to_color[val] = l_cols[idx % n_colors]
# Generate cmap and norm
cmap_colors = [value_to_color[val] for val in unique_values]
cmap = mpl.colors.ListedColormap(cmap_colors)
bounds = unique_values + [unique_values[-1] + 1]
norm = mpl.colors.BoundaryNorm(bounds, len(cmap_colors))

# Dissolve ignoring -999 for boundaries
d_dissolve = d_node_comm_mean[d_node_comm_mean['consensus_community'] != -999]
d_boundaries = d_dissolve.dissolve(by='consensus_community', as_index=False)
d_boundaries['geometry'] = d_boundaries.buffer(0.01).buffer(-0.01)


# PLOT
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.EqualEarth()})
ax.set_global()
ax.add_feature(cfeature.COASTLINE, color='black')

plot = d_node_comm_mean.plot(
    column='consensus_community',
    cmap=cmap,
    norm=norm,
    ax=ax,
    alpha=1,
    linewidth=0.2,
    edgecolor='white',
    transform=ccrs.PlateCarree()
)

d_boundaries.boundary.plot(
    ax=ax,
    linewidth=1.0,
    edgecolor='black',
    transform=ccrs.PlateCarree()
)

plt.show()
plt.savefig(OUTPUT_PATH + "Fig5S1c.png", dpi=500, bbox_inches='tight')



# %% Heterogeneity

# Compute heterogeneity, i.e., how inconsistently is a node getting grouped?
d_node_comm_disp = ana.compute_heterogeneity_setwise(d_node_comm, seeds)

# Copy heterogeneity and set NaNs and 0 to -1 (for coloring as gray)
het_vals = d_node_comm_disp['heterogeneity'].copy()
het_vals[(het_vals.isna()) | (het_vals == 0)] = -1

# Create a colormap with gray for -1 + continuous map for (0, 1]
cmap_het = cm.bilbao_r
colors = cmap_het(np.linspace(0, 1, 256))
# prepend gray color for -1 and 0
colors = np.vstack((np.array([0.827, 0.827, 0.827, 1.0]), colors))
cmap_het_custom = mpl.colors.ListedColormap(colors)

# Normalize: -1 maps to 0 (gray), (0,1] maps to [1, 257]
norm_het = mpl.colors.BoundaryNorm([-1] + list(np.linspace(0, 1, 257)), cmap_het_custom.N)

# Assign for plotting
d_node_comm_disp['heterogeneity_plot'] = het_vals


# PLOT
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.EqualEarth()})
ax.set_global()
ax.add_feature(cfeature.COASTLINE, color='black')

# Plot without border for gray entries
PLOT = d_node_comm_disp.plot(
    column='heterogeneity_plot',
    cmap=cmap_het_custom,
    norm=norm_het,
    ax=ax,
    alpha=1,
    linewidth=0.1,
    edgecolor=d_node_comm_disp['heterogeneity_plot'].apply(lambda x: 'none' if x == -1 else 'white'),
    transform=ccrs.PlateCarree()
)
# Attach a new axis for the colorbar that's the same height as the map
sm = mpl.cm.ScalarMappable(cmap=cmap_het, norm=mpl.colors.Normalize(vmin=0, vmax=1))
sm.set_array([])
# Get position of the main map axis
bbox = ax.get_position()
# Create new axis for the colorbar manually
cax = fig.add_axes([bbox.x1 + 0.01, bbox.y0, 0.02, bbox.height])  # [left, bottom, width, height]
cbar = fig.colorbar(sm, cax=cax)
cbar.set_label('Label Inconsistency')
plt.show()
plt.savefig(OUTPUT_PATH + "Fig5S1d.png", dpi=500, bbox_inches='tight')


# %% MOST CENTRAL REALIZATION 

# Find central seed and ARI scores
central_seed, scores = ana.pick_most_consistent_seed(d_node_comm, d_node_comm_disp, seeds)
central_idx = np.where(seeds == central_seed)[0][0]
# Assign central community to a new column
d_node_comm['central_community'] = d_node_comm[f'community_{central_idx+1}']


# Define the color mapping with gray for -1
unique_values = sorted(d_node_comm['central_community'].unique())  # Get all unique values in 'degc'
n_colors = len(l_cols)

print('Number of communities: ' + str(len(unique_values)))
print('Number of colours: ' +  str(n_colors-1))

# Map unique values to colors with gray explicitly mapped to -999
value_to_color = {-999: '#D3D3D3'}  # Explicitly map gray to -999
for idx, value in enumerate([val for val in unique_values]):
    value_to_color[value] = l_cols[idx]  # Map remaining values to l_cols
# Create cmap_colors in the correct order of unique_values
cmap_colors = [value_to_color[val] for val in unique_values]
# Create a ListedColormap and ensure it aligns with the unique_values
cmap = mpl.colors.ListedColormap(cmap_colors)
# Ensure the boundaries and normalization align correctly
bounds = unique_values + [unique_values[-1] + 1]  # Extend bounds for proper normalization
norm = mpl.colors.BoundaryNorm(bounds, len(cmap_colors))


# DEFINE BOUNDARIES
d_dissolve = d_node_comm[d_node_comm['central_community'] != -999]
## Dissolve geometries by community
d_boundaries = d_dissolve.dissolve(by='central_community', as_index=False)
## Smooth geometries using buffer-unbuffer trick 
d_boundaries['geometry'] = d_boundaries.buffer(0.01).buffer(-0.01)


# FIGURE
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.EqualEarth()})
ax.set_global()
ax.add_feature(cfeature.COASTLINE, color='black')

# Plot GeoDataFrame with specified edge widths and corrected colormap
plot = d_node_comm.plot(
    column='central_community',
    cmap=cmap,
    norm=norm,
    ax=ax,
    alpha=1,
    linewidth=0.2,  # Adjust edge width
    edgecolor='white',  # Optional: specify edge color
    transform=ccrs.PlateCarree()
)
# Plot community boundaries
d_boundaries.boundary.plot(
    ax=ax,
    linewidth=1.0,
    edgecolor='black',
    transform=ccrs.PlateCarree()
)
plt.savefig(OUTPUT_PATH + "Fig5S1a.png", dpi=500, bbox_inches='tight')



# %% FLOW RATIOS

%matplotlib 

# Run again for most central partition to retain flows
communities, d_flow = ana.detect_non_hierarchical_communities(
    G,
    use_node_weights_as_flow=use_node_weights_as_flow,
    filename=None,
    return_flows=True,
    seed = central_seed
)

# Use central_community as the community assignment
comm_series = d_node_comm['central_community']

# Count sizes of communities (excluding -999)
comm_sizes = comm_series[comm_series != -999].value_counts()

# Filter out small communities; assign -999 to nodes in small communities or -999 themselves
filtered_communities = {
    node: (comm if (comm != -999 and comm_sizes.get(comm, 0) >= Nmin) else -999)
    for node, comm in zip(d_node_comm.index, comm_series)
}

# Now compute flow ratios on this filtered dict
d_flowrat = ana.compute_community_flow_ratio(d_flow, filtered_communities)

# Map flow ratios back onto the nodes in d_node_comm by index (node IDs)
d_node_comm['flow_ratio'] = d_node_comm['central_community'].map(d_flowrat)

# Prepare flow_ratio_plot for plotting: set NaN or zero to -1 (gray)
flow_vals = d_node_comm['flow_ratio'].copy()
flow_vals[(flow_vals.isna()) | (flow_vals == 0)] = -1

# Determine thresholds
valid_vals = flow_vals[flow_vals > 0]
vmin = valid_vals.min()
vmax = np.nanquantile(valid_vals, .95)
supermax_val = vmax + 1  # value to assign for visual outliers

# Mark values above vmax
flow_vals_plot = flow_vals.copy()
flow_vals_plot[flow_vals_plot > vmax] = supermax_val

d_node_comm['flow_ratio_plot'] = flow_vals_plot

# Create custom colormap: gray (-1), gradient, then deep purple for outliers
cmap_main = cm.bamako_r
colors_main = cmap_main(np.linspace(0, 1, 256))
gray = np.array([[0.827, 0.827, 0.827, 1.0]])  # light gray
highlight = np.array([[0.3, 0.0, 0.3, 1.0]])   # deep purple for outliers
colors_all = np.vstack((gray, colors_main, highlight))
cmap_combined = ListedColormap(colors_all)

# BoundaryNorm: map -1 to 0, [vmin, vmax] to 1-256, and supermax to 257
boundaries = [-1] + list(np.linspace(vmin, vmax, 256)) + [supermax_val + 0.01]
norm_combined = BoundaryNorm(boundaries, cmap_combined.N)

# Plotting
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.EqualEarth()})
ax.set_global()
ax.add_feature(cfeature.COASTLINE, color='black')

# Plot nodes
d_node_comm.plot(
    column='flow_ratio_plot',
    cmap=cmap_combined,
    norm=norm_combined,
    ax=ax,
    alpha=1,
    linewidth=0.1,
    edgecolor=d_node_comm['flow_ratio_plot'].apply(lambda x: 'none' if x == -1 else 'white'),
    transform=ccrs.PlateCarree()
)

# Colorbar for normal range (excluding gray and outlier color)
sm = mpl.cm.ScalarMappable(cmap=cmap_main, norm=Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])

# Add colorbar next to the map
bbox = ax.get_position()
cax = fig.add_axes([bbox.x1 + 0.01, bbox.y0, 0.02, bbox.height])
cbar = fig.colorbar(sm, cax=cax)
cbar.set_label('Flow Ratio (clipped at 95th percentile)')
plt.show()
plt.savefig(OUTPUT_PATH + "Fig5S1b.png", dpi=500, bbox_inches='tight')




