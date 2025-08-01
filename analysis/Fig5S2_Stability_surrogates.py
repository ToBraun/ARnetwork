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
COLPATH = '/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/data/'

# %% IMPORT MODULES

# standard packages
import numpy as np
import pandas as pd
import matplotlib as mpl
#mpl.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm as CMAP
from matplotlib.colors import Normalize, LogNorm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap
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


# COLOURS: stored in a json
with open(COLPATH + "null_model_colours.json", "r") as f:
    colour_dict = json.load(f)
random_colours = colour_dict["Random"]
genesis_colours = colour_dict["Genesis"]
termination_colours = colour_dict["Termination"]
rewired_colours = colour_dict["rewired"]



# %% PARAMETERS

# Fixed global parameters
T = None#[1979, 2023]
X = 'global'#[-90, -45, 90, 180]
weighing='absolute'
self_links = False
res = 2
weighted = True
directed = True
ndec = 8.4
eps = 4
thresh = 1.25*eps
## fix locator
grid_type = 'hexagonal'
loc = 'centroid'
cond = None#'season'
LC_cond = None


# %% NETWORKS

# Random graphs
## We choose only 20 realizations due to computation times...
Nrealiz = 20
l_Gcons_rndm, l_Gcons_genesis, l_Gcons_term, l_Gcons_rewired = [], [], [], []
for n in tqdm(range(Nrealiz)):
    l_Gcons_rndm.append(nx.read_gml(INPUT_PATH + 'random_graphs/l_Gwalk_rndm_' + str(n) + '._cons_' + loc + '.gml'))
    l_Gcons_rewired.append(nx.read_gml(INPUT_PATH + 'random_graphs/l_G_rewired_' + str(n) + '_cons_' + loc + '.gml'))
    l_Gcons_genesis.append(nx.read_gml(INPUT_PATH + 'random_graphs/l_Gwalk_genesis_' + str(n) + '_cons_' + loc + '.gml'))
    l_Gcons_term.append(nx.read_gml(INPUT_PATH + 'random_graphs/l_Gwalk_term_' + str(n) + '_cons_' + loc + '.gml'))


# %% SEEDS FOR ONE RANDOM NETWORK EACH


# FRW, GCW, TCW, RWG
MODEL = 'FRW'

# PARAMETERS
use_node_weights_as_flow = True
Nmin = 15
n_seeds = 50
seeds = np.random.uniform(0, 1000, n_seeds)

# Choose one graph: e.g., first rewired graph
if MODEL == 'FRW':
    G0 = l_Gcons_rndm[0].copy()
    l_cols = random_colours
if MODEL == 'GCW':
    G0 = l_Gcons_genesis[0].copy()
    l_cols = genesis_colours
if MODEL == 'TCW':
    G0 = l_Gcons_term[0].copy()
    l_cols = termination_colours
if MODEL == 'RWG':
    G0 = l_Gcons_rewired[0].copy()
    l_cols = rewired_colours


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

    # Filter out small communities
    community_sizes = Counter(communities.values())
    filtered_communities = {
        node: (comm if community_sizes[comm] >= Nmin else -999)
        for node, comm in communities.items()
    }

    colname = f"community_{i+1}"
    d_node_comm[colname] = [filtered_communities[node] for node in G.nodes()]

# Grab the list of nodes
node_list = d_node_comm.index.values



# %% EXAMPLE REALIZATION


# Add hex IDs as a column
d_node_comm['hex_id'] = [d_coordID[node] for node in node_list]
# Add community assignments - we just take the last iteration from the loop!!!!
d_node_comm['community'] = [filtered_communities[node] for node in node_list]
## add hexagon geometries
d_node_comm["geometry"] = [PLOT.boundary_geom(hex_id) for hex_id in d_node_comm.hex_id]
# split hexagons
d_node_comm = PLOT.split_hexagons(d_node_comm)
# set geometry
d_node_comm = d_node_comm.set_geometry('geometry');

# Define the color mapping with gray for -1
unique_values = sorted(d_node_comm['community'].unique())  # Get all unique values in 'degc'
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
d_dissolve = d_node_comm[d_node_comm['community'] != -999]
## Dissolve geometries by community
d_boundaries = d_dissolve.dissolve(by='community', as_index=False)
## Smooth geometries using buffer-unbuffer trick 
d_boundaries['geometry'] = d_boundaries.buffer(0.01).buffer(-0.01)


# FIGURE
%matplotlib
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.EqualEarth()})
ax.set_global()
ax.add_feature(cfeature.COASTLINE, color='black')

# Plot GeoDataFrame with specified edge widths and corrected colormap
plot = d_node_comm.plot(
    column='community',
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
plt.savefig(SUPPPATH + "Fig6S3j.png", dpi=500, bbox_inches='tight')



# %% Mean

# Compute a 'consensus, i.e., what is the set of nodes a node becomes most sonsistently grouped with?
d_node_comm_mean = compute_consensus_communities_setwise(d_node_comm, seeds)

# Prepare discrete color mapping similar to your example
unique_values = sorted(d_node_comm_mean['consensus_community'].unique())
n_colors = len(l_cols)
print('Number of consensus communities (incl -999):', n_colors)

value_to_color = {-999: '#D3D3D3'}
for idx, val in enumerate(unique_values):
    if val == -999:
        continue
    value_to_color[val] = l_cols[idx % n_colors]

cmap_colors = [value_to_color[val] for val in unique_values]
cmap = mpl.colors.ListedColormap(cmap_colors)

bounds = unique_values + [unique_values[-1] + 1]
norm = mpl.colors.BoundaryNorm(bounds, len(cmap_colors))

# Dissolve ignoring -999 for boundaries
d_dissolve = d_node_comm_mean[d_node_comm_mean['consensus_community'] != -999]
d_boundaries = d_dissolve.dissolve(by='consensus_community', as_index=False)
d_boundaries['geometry'] = d_boundaries.buffer(0.01).buffer(-0.01)

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
plt.savefig(SUPPPATH + "Fig6S3k.png", dpi=500, bbox_inches='tight')



# %% Heterogeneity

# Compute heterogeneity, i.e., how inconsistently is a node getting grouped?
d_node_comm_disp = compute_heterogeneity_setwise(d_node_comm, seeds)

# Copy heterogeneity and set NaNs and 0 to -1 (for coloring as gray)
het_vals = d_node_comm_disp['heterogeneity'].copy()
het_vals[(het_vals.isna()) | (het_vals == 0)] = -1

# Create a colormap with gray for -1 + continuous bilbao_r for (0, 1]
cmap_het = cm.bilbao_r
colors = cmap_het(np.linspace(0, 1, 256))
# prepend gray color for -1 and 0
colors = np.vstack((np.array([0.827, 0.827, 0.827, 1.0]), colors))
cmap_het_custom = mpl.colors.ListedColormap(colors)

# Normalize: -1 maps to 0 (gray), (0,1] maps to [1, 257]
norm_het = mpl.colors.BoundaryNorm([-1] + list(np.linspace(0, 1, 257)), cmap_het_custom.N)

# Assign for plotting
d_node_comm_disp['heterogeneity_plot'] = het_vals

# Plot
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
    linewidth=0.2,
    edgecolor=d_node_comm_disp['heterogeneity_plot'].apply(lambda x: 'none' if x == -1 else 'white'),
    transform=ccrs.PlateCarree()
)
# Colorbar without gray (starts at 0)

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
plt.savefig(SUPPPATH + "Fig6S3c.png", dpi=500, bbox_inches='tight')


