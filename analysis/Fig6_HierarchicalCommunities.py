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
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize, LogNorm
import time

# specific packages
import networkx as nx
from networkx.algorithms.community import modularity, partition_quality
from cmcrameri import cm
import cartopy.feature as cfeature
from tqdm import tqdm
import cartopy.crs as ccrs
from infomap import Infomap
import geopandas as gpd
from h3 import h3
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'notebook'


# my packages
import ARnet_sub as artn
import NETanalysis_sub as ana
import NETplots_sub as nplot


# %% PLOT PARAMETERS´´
plt.style.use('default')
# Update Matplotlib parameters
colorbar_dir = 'horizontal'

# Change default tick direction
params = {'xtick.direction': 'in',
          'ytick.direction': 'in'}
plt.rcParams.update(params)
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['font.size'] = 16



# %% LOAD DATA


# PIKART
d_ars_pikart = pd.read_pickle(INPUT_PATH + 'PIKART' + '_hex.pkl')
# tARget v4
d_ars_target = pd.read_pickle(INPUT_PATH + 'target' + '_hex.pkl')



# %% Global network with different edge definitions

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
eps = 8 # threshold: low value here
thresh = 1.25*eps
# conditioning
cond = None # any network conditioning
LC_cond = None # lifecycle conditioning


# PIKART
ARcat = d_ars_pikart.copy()
# TEMPORARY: need to set all zero lflons to NAN cause I fucked that up in the original script
ARcat['lf_lon'] = ARcat['lf_lon'].replace(0, np.nan)
# Convert landfall latitudes and longitudes to hexagon index
l_arcats_pikart, d_coord_dict = artn.preprocess_catalog(ARcat, T, loc, grid_type, X, res, cond, LC_cond)
Apik, t_idx, t_ivt, t_hexidx, t_grid_pik = artn.generate_transport_matrix(l_arcats_pikart, grid_type, d_coord_dict, LC_cond)
Gpikart = artn.generate_network(Apik, t_grid_pik, weighted, directed, eps, self_links, weighing)


# tARget
ARcat = d_ars_target.copy()
l_arcats_target, d_coord_dict = artn.preprocess_catalog(ARcat, T, loc, grid_type, X, res, cond, LC_cond)
Atarget, t_idx, t_hexidx, t_ivt, t_grid_target = artn.generate_transport_matrix(l_arcats_target, grid_type, d_coord_dict, LC_cond)
Gtarget = artn.generate_network(Atarget, t_grid_target, weighted, directed, eps, self_links, weighing)

# Consensus
G = artn.consensus_network([Gpikart, Gtarget], 1.25*eps, eps)


# %% FUNCTIONS

l_cols1 = ['#D3D3D3', "#f4a084", "#8eb680","#a0a9d8", "#25c0b8", "#b5c4aa", "#84e2b6","#ffc0da","#e2e496",
           "#bed1ff"]

l_cols2 = ['#D3D3D3', "#f4a084","#8eb680", "#25c0b8", "#b5c4aa", "#84e2b6","#ffc0da","#a0a9d8","#e2e496",
           "#bed1ff", "#ffe9c9","#85b3bb","#c0ffe0"]


l_cols3 = ['#D3D3D3', "#f4a084","#25c0b8", "#47bea3","#d7abef","#daf2a6","#eaaecf",
           "#d8da8d","#60b3ef","#d7a16a","#64c9ff","#faa18a","#76fef5","#ffb1ad","#86f3d2",
           "#d29ea7","#83d9a9","#e3d2ff","#88be7f","#ffd7f5","#daffd8","#92cdff","#ffdca2",
           "#72e3ff","#a0a9d8","#77b3d6","#d6ffc0","#c0a3b9","#9cd799","#ffcaca","#9bfcff",
           "#b5ac87","#97aec8","#f9ffdc","#67b9b7","#ffe9c9","#85b3bb","#c0ffe0","#96b394","#b5c4aa"]

l_cols4 = ['#D3D3D3', "#14bcda", "#b6ffde", "#7cfceb", "#ffdf9c", "#a2b273", "#7ffffc",
           "#cf9fa9", "#4fcdff", "#c0d2b8", "#b4a2e3", "#9ee4ff", "#54b8d6",
           "#deffb6", "#d9a2e3", "#ffa6b3", "#afaf78", "#29c1b8", "#33d2f4", "#d8d0ff",
           "#cce3cb", "#eab278", "#83e6bd", "#a6c1ab", "#81d3a1", "#81afe7", "#7eb4c4", "#bbab6e",
           "#71b9a2", "#c0d8ff", "#aedbdf", "#ffd4ff", "#ccffd0", "#e89595", "#b9a4c3",
           "#90b49e", "#c7ffe0", "#ffd0e4", "#cdb7ff", "#ffe0aa", "#87b789", "#9bc783",
           "#ffcfbf", "#c3cd81", "#cafcff", "#e4ffdc", "#ffbdf0", "#d09cc0", "#87b4ad",
           "#bbbcff", "#65e1cb", "#ffc4f2", "#f2de92", "#fffbc1", "#35d4d9", "#54b8f4",
           "#ffb8c1", "#f4c383", "#7ef4ff", "#d7d084", "#bbda91"]


l_allcols = [l_cols1, l_cols2, l_cols3, l_cols4]


# %% COMMUNITY DETECTION WITH INFOMAP

# PARAMETERS
use_node_weights_as_flow = True
Nmin = 15 #20 #minimum community size
minlvl, maxlvl = 1, 5
nLVL = 2


# Detect hierarchical communities using Infomap
## Complete nodes to fully cover the globe
G = artn.complete_nodes(G, res)
## Relabel the nodes, starting from 0
G = nx.relabel_nodes(G, dict(zip(list(G.nodes()), range(len(list(G.nodes()))))))
## INFOMAP ALGORITHM
a_lvlcomms00, d_total_flows = ana.detect_hierarchical_communities(G, use_node_weights_as_flow=True,
                                                  filename=None, return_flows=True)

# Limit to specified hierarchical levels
a_lvlcomms0 = a_lvlcomms00[minlvl:maxlvl]

## Filtering out the small communities
a_lvlcomms_mask = ana.filter_small_communities(a_lvlcomms0, Nmin)#[1:5]
a_lvlcomms_filtered = a_lvlcomms0.copy()
a_lvlcomms_filtered[a_lvlcomms_mask] = -999
# # FILTERING by flow
# d_total_flows, d_flow_matrices = ana.module_flow(a_lvlcomms0, d_flows)
# a_lvlcomms_filtered = ana.filter_by_flow(a_lvlcomms0, d_total_flows, flow_threshold=0.95)


# COLOURS: Do colours and communities match...?
l_cols = l_allcols[nLVL]
n_comm = np.unique(a_lvlcomms_filtered[nLVL,]).size
n_colors = len(l_cols)
a_unicomms, a_unifreqs = np.unique(a_lvlcomms_filtered[0,], return_counts=True)
print('Number of colours: ' +  str(n_colors))
print('Number of communities: ' +  str(n_comm))


# Generate the dataframe for plotting
Gplot = G.copy()
l_hexID = list(nx.get_node_attributes(Gplot, "coordID").values())
d_degc = gpd.GeoDataFrame({
    'degc': a_lvlcomms_filtered[nLVL,],
    'hex_idx': l_hexID,
    'geometry': [nplot.boundary_geom(hex_id) for hex_id in l_hexID]
})
# Use the function to split hexagons crossing the dateline
d_degc = nplot.split_hexagons(d_degc)
d_degc = d_degc.set_geometry('geometry');


# DEFINE BOUNDARIES
d_dissolve = d_degc[d_degc['degc'] != -999]
## Dissolve geometries by community
d_boundaries = d_dissolve.dissolve(by='degc', as_index=False)
## Smooth geometries using buffer-unbuffer trick 
d_boundaries['geometry'] = d_boundaries.buffer(0.01).buffer(-0.01)

# Define the color mapping with gray for -1
unique_values = sorted(d_degc['degc'].unique())  # Get all unique values in 'degc'
# Map unique values to colors with gray explicitly mapped to -999
value_to_color = {-999: '#D3D3D3'}  # Explicitly map gray to -999
for idx, value in enumerate([val for val in unique_values]):
    value_to_color[value] = l_cols[idx]  # Map remaining values to l_cols


# COLORMAP
## Create cmap_colors in the correct order of unique_values
cmap_colors = [value_to_color[val] for val in unique_values]
## Create a ListedColormap and ensure it aligns with the unique_values
cmap = mpl.colors.ListedColormap(cmap_colors)
## Ensure the boundaries and normalization align correctly
bounds = unique_values + [unique_values[-1] + 1]  # Extend bounds for proper normalization
norm = mpl.colors.BoundaryNorm(bounds, len(cmap_colors))



# FIGURE
%matplotlib 
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.EqualEarth()})
ax.set_global()
ax.add_feature(cfeature.COASTLINE, color='black')
# Plot GeoDataFrame:
plot = d_degc.plot(
    column='degc',
    cmap=cmap,
    norm=norm,
    ax=ax,
    alpha=1,
    linewidth=0.2, 
    edgecolor='white',  
    transform=ccrs.PlateCarree()
)

# Plot community boundaries
d_boundaries.boundary.plot(
    ax=ax,
    linewidth=1.0,
    edgecolor='black',
    transform=ccrs.PlateCarree()
)
plt.savefig('/Users/tbraun/Desktop/' + "Fig6d.png", dpi=600, bbox_inches='tight', transparent=True)
#plt.savefig(OUTPUT_PATH + "Fig6d.png", dpi=500, bbox_inches='tight')


# %% MODULE GRAPHS

# PARAMETERS
nLVL = 0
l_allcols = [l_cols1[1:], l_cols2[1:], l_cols3[1:], l_cols4[1:]]
SEEED = 7 #6-x-1-1

# Flows: obtain matrix-form from FILTERED partition!
d_dict_flows, d_flow_matrices = ana.module_flow(a_lvlcomms_filtered, d_total_flows)

# SELECT level
flow_matrix = d_flow_matrices[nLVL]
total_flow = d_dict_flows[nLVL]
l_cols = l_allcols[nLVL]

# GRAPH creation
## Create the directed graph
Gmod = nx.DiGraph()

## Add nodes with square-root scaled sizes
num_nodes = flow_matrix.shape[0]
for i in range(num_nodes):
    node_flow = np.nansum(flow_matrix[i, :]) + np.nansum(flow_matrix[:, i])  # or just total_flow[i]
    Gmod.add_node(i, size=np.sqrt(node_flow))
    

## Add edges, excluding self-loops
for i in tqdm(range(num_nodes)):
    for j in range(num_nodes):
        if i != j and flow_matrix[i, j] > np.nanquantile(flow_matrix, .01):
            Gmod.add_edge(i, j, weight=flow_matrix[i, j])


## Remove disconnected nodes: only remove nodes with zero total flow
disconnected_nodes = [node for node in Gmod.nodes if total_flow[node] == 0]
Gmod.remove_nodes_from(disconnected_nodes)


# Layout
## Get node sizes (scaled by square root) and colours
node_sizes = [Gmod.nodes[node]['size'] * 1000 for node in Gmod.nodes]
node_colors = [l_cols[node] for node in Gmod.nodes]
## Compute positions for the layout
k_val = 2 / np.sqrt(Gmod.number_of_nodes())
pos = nx.spring_layout(Gmod, k=k_val, iterations=100, seed=SEEED)
for key in pos:
    pos[key][0] *= 60  # Stretch along y-axis for elongation
    pos[key][1] *= 1  # Stretch along y-axis for elongation


%matplotlib 
# FIGURE
plt.figure(figsize=(10,10))
ax = plt.gca()

# Draw curved edges with arrows
for u, v, data in Gmod.edges(data=True):
    rad = 0.25 if Gmod.has_edge(v, u) else 0  # Curvature only for bidirectional edges
    nx.draw_networkx_edges(
        Gmod, pos,
        edgelist=[(u, v)],
        connectionstyle=f"arc3,rad={rad}",  # Apply curvature for bidirectional
        edge_color="gray",
        arrowstyle="-|>",  # Single-direction arrow
        min_source_margin=15,  # Add margin to improve arrow clarity
        min_target_margin=15,
        width=data['weight']*2000,  # Scaled edge width
        alpha=0.9
    )

# Draw nodes
nx.draw_networkx_nodes(
    Gmod, pos,
    node_size= 10 * np.array(node_sizes),
    node_color=node_colors,
    edgecolors="black"
)

# Configure plot aesthetics
plt.axis("off")
#plt.tight_layout()
plt.margins(0.1)  # 10% margin on all sides
plt.savefig(OUTPUT_PATH + "Fig6e.png", dpi=500)



# %% CONDUCTANCE

# PARAMETERS
Nrealiz = 200
alpha = .95
cutoff = 20

# Load the RWG realizations
l_Gwalk = []
for n in tqdm(range(Nrealiz)):
    l_Gwalk.append(nx.read_gml(INPUT_PATH + 'random_graphs/l_Gwalk_rndm_' + str(n) + '._cons_' + loc + '.gml'))

# FLOW COMPUTATIONS
l_total_flows_rndm, l_flow_matrices_rndm = [], []
for nrndm in tqdm(range(Nrealiz)):
    G = artn.complete_nodes(l_Gwalk[nrndm], res)
    ## Relabel the nodes, starting from 0
    G = nx.relabel_nodes(G, dict(zip(list(G.nodes()), range(len(list(G.nodes()))))))
    ## INFOMAP ALGORITHM
    a_lvlcomms0, d_flows = ana.detect_hierarchical_communities(G, use_node_weights_as_flow=True,filename=None, return_flows=True)
    ## flow computations
    d_total_flows_rndm, d_flow_matrices_rndm = ana.module_flow(a_lvlcomms0, d_flows)
    ## append
    l_total_flows_rndm.append(d_total_flows_rndm)
    l_flow_matrices_rndm.append(d_flow_matrices_rndm)


# CONDUCTANCE 
a_conduct, a_conduct_rndm = np.zeros((len(d_flow_matrices), cutoff)), np.zeros((len(d_flow_matrices), Nrealiz, cutoff))
for nlvl in range(len(d_flow_matrices)): 
    ## REAL
    tmp_flowmat = d_flow_matrices[nlvl]
    tmp_conduct = ana.conductance(tmp_flowmat)
    tmp_nonan_conduct = np.flip(np.sort(tmp_conduct[~ np.isnan(tmp_conduct)]))
    idx = np.min([tmp_nonan_conduct.size, cutoff])    
    a_conduct[nlvl,:idx] = tmp_nonan_conduct[:idx]
    if idx < cutoff:
        a_conduct[nlvl,idx:] = np.nan
    
    
    ## RANDOM
    for nrndm in range(Nrealiz):
        tmp_flowmat = l_flow_matrices_rndm[nrndm][nlvl]
        if tmp_flowmat.size > 1:
            tmp_conduct = ana.conductance(tmp_flowmat)
            tmp_nonan_conduct = np.flip(np.sort(tmp_conduct[~ np.isnan(tmp_conduct)]))
            idx = np.min([tmp_nonan_conduct.size, cutoff])    
            a_conduct_rndm[nlvl,nrndm,:idx] = tmp_nonan_conduct[:idx]
            if idx < cutoff:
                a_conduct_rndm[nlvl,nrndm,idx:] = np.nan
        else:
            a_conduct_rndm[nlvl, nrndm, :] = np.nan*np.ones(cutoff)
        


# PLOT
nlvl = 3
a_q = np.log10(np.nanquantile(a_conduct_rndm[nlvl,:], alpha, 0))
mpl.rcParams['font.size'] = 20

fig = plt.figure(figsize=(8,10), constrained_layout=True)
plt.barh(np.arange(cutoff), np.log10(a_conduct[nlvl,]), color='darkslategray', label='real network')  # Use barh for horizontal bars
plt.plot(
    np.hstack([a_q[0], a_q, a_q[-1]]), 
    np.arange(-1, cutoff+1, 1), 
    color='darkorange', linestyle='dashed', linewidth=5, label='random networks'
)  # Flip x and y for the line
plt.ylim(-0.5, 9.5)
plt.yticks(np.arange(5,cutoff+5,5)-1, labels=np.arange(5,cutoff+5,5))  # Adjust for y-axis ticks
plt.xlabel(r'$\eta$')  # Swap x and y labels
plt.ylabel('Top-20 communities')
plt.legend(fontsize=20)
plt.savefig(OUTPUT_PATH + "Fig6m_cbar.pdf", dpi=500, bbox_inches='tight', pad_inches=0.2)



