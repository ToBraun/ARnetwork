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
import matplotlib.patches as mpatches
from matplotlib import cm as CMAP
from matplotlib.colors import Normalize, LogNorm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap
import random
import time
from scipy.stats import linregress, t

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

# %% NETWORKS

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
eps = 8#4 # threshold: low value here
thresh = 1.25*eps
# conditioning
cond = None # any network conditioning
LC_cond = None # lifecycle conditioning

# tARget
ARcat = d_ars_pikart.copy()

# Generate real networks for one decade    
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
G = nx.relabel_nodes(G, dict(zip(list(G.nodes()), range(len(list(G.nodes()))))))



# %% NON-HIERARCHICAL COMMUNITIES


# PARAMETERS
use_node_weights_as_flow = True
Nmin = 15
SEEED = 38.22204983288413

## INFOMAP ALGORITHM - NON-HIERARCHICAL VERSION
communities = ana.detect_non_hierarchical_communities(G, 
                                        use_node_weights_as_flow=True,
                                        filename=None, 
                                        return_flows=False,
                                        seed = SEEED)

## Filtering for small communities 
# Get community sizes
community_sizes = {}
for node, comm in communities.items():
    if comm not in community_sizes:
        community_sizes[comm] = 0
    community_sizes[comm] += 1

# Create filtered communities dictionary
filtered_communities = communities.copy()
for node, comm in communities.items():
    if community_sizes[comm] < Nmin:
        filtered_communities[node] = -999  # Mark as filtered

# Node labels
d_coordID = nx.get_node_attributes(G, "coordID")
node_list = list(G.nodes())  

# Create dataframe for this time step
d_node_comm = pd.DataFrame()

# Add hex IDs as a column
d_node_comm['hex_id'] = [d_coordID[node] for node in node_list]
# Add community assignments - just one column now
d_node_comm['community'] = [filtered_communities[node] for node in node_list]
## add hexagon geometries
d_node_comm["geometry"] = [nplot.boundary_geom(hex_id) for hex_id in d_node_comm.hex_id]
# split hexagons
d_node_comm = nplot.split_hexagons(d_node_comm)
# set geometry
d_node_comm = d_node_comm.set_geometry('geometry');



# %% COMMUNITY FIGURE

# manually defined set of colours, generated using IwantHue
l_cols = ['#D3D3D3', "#14bcda", "#b6ffde", "#7cfceb", "#ffdf9c", "#a2b273", "#7ffffc",
           "#cf9fa9", "#4fcdff", "#c0d2b8", "#b4a2e3", "#9ee4ff", "#54b8d6",
           "#deffb6", "#d9a2e3", "#ffa6b3", "#afaf78", "#29c1b8", "#33d2f4", "#d8d0ff",
           "#cce3cb", "#eab278", "#83e6bd", "#a6c1ab", "#81d3a1", "#81afe7", "#7eb4c4", "#bbab6e",
           "#71b9a2", "#c0d8ff", "#aedbdf", "#ffd4ff", "#ccffd0", "#e89595", "#b9a4c3",
           "#90b49e", "#c7ffe0", "#ffd0e4", "#cdb7ff", "#ffe0aa", "#87b789", "#9bc783",
           "#ffcfbf", "#c3cd81", "#cafcff", "#e4ffdc", "#ffbdf0", "#d09cc0", "#87b4ad",
           "#bbbcff", "#65e1cb", "#ffc4f2", "#f2de92", "#fffbc1", "#35d4d9", "#54b8f4",
           "#ffb8c1", "#f4c383", "#7ef4ff", "#d7d084", "#bbda91"]



%matplotlib

# Define the color mapping with gray for -1
unique_values = sorted(d_node_comm['community'].unique())  # Get all unique values in 'degc'
n_colors = len(l_cols)

# Enough colours?
print('Number of communities: ' + str(len(unique_values)))
print('Number of colours: ' +  str(n_colors))


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


# Update the plot call
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
plt.savefig(OUTPUT_PATH + "Fig5a.png", dpi=500, bbox_inches='tight')




# %% IVT DIFFERENCE FIGURE
# %% UPSTREAM NETWORKS

# Pick a lower threshold cause some communities are not visited that frequently
eps = 2

# IVT differences
a_ivt_diffs1 = t_ivt1[0][1] - t_ivt1[0][0]
a_ivt_diffs2 = t_ivt2[0][1] - t_ivt2[0][0]
a_all_ivtdiffs = np.hstack([a_ivt_diffs1, a_ivt_diffs2])
# quantiles
qh1, qh2, qh3 = np.array([.6, .75, .90])
ql1, ql2, ql3 = 1 - qh1, 1 - qh2, 1 - qh3
# thresholds
a_thresholds = np.hstack([np.nanquantile(a_all_ivtdiffs, ql3),
                          np.nanquantile(a_all_ivtdiffs, ql2),
                          np.nanquantile(a_all_ivtdiffs, ql1),
                          np.nanquantile(a_all_ivtdiffs, qh1),
                          np.nanquantile(a_all_ivtdiffs, qh2),
                          np.nanquantile(a_all_ivtdiffs, qh3)])
edges = np.concatenate(([-np.inf], a_thresholds, [np.inf]))




# LOOP over ALL communities
for ncomm in tqdm(unique_values[1:]):
    
    # PIKART
    ARtraj_comm_pik = artn.condition_backwards_to_entry_region(tmp_arcat1, d_node_comm, comm_id=ncomm)
    A1, t_idx1, t_hashidx1, t_ivt1, t_grid1 = artn.generate_transport_matrix([ARtraj_comm_pik], grid_type, d_coord_dict1, LC_cond)
    Gcomm_pik = artn.generate_network(A1, t_grid1, weighted, directed, eps, self_links, weighing)
    # tARget
    ARtraj_comm_target = artn.condition_backwards_to_entry_region(tmp_arcat2, d_node_comm, comm_id=ncomm)
    A2, t_idx2, t_hashidx2, t_ivt2, t_grid2 = artn.generate_transport_matrix([ARtraj_comm_target], grid_type, d_coord_dict2, LC_cond)
    Gcomm_target = artn.generate_network(A2, t_grid2, weighted, directed, eps, self_links, weighing)
    
    
    # MOISTURE TRANSPORT
    # PIKART
    tmp_edgesigns1 = ana.compute_edge_moisture_transport(t_hashidx1[0],
                                                    t_ivt1[0],
                                                    output = 'manual',
                                                    thresholds = a_thresholds) 
    Gcomm_pik = artn.add_edge_attr_to_graph(Gcomm_pik, tmp_edgesigns1, attr_name='IVTdiff')
    # tARget
    tmp_edgesigns2 = ana.compute_edge_moisture_transport(t_hashidx2[0],
                                                    t_ivt2[0],
                                                    output = 'manual',
                                                    thresholds = a_thresholds) 
    Gcomm_target = artn.add_edge_attr_to_graph(Gcomm_target, tmp_edgesigns2, attr_name='IVTdiff')
    
    
    # Consensus
    Gcons = artn.average_networks_by_attributes(Gcomm_pik, Gcomm_target, attr_name='IVTdiff') 
    # complete the network
    Gplot = artn.complete_nodes(Gcons, 2)
    
    
    # Plot settings
    proj = ccrs.EqualEarth(central_longitude=0)
    d_position = {i: proj.transform_point(Gplot.nodes[i]['Longitude'], Gplot.nodes[i]['Latitude'],
                                          src_crs=ccrs.PlateCarree()) for i in Gplot.nodes}
    
    # EDGE WIDTHS: EBC
    a_weights = np.array([np.abs(data['IVTdiff']) for _, _, data in Gplot.edges(data=True)])
    wmax = np.nanmax(a_weights)
    linewidth = 5
    # COLOURS: moisture transport
    a_ecolours, a_ewidths = nplot.get_edge_signs(Gplot, 'IVTdiff', linewidth)
    CMAP = ListedColormap(['#B22222', '#E66100', '#FDB863', '#999999', 'deepskyblue', 'dodgerblue', 'navy'])
    norm = Normalize(vmin=-3, vmax=3)#
    
    
    # Community contour
    community_mask = d_node_comm[d_node_comm['community'] == ncomm]
    d_boundaries = community_mask.dissolve(by='community', as_index=False)
    d_boundaries['geometry'] = d_boundaries.buffer(0.01).buffer(-0.01)

    
    
    # Determine hemisphere based on the community centroid
    centroid_lat = d_boundaries.geometry.centroid.y.values[0]
    
    if centroid_lat >= 0:
        # Northern Hemisphere
        lat_min, lat_max = 0, 90
    else:
        # Southern Hemisphere
        lat_min, lat_max = -90, 0
    
    # Apply longitudinal extent to keep global view, adjust if needed
    lon_min, lon_max = -180, 180

    # FIGURE
    %matplotlib 
    fig, ax = plt.subplots(subplot_kw={'projection': proj}, figsize=(10, 10))
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

#    ax.set_global()
    ax.coastlines(color='black', linewidth=0.5)
    nplot.plot_nodes(ax, Gplot, d_position)
    k=0
    for node1, node2 in tqdm(Gplot.edges()):
        edgecol = a_ecolours[k]
        edge_weight = np.abs(Gplot.edges[node1, node2]['IVTdiff'])
        width = edge_weight / wmax
        # Map the edge weight to a color in the colormap
        color = CMAP(norm(edgecol))
        # Get node coordinates
        lon1, lat1 = Gplot.nodes[node1]['Longitude'], Gplot.nodes[node1]['Latitude']
        lon2, lat2 = Gplot.nodes[node2]['Longitude'], Gplot.nodes[node2]['Latitude']
        
    
        # Check if both nodes are inside the community polygon
        pt1, pt2 = Point(lon1, lat1), Point(lon2, lat2)
        inside1 = d_boundaries.contains(pt1).any()
        inside2 = d_boundaries.contains(pt2).any()
        if inside1 and inside2:
            alpha = 0.03
        else:
            alpha = 0.4
        
        # Split and draw edge segments
        segments = nplot.split_edges_at_meridian(lon1, lat1, lon2, lat2)
        for segment in segments:
            (lon1, lat1), (lon2, lat2) = segment
            nplot.draw_curved_edge_with_arrow(
                ax, lon1, lat1, lon2, lat2, color, width, ax.projection, 
                False, l0=1, curvature=0.3, alpha=alpha, arrow_size=0
            )
        k += 1
        
    
    # Plot the boundary of the community
    d_boundaries.boundary.plot(ax=ax, edgecolor='black', linewidth=2, zorder=5, transform=ccrs.PlateCarree())
    plt.show()
    plt.savefig("/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/results/backwards_networks_into_communities/centroids/" + "backwnet_into_comm" + str(ncomm) + ".png", dpi=500, bbox_inches='tight')

    if ncomm == 1:
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
            [f"< {a_thresholds[0]:.0f}"] +
            [f"({lo:.0f},{hi:.0f})" for lo, hi in zip(a_thresholds[:-1], a_thresholds[1:])] +
            [f"> {a_thresholds[-1]:.0f}"]
        )
        cbar.set_ticklabels(tick_labels)
        plt.show()
        plt.savefig("/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/results/backwards_networks_into_communities/centroids/cbar.pdf", dpi=500, bbox_inches='tight')


