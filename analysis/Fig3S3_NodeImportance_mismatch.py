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
from matplotlib.colors import Normalize, LogNorm
import random
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
import netCDF4 as nc
import h3 as h3
from shapely.geometry import shape, mapping, Polygon, MultiPolygon, Point
import xarray as xr
from shapely.geometry import Polygon
default_geometry = Polygon()

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



# %% REAL CATALOG

"""
Figure 3 S3: differences between catalogs in terms of AR hubs
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
eps = int(ndec) # threshold: at least 2ARs/decade
thresh = 1.25*eps
# conditioning
cond = None # any network conditioning
LC_cond = None # lifecycle conditioning
# analysis
alpha = 0.1
fdamp = .85



# PIKART
ARcat = d_ars_pikart.copy()
l_arcats_pikart, d_coord_dict = artn.preprocess_catalog(ARcat, T, loc, grid_type, X, res, cond, LC_cond)
Apik, t_idx, t_hashidx, t_ivt, t_gridpik = artn.generate_transport_matrix(l_arcats_pikart, grid_type, d_coord_dict, LC_cond)
Gpikart = artn.generate_network(Apik, t_gridpik, weighted, directed, eps, self_links, weighing)

# tARget
ARcat = d_ars_target.copy()
l_arcats_target, d_coord_dict = artn.preprocess_catalog(ARcat, T, loc, grid_type, X, res, cond, LC_cond)
Atarget, t_idx, t_hashidx, t_ivt, t_gridtarget = artn.generate_transport_matrix(l_arcats_target, grid_type, d_coord_dict, LC_cond)
Gtarget = artn.generate_network(Atarget, t_gridtarget, weighted, directed, eps, self_links, weighing)

# Consensus
Gcons = artn.consensus_network([Gpikart, Gtarget], thresh, eps)



# %% SUPPL FIGs - MATCHES AND MISMATCHES

# PARAMETERS
importance = 'PageRank'#'degree_centrality'
FACT = 1#(10*ndec)#1

#cbar_label = 'PIKART - tARget (ARs/year)'
#cbar_label = 'PIKART - tARget (Node Strength (ARs/year))'
cbar_label = 'PIKART - tARget (PageRank)'


# FREQUENCY DISTRIBUTIONS
if importance == 'frequency':
    d_pikcat, d_targetcat = l_arcats_pikart[0], l_arcats_target[0]
    
    hex_counts = d_pikcat['coord_idx'].value_counts()
    hex_counts_df = hex_counts.reset_index()
    hex_counts_df.columns = ['hex_idx', 'count']
    d_degc_pikart = gpd.GeoDataFrame({
        'degc': hex_counts_df['count'],
        'hex_idx': hex_counts_df['hex_idx'],
        'geometry': [nplot.boundary_geom(hex_id) for hex_id in hex_counts_df['hex_idx']]
    })
    
    
    hex_counts = d_targetcat['coord_idx'].value_counts()
    hex_counts_df = hex_counts.reset_index()
    hex_counts_df.columns = ['hex_idx', 'count']
    d_degc_target = gpd.GeoDataFrame({
        'degc': hex_counts_df['count'],
        'hex_idx': hex_counts_df['hex_idx'],
        'geometry': [nplot.boundary_geom(hex_id) for hex_id in hex_counts_df['hex_idx']]
    })
    
    
    # Merge the two GeoDataFrames on 'hex_idx' and sum over both:
    d_degc_sum = d_degc_pikart.merge(d_degc_target, on='hex_idx', suffixes=('_pikart', '_target'), how='outer')
    
    # Fill NaN values in degc columns
    d_degc_sum['degc_pikart'].fillna(0, inplace=True)
    d_degc_sum['degc_target'].fillna(0, inplace=True)
    
    # Sum the 'degc' columns
    d_degc_sum['degc_sum'] = d_degc_sum['degc_pikart'] + d_degc_sum['degc_target']
    
    # Create a combined geometry column using combine_first
    d_degc_sum['geometry'] = d_degc_sum['geometry_pikart'].combine_first(d_degc_sum['geometry_target'])
    
    # Replace remaining None values with the default geometry
    d_degc_sum['geometry'].fillna(default_geometry, inplace=True)
    
    # Check for None values in the final geometry column
    print("Checking None values in the final geometry column:")
    print(d_degc_sum['geometry'].isnull().sum())
    
    # Convert to GeoDataFrame and set the geometry
    d_degc_sum = gpd.GeoDataFrame(d_degc_sum, geometry='geometry')
    
    # Drop the original geometry columns
    d_degc_sum.drop(columns=['geometry_pikart', 'geometry_target'], inplace=True, errors='ignore')


elif importance == 'degree_centrality':
    
    Gplot = Gpikart.copy()
    l_hexID = list(nx.get_node_attributes(Gplot, "coordID").values())
    # Create GeoDataFrame with hexagon boundaries
    d_degc_pikart = gpd.GeoDataFrame({
        'degc': ana.degree_centrality(Gplot),  # Flatten the mean temperature array
        'hex_idx': l_hexID,
        'geometry': [nplot.boundary_geom(hex_id) for hex_id in l_hexID]
    })
    
    
    Gplot = Gtarget.copy()
    l_hexID = list(nx.get_node_attributes(Gplot, "coordID").values())
    # Create GeoDataFrame with hexagon boundaries
    d_degc_target = gpd.GeoDataFrame({
        'degc': ana.degree_centrality(Gplot),  # Flatten the mean temperature array
        'hex_idx': l_hexID,
        'geometry': [nplot.boundary_geom(hex_id) for hex_id in l_hexID]
    })


elif importance == 'PageRank':

    Gplot = Gpikart.copy()
    l_hexID = list(nx.get_node_attributes(Gplot, "coordID").values())
    # Create GeoDataFrame with hexagon boundaries
    d_degc_pikart = gpd.GeoDataFrame({
        'degc': ana.pagerank(Gplot),  # Flatten the mean temperature array
        'hex_idx': l_hexID,
        'geometry': [nplot.boundary_geom(hex_id) for hex_id in l_hexID]
    })
    
    
    Gplot = Gtarget.copy()
    l_hexID = list(nx.get_node_attributes(Gplot, "coordID").values())
    # Create GeoDataFrame with hexagon boundaries
    d_degc_target = gpd.GeoDataFrame({
        'degc': ana.pagerank(Gplot),  # Flatten the mean temperature array
        'hex_idx': l_hexID,
        'geometry': [nplot.boundary_geom(hex_id) for hex_id in l_hexID]
    })
    
    
# Define indicator values for missing entries
indicator_value_target = -9999  # for hex cells missing in target
indicator_value_pikart = 9999  # for hex cells missing in pikart

# Perform an outer merge to retain hex cells from both DataFrames
merged_df = d_degc_target.merge(
    d_degc_pikart[['hex_idx', 'degc', 'geometry']],
    on='hex_idx', how='outer', suffixes=('_target', '_pikart')
)


# Fill "missing" values in 'degc' columns with indicator values
merged_df['degc_target'] = merged_df['degc_target'].fillna(indicator_value_target)
merged_df['degc_pikart'] = merged_df['degc_pikart'].fillna(indicator_value_pikart)


# For missing geometries, choose either geometry_target or geometry_pikart, preferring the available one
merged_df['geometry'] = merged_df['geometry_target'].combine_first(merged_df['geometry_pikart'])
merged_df = merged_df.set_geometry('geometry')

# Calculate 'degc_diff' as the difference between 'degc_target' and 'degc_comparison'
merged_df['degc_diff'] = merged_df.apply(
    lambda row: (row['degc_pikart'] - row['degc_target'])/FACT if row['degc_target'] != indicator_value_target and row['degc_pikart'] != indicator_value_pikart
    else indicator_value_target if row['degc_pikart'] == indicator_value_pikart
    else indicator_value_pikart,  # This covers the case where degc_target is the indicator
    axis=1
)


# Drop some columns we do not need...
merged_df = merged_df[['hex_idx', 'geometry', 'degc_target', 'degc_pikart', 'degc_diff']]

# Separate the data: real values and indicator values (where one catalg "misses" a value)
main_values = merged_df[(merged_df['degc_diff'] != indicator_value_target) & (merged_df['degc_diff'] != indicator_value_pikart)]
not_in_target = merged_df[merged_df['degc_diff'] == indicator_value_pikart]
not_in_pikart = merged_df[merged_df['degc_diff'] == indicator_value_target]

# Filter out hexagons that cross the dateline
main_values = nplot.split_hexagons(main_values)
not_in_target = nplot.split_hexagons(not_in_target)
not_in_pikart = nplot.split_hexagons(not_in_pikart)
main_values = main_values.set_geometry('geometry')
not_in_target = not_in_target.set_geometry('geometry')
not_in_pikart = not_in_pikart.set_geometry('geometry')

# Set minimum and maximum color, pruning a bit to not distort the color scale
#vmin_merge, vmax_merge = np.nanquantile(main_values.degc_diff, .01), np.nanquantile(main_values.degc_diff, .99)
vmin_merge, vmax_merge = -np.nanquantile(np.abs(main_values.degc_diff), .99), np.nanquantile(np.abs(main_values.degc_diff), .99)



%matplotlib 
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.EqualEarth()})
ax.set_global()
ax.add_feature(cfeature.COASTLINE, color='black')

main_values.plot(column='degc_diff', cmap=cm.bam, legend=False, ax=ax, transform=ccrs.PlateCarree(), vmin=vmin_merge, vmax=vmax_merge)
sm = plt.cm.ScalarMappable(cmap=cm.bam, norm=plt.Normalize(vmin=vmin_merge, vmax=vmax_merge))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.04, aspect=30)
cbar.set_label(cbar_label, color='black', fontsize=20)
cbar.ax.tick_params()
cbar.ax.tick_params(labelsize=18)  # Set to your preferred size


# Extract hexagon coordinates with indicator values
hex_centroids_target = not_in_target.geometry.centroid  # Get the centroid for each hexagon
# Scatter crosses at the centroids of hexagons with indicator values
ax.scatter(
    hex_centroids_target.x, hex_centroids_target.y,  # x and y coordinates
    color='thistle',  # Color of the crosses
    marker='x',  # Cross marker
    s=20,  # Size of the crosses
    label='Indicator Value',  # Label for legend
    zorder=5, transform=ccrs.PlateCarree(), alpha=.75  # Set zorder to place it above other plots
)


# Extract hexagon coordinates with indicator values
hex_centroids_pikart = not_in_pikart.geometry.centroid  # Get the centroid for each hexagon
# Scatter crosses at the centroids of hexagons with indicator values
ax.scatter(
    hex_centroids_pikart.x, hex_centroids_pikart.y,  # x and y coordinates
    color='olive',  # Color of the crosses
    marker='x',  # Cross marker
    s=20,  # Size of the crosses
    label='Indicator Value',  # Label for legend
    zorder=5, transform=ccrs.PlateCarree(), alpha=.75  # Set zorder to place it above other plots
)

plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'Fig3S3f.png', dpi=300, bbox_inches='tight')



# %% ONLY HISTOGRAMS


# ONLY FREQUENCIES
d_degc_sum.dropna(subset=['degc_sum'], inplace=True)
d_degc_summed = nplot.split_hexagons(d_degc_sum)
d_degc_summed = d_degc_summed.set_geometry('geometry')

%matplotlib inline
vmin_sum, vmax_sum = 0, np.nanquantile(d_degc_summed.degc_sum, .99)
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.EqualEarth()})
ax.set_global()
ax.add_feature(cfeature.COASTLINE, color='black')

d_degc_summed.plot(column='degc_sum', cmap=cm.oslo_r, legend=False, ax=ax, transform=ccrs.PlateCarree(), vmin=vmin_sum, vmax=vmax_sum)
sm = plt.cm.ScalarMappable(cmap=cm.oslo_r, norm=plt.Normalize(vmin=vmin_sum, vmax=vmax_sum))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.04, aspect=30)
cbar.set_label('AR frequency', color='black', fontsize=16)
cbar.ax.tick_params()
cbar.set_ticks(np.linspace(vmin_sum, vmax_sum, 5))
