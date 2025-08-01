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
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

# specific packages
import networkx as nx
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import geopandas as gpd


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
d_ars_pikart_raw = pd.read_csv(INPUT_PATH + 'CATALOG_1940-2023_PIKARTV1_Vallejo-Bernal_Braun_etal_2025_ERA5_full.csv')


# %% CATALOGs

"""
Fig1 S6: spherical distortions
"""

## Network parameters
# spatiotemporal extent
T = None # no clipping
X = 'global'
# nodes
hexres = 1 # h3 system, corresponds to closest resolution to 2 degrees
rectres = 7
loc = 'head'
# edges
weighing = 'absolute'
self_links = False
weighted = True
directed = True
eps = 2 # low value here
thresh = 1.25*eps
# conditioning
cond = None # any network conditioning
LC_cond = None # lifecycle conditioning


# RECTANGULAR
grid_type = 'rectangular'
ARcat = d_ars_pikart_raw.copy()
l_arcats_rect, d_coord_dict = artn.preprocess_catalog(ARcat, T, loc, grid_type, X, rectres, cond, LC_cond)
Atrect, t_idx, t_hexidx, t_ivt, t_gridrect = artn.generate_transport_matrix(l_arcats_rect, grid_type, d_coord_dict, LC_cond)
Grect = artn.generate_network(Atrect, t_gridrect, weighted, directed, eps, self_links, weighing)

# HEXAGONAL
grid_type = 'hexagonal'
ARcat = d_ars_pikart.copy()
l_arcats_hex, d_coord_dict = artn.preprocess_catalog(ARcat, T, loc, grid_type, X, hexres, cond, LC_cond)
Ahex, t_idx, t_hexidx, t_ivt, t_gridhex = artn.generate_transport_matrix(l_arcats_hex, grid_type, d_coord_dict, LC_cond)
Ghex0 = artn.generate_network(Ahex, t_gridhex, weighted, directed, eps, self_links, weighing)
Ghex = artn.complete_nodes(Ghex0, hexres)


# %% PANEL A -  RECTANGULAR


# Compute node strength from rectangular gridded AR network
d_degc_rect = gpd.GeoDataFrame({
    'degc': ana.pagerank(Grect),  
    'latitude': np.array(list(nx.get_node_attributes(Grect, "Latitude").values())),
    'longitude': np.array(list(nx.get_node_attributes(Grect, "Longitude").values()))
})


# Get unique sorted grid edges
lat_centers = np.sort(d_degc_rect['latitude'].unique())
lon_centers = np.sort(d_degc_rect['longitude'].unique())

# Estimate grid resolution
lat_res = np.min(np.diff(lat_centers))
lon_res = np.min(np.diff(lon_centers))

# Create grid edges
lat_edges = np.append(lat_centers - lat_res/2, lat_centers[-1] + lat_res/2)
lon_edges = np.append(lon_centers - lon_res/2, lon_centers[-1] + lon_res/2)

# Create 2D grid for pcolormesh
lon_grid, lat_grid = np.meshgrid(lon_edges, lat_edges)

# Pivot data into 2D array matching the grid
degc_grid = d_degc_rect.pivot(index='latitude', columns='longitude', values='degc').values


# %% PANEL B -  HEXAGONAL

l_hexID = list(nx.get_node_attributes(Ghex, "coordID").values())
d_degc_hex = gpd.GeoDataFrame({
    'nodestr': ana.pagerank(Ghex),
    'hex_idx': l_hexID,
    'geometry': [nplot.boundary_geom(hex_id) for hex_id in l_hexID]
})
# set geometry
d_degc_hex = nplot.split_hexagons(d_degc_hex)
d_degc_hex = d_degc_hex.set_geometry('geometry')


# Combine both data ranges for consistent colorbar
vmin = min(np.nanmin(degc_grid), np.nanmin(d_degc_hex['nodestr']))
vmax = max(np.nanmax(degc_grid), np.nanmax(d_degc_hex['nodestr']))
norm = Normalize(vmin=vmin, vmax=vmax)
cmap = 'Blues'


# Create figure and subplots
proj = ccrs.EqualEarth()
fig, axs = plt.subplots(
    2, 1, figsize=(8, 16),
    subplot_kw={'projection': proj},
    constrained_layout=True
)

# Left plot: pcolormesh
axs[0].set_extent([0, 180, -80, -50], crs=ccrs.PlateCarree())
axs[0].add_feature(cfeature.COASTLINE, color='black')
pcm = axs[0].pcolormesh(
    lon_grid,
    lat_grid,
    degc_grid,
    cmap=cmap,
    edgecolor='black',
    linewidth=0.1,
    shading='auto',
    transform=ccrs.PlateCarree(),
    norm=norm
)

# Right plot: GeoDataFrame plot (hexagons)
axs[1].set_extent([0, 180, -80, -50], crs=ccrs.PlateCarree())
axs[1].add_feature(cfeature.COASTLINE, color='black')
d_degc_hex.plot(
    column='nodestr',
    cmap=cmap,
    ax=axs[1],
    transform=ccrs.PlateCarree(),
    edgecolor='black',
    linewidth=0.1,
    alpha=0.9,
    norm=norm
)

# Create ScalarMappable manually for colorbar
from matplotlib.cm import ScalarMappable
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])  

# Shared colorbar
fig.colorbar(
    sm,
    ax=axs,
    orientation='horizontal',
    pad=0.05,
    shrink=0.5,
    label='PageRank'
)
plt.savefig(OUTPUT_PATH + 'Fig1S6.pdf', dpi=300, bbox_inches='tight')



