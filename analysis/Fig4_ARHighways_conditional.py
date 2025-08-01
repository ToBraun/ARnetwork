# Copyright (C) 2025 by
# Tobias Braun

#------------------ PATH ---------------------------#
# working directory
import sys
WDPATH = "/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/paper/scripts/ARnetlab"
sys.path.insert(0, WDPATH)
# input and output
INPUT_PATH = '/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/paper/data/'
INDICES_PATH = '/Users/tbraun/Desktop/data/Global_Climate_Indices/'
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

# Function to classify phases with sustained conditions
def classify_sustained_phases(index_series, threshold, min_periods, phase_value, zscore = False):
    if zscore:
        index_series = (index_series - np.nanmean(index_series))/np.std(index_series)
    # Find consecutive periods meeting the threshold
    condition_met = index_series > threshold if phase_value == 1 else index_series < threshold
    condition_met = condition_met.astype(int)
    
    # Identify consecutive runs
    sustained_periods = (condition_met.groupby((condition_met != condition_met.shift()).cumsum())
                         .cumsum() >= min_periods)
    return sustained_periods * phase_value

# %% LOAD DATA

# PIKART
d_ars_pikart = pd.read_pickle(INPUT_PATH + 'PIKART' + '_hex.pkl')
# tARget v4
d_ars_target = pd.read_pickle(INPUT_PATH + 'target' + '_hex.pkl')
# we also import the untransformed one as it contains the lf_lons needed here (only)
d_ars_target_nohex = pd.read_pickle(INPUT_PATH + 'tARget_globalARcatalog_ERA5_1940-2023_v4.0_converted.pkl')
d_ars_target['lf_lon'] = d_ars_target_nohex['lf_lon']

# ENSO
d_enso = pd.read_csv(INDICES_PATH + 'ONI.txt', delim_whitespace=True, header=None, dtype=str, skiprows=1, engine='python')
d_enso = d_enso[d_enso.apply(lambda row: not row.str.contains('-99.90').any(), axis=1)]


# %% ENSO pre-processing

# We set ENSO as the oscillation index dataframe
d_oscidx = d_enso.copy()

# Parameters for event detection
WIN = 3 #months
thresh1, thresh2 = 0.5, -0.5 # amplitude threshold
min_consecutive_periods = 5 #months
zscore = True

# TRANSFORM IT TO MAKE IT COMPATIBLE WITH CATALOG FORMAT
# Convert remaining valid data to float
d_oscidx = d_oscidx.astype(float)
# Assign column names for readability
d_oscidx.columns = ['Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
# Reshape data to long format with a Date column for easier time-based processing
d_oscidx_long = d_oscidx.melt(id_vars='Year', var_name='Month', value_name='COidx')
d_oscidx_long['Month'] = pd.to_datetime(d_oscidx_long['Month'], format='%b').dt.month
d_oscidx_long['Date'] = pd.to_datetime(d_oscidx_long[['Year', 'Month']].assign(Day=1))
d_oscidx_long = d_oscidx_long.set_index('Date').sort_index()

# Calculate 3-month rolling averages to simulate overlapping three-month periods
d_oscidx_long['COidx_3month_avg'] = d_oscidx_long['COidx'].rolling(window=WIN, min_periods=3).mean()

# Initialize the 'teleconnection' column with NaNs
d_oscidx_long['teleconnection'] = 0  # Start with Neutral (0) for all periods

# Classify two regimes: 1: El Nino, -1: La Nina
d_oscidx_long['teleconnection'] = classify_sustained_phases(d_oscidx_long['COidx_3month_avg'], thresh1, min_consecutive_periods, 1, zscore=zscore)
d_oscidx_long['teleconnection'] += classify_sustained_phases(d_oscidx_long['COidx_3month_avg'], thresh2, min_consecutive_periods, -1, zscore=zscore)

# Set any remaining periods as Neutral (0)
d_oscidx_long['teleconnection'] = d_oscidx_long['teleconnection'].replace({np.nan: 0}).astype(int)
d_oscidx = d_oscidx_long.copy()



# %% PARAMETERS

# Merge with ARcat DataFrame based on time
ARcat = d_ars_pikart.copy()
# Resample to daily res
d_oscidx_daily = d_oscidx['teleconnection'].resample('D').ffill()
# Merge
ARcat['time'] = pd.to_datetime(ARcat['time']).dt.floor('D')
ARcat = ARcat.merge(d_oscidx_daily.rename('teleconnection'), left_on='time', right_index=True, how='left')


# SANITY CHECK
%matplotlib inline
# Plot the original and 3-month average ONI time series
plt.figure(figsize=(12, 6))
plt.plot(d_oscidx_long.index, d_oscidx_long['COidx'], label='Original', color='gray', alpha=0.5)
secx = plt.twinx()
secx.plot(d_oscidx_long.index, d_oscidx_long['COidx_3month_avg'], label='3-Month Avg', color='black', linewidth=2)

# Overlay classified points from `d_oscidx_long`
el_nino = d_oscidx_long[d_oscidx_long['teleconnection'] == 1]
la_nina = d_oscidx_long[d_oscidx_long['teleconnection'] == -1]
neutral = d_oscidx_long[d_oscidx_long['teleconnection'] == 0]

secx.scatter(el_nino.index, el_nino['COidx_3month_avg'], color='red', label='El Niño (Original)', marker='o', s=30)
secx.scatter(la_nina.index, la_nina['COidx_3month_avg'], color='blue', label='La Niña (Original)', marker='o', s=30)
secx.scatter(neutral.index, neutral['COidx_3month_avg'], color='goldenrod', label='Neutral (Original)', marker='o', s=30)

# Overlay classified points from `ARcat`'s `teleconnection` column for comparison
el_nino_merged = ARcat[ARcat['teleconnection'] == 1]
la_nina_merged = ARcat[ARcat['teleconnection'] == -1]
neutral_merged = ARcat[ARcat['teleconnection'] == 0]

# Adjust y-position for markers from `ARcat` to avoid overlap with ONI values
secx.scatter(el_nino_merged['time'], [0.25]*len(el_nino_merged), color='red', label='El Niño (Merged)', marker='x', s=50)
secx.scatter(la_nina_merged['time'], [0.25]*len(la_nina_merged), color='blue', label='La Niña (Merged)', marker='x', s=50)
secx.scatter(neutral_merged['time'], [0.25]*len(neutral_merged), color='goldenrod', label='Neutral (Merged)', marker='x', s=50)

# Add labels and legend
plt.xlabel('Date')
plt.ylabel('Index Value')
secx.set_ylabel('Teleconnection Classification')

# Combine legends from both plots
plt.legend(loc="upper left")
secx.legend(loc="upper right")
plt.show()


# %% CONDITIONAL AR NETWORKS

"""
SEASONS order: winter, summer
ENSO order: La Nina, neutral, El Nino
"""

# PICK THE INDEX
cond = 'season'

if cond == 'teleconnection':
    l_eps = np.array([3, 9, 3])
elif cond == 'season':
    l_eps = np.array([5,5,5,5])

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
eps = l_eps
thresh = np.array(1.25*eps, dtype=int)
# conditioning
LC_cond = None # lifecycle conditioning


# PIKART
ARcat = d_ars_pikart.copy()
ARcat['time'] = pd.to_datetime(ARcat['time']).dt.floor('D')
if cond != 'season':
    d_oscidx_daily = d_oscidx['teleconnection'].resample('D').ffill()
    ARcat = ARcat.merge(d_oscidx_daily.rename('teleconnection'), left_on='time', right_index=True, how='left')
ARcat['lf_lon'] = ARcat['lf_lon'].replace(0, np.nan)
# Convert landfall latitudes and longitudes to hexagon index
l_arcats_pikart, d_coord_dict = artn.preprocess_catalog(ARcat, T, loc, grid_type, X, res, cond, LC_cond)
Apik, t_idx_pikart, t_hexidx_pikart, t_ivt_pikart, t_grid_pikart = artn.generate_transport_matrix(l_arcats_pikart, grid_type, d_coord_dict, LC_cond)
Gpikart = artn.generate_network(Apik, t_grid_pikart, weighted, directed, eps, self_links, weighing)
# SEASONAL: only look at DJF and JJA
if cond == 'season':
    ns1, ns2 = 2, 3
    Gpikart, t_idx_pikart, t_hexidx_pikart, t_ivt_pikart = [Gpikart[ns1], Gpikart[ns2]], [t_idx_pikart[ns1], t_idx_pikart[ns2]], [t_hexidx_pikart[ns1], t_hexidx_pikart[ns2]], [t_ivt_pikart[ns1], t_ivt_pikart[ns2]]

# tARget
ARcat = d_ars_target.copy()
ARcat['time'] = pd.to_datetime(ARcat['time']).dt.floor('D')
if cond != 'season':
    d_oscidx_daily = d_oscidx['teleconnection'].resample('D').ffill()
    ARcat = ARcat.merge(d_oscidx_daily.rename('teleconnection'), left_on='time', right_index=True, how='left')
ARcat['lf_lon'] = ARcat['lf_lon'].replace(0, np.nan)
# Convert landfall latitudes and longitudes to hexagon index
l_arcats_target, d_coord_dict = artn.preprocess_catalog(ARcat, T, loc, grid_type, X, res, cond, LC_cond)
Atarget, t_idx_target, t_hexidx_target, t_ivt_target, t_grid_target = artn.generate_transport_matrix(l_arcats_target, grid_type, d_coord_dict, LC_cond)
Gtarget = artn.generate_network(Atarget, t_grid_target, weighted, directed, eps, self_links, weighing)
# SEASONAL: only look at DJF and JJA
if cond == 'season':
    Gtarget, t_idx_target, t_hexidx_target, t_ivt_target = [Gtarget[ns1], Gtarget[ns2]], [t_idx_target[ns1], t_idx_target[ns2]], [t_hexidx_target[ns1], t_hexidx_target[ns2]], [t_ivt_target[ns1], t_ivt_target[ns2]]
Lp = len(Gtarget)

# Check: did we set the thresholds right?
print(Gpikart[0].number_of_edges())
print(Gpikart[1].number_of_edges())
print(Gpikart[2].number_of_edges())




# UNCONDITIONED NETWORK
ndec = 8.4 # number of decades
eps = int(2*ndec) # threshold: at least 2ARs/decade
thresh = 1.25*eps
## PIKART
ARcat = d_ars_pikart.copy()
l_arcats_pik, d_coord_dict = artn.preprocess_catalog(ARcat, T, loc, grid_type, X, res, None, LC_cond)
Apik, t_idx, t_ivt, t_hashidx, t_grid = artn.generate_transport_matrix(l_arcats_pik, grid_type, d_coord_dict, LC_cond)
Gpik_uncond = artn.generate_network(Apik, t_grid, weighted, directed, eps, self_links, weighing)
# tARget
ARcat = d_ars_target.copy()
l_arcats_target, d_coord_dict = artn.preprocess_catalog(ARcat, T, loc, grid_type, X, res, None, LC_cond)
Atarget, t_idx, t_hashidx, t_ivt, t_grid = artn.generate_transport_matrix(l_arcats_target, grid_type, d_coord_dict, LC_cond)
Gtarget_uncond = artn.generate_network(Atarget, t_grid, weighted, directed, eps, self_links, weighing)

# consensus
Gcons_uncond = artn.consensus_network([Gpik_uncond, Gtarget_uncond], thresh, eps)

# %% MOISTURE TRANSPORT

# Quantiles
qh1, qh2, qh3 = np.array([.6, .75, .90])
ql1, ql2, ql3 = 1 - qh1, 1 - qh2, 1 - qh3

# Iterate over seasons/regimes, collecting IVT differences
a_all_ivtdiffs = np.array([])
for n in tqdm(range(Lp)):
    # IVT differences
    a_ivt_diffs1 = t_ivt_pikart[n][1] - t_ivt_pikart[n][0]
    a_ivt_diffs2 = t_ivt_target[n][1] - t_ivt_target[n][0]
    a_regime_ivtdiffs = np.hstack([a_ivt_diffs1, a_ivt_diffs2])
    a_all_ivtdiffs = np.hstack([a_all_ivtdiffs, a_regime_ivtdiffs])
    
# Thresholds
a_IVTthresholds = np.hstack([np.nanquantile(a_all_ivtdiffs, ql3),
                          np.nanquantile(a_all_ivtdiffs, ql2),
                          np.nanquantile(a_all_ivtdiffs, ql1),
                          np.nanquantile(a_all_ivtdiffs, qh1),
                          np.nanquantile(a_all_ivtdiffs, qh2),
                          np.nanquantile(a_all_ivtdiffs, qh3)])
edges = np.concatenate(([-np.inf], a_IVTthresholds, [np.inf]))


# MOISTURE TRANSPORT
# Iterate over seasons/regimes, assigning IVT classes
Gsigned_pikart, Gsigned_target = Gpikart.copy(), Gtarget.copy()
for n in tqdm(range(Lp)):
    # PIKART
    tmp_edgesigns1 = ana.compute_edge_moisture_transport(t_hexidx_pikart[n],
                                                    t_ivt_pikart[n],
                                                    output = 'manual',
                                                    thresholds = a_IVTthresholds) 
    Gtmp = artn.add_edge_attr_to_graph(Gpikart[n], tmp_edgesigns1, attr_name='IVTdiff')
    Gsigned_pikart[n] = Gtmp.copy()
    # tARget
    tmp_edgesigns2 = ana.compute_edge_moisture_transport(t_hexidx_target[n],
                                                    t_ivt_target[n],
                                                    output = 'manual',
                                                    thresholds = a_IVTthresholds) 
    Gtmp = artn.add_edge_attr_to_graph(Gtarget[n], tmp_edgesigns2, attr_name='IVTdiff')
    Gsigned_target[n] = Gtmp.copy()


# %% EDGE BETWEENNESS: EBC CONSENSUS


# LOOP OVER REGIMES
l_Gbetw_phases = []
for nph in tqdm(range(Lp)):
    l_Gs = [Gsigned_pikart[nph], Gsigned_target[nph]]
    l_Gbetw_cat = []
    for n in range(2):
        # Invert weights so that shortest paths correspond to maximum weight paths:
        G = ana.invert_weights(l_Gs[n])
        # EBC
        d_ebetw = nx.edge_betweenness_centrality(G, weight='weight')
        nx.set_edge_attributes(G, d_ebetw, "edge_betweenness")
        l_Gbetw_cat.append(G)
    l_Gbetw_phases.append(l_Gbetw_cat)

# EDGE BETWEENNESS CONSENSUS (basically averaging)
#l_Gcons = [artn.consensus_network([l_Gbetw_phases[nph][0], l_Gbetw_phases[nph][1]], 0, 0, weight_variable='edge_betweenness') for nph in range(Lp)]
## define network for neutral state
#Gneutr = l_Gcons[0].copy()

# Averaging of edge betweenness, edge weights and IVT classes:
l_Gcons0 = [artn.average_networks_by_attributes(l_Gbetw_phases[nph][0], l_Gbetw_phases[nph][1], attr_name="edge_betweenness") for nph in range(Lp)]
#l_Gcons0 = [artn.average_networks_by_attributes(l_Gbetw_phases[nph][1], l_Gbetw_phases[nph][1]) for nph in range(Lp)]
# Complete nodes for plotting
l_Gcons = [artn.complete_nodes(l_Gcons0[nph], res) for nph in range(Lp)]

# UNCONDITIONED
Ginv = ana.invert_weights(Gcons_uncond)
Gcons_uncond_inv = artn.complete_nodes(Ginv, res)
d_ebetw = nx.edge_betweenness_centrality(Gcons_uncond_inv)
nx.set_edge_attributes(Gcons_uncond_inv, d_ebetw, "edge_betweenness")



# %% PANEL A/D/G - BOTH PHASES

# Neutral state
if cond == 'season':
    # For seasonality, the 'neutral' network is the normal all-year network
    Gneutr = Gcons_uncond_inv.copy()
elif cond == 'teleconnection':
    Gneutr = l_Gcons[1].copy()


# Define colornorm based on ALL EBC values
l_allweights = []
for nph in range(Lp):
    l_allweights.extend([data['edge_betweenness'] for _, _, data in l_Gcons[nph].edges(data=True)])
a_allweights = np.hstack(l_allweights)
wmax = np.nanmax(a_allweights)
norm = LogNorm(vmin=np.nanmin(a_allweights[a_allweights>0]), vmax=wmax)#Normalize(vmin=0, vmax=wmax)#


# Plot settings
proj = ccrs.EqualEarth(central_longitude=0)
d_position = {i: proj.transform_point(Gneutr.nodes[i]['Longitude'], Gneutr.nodes[i]['Latitude'],
                                      src_crs=ccrs.PlateCarree()) for i in Gneutr.nodes}
l_colmaps = [plt.get_cmap('Purples'), plt.get_cmap('Greens')]
l_alphas = [.75, .5]

# FIGURE
%matplotlib 
fig, ax = plt.subplots(subplot_kw={'projection': proj}, figsize=(10, 10))
ax.set_global()
ax.coastlines(color='black', linewidth=0.5)
nplot.plot_nodes(ax, Gneutr, d_position)

#Plot NEUTRAL STATE
for node1, node2 in tqdm(Gneutr.edges()):
    edge_weight = Gneutr.edges[node1, node2]['edge_betweenness']
    width = edge_weight / wmax
    #color = plt.get_cmap('Purples')
    CMAP = plt.get_cmap('Greys')
    color = CMAP(norm(edge_weight))
    lon1, lat1 = Gneutr.nodes[node1]['Longitude'], Gneutr.nodes[node1]['Latitude']
    lon2, lat2 = Gneutr.nodes[node2]['Longitude'], Gneutr.nodes[node2]['Latitude']
    segments = nplot.split_edges_at_meridian(lon1, lat1, lon2, lat2)
    for segment in segments:
        (lon1, lat1), (lon2, lat2) = segment
        nplot.draw_curved_edge_with_arrow(
            ax, lon1, lat1, lon2, lat2, color, width, ax.projection, 
            False, l0=10, curvature=0.3, alpha=1, arrow_size=0
        )

k=0
for nph in [0,-1]:
    Gplot = l_Gcons[nph]
    CMAP = l_colmaps[k] 

    # Plot edges with color mapping
    for node1, node2 in tqdm(Gplot.edges()):
        edge_weight = Gplot.edges[node1, node2]['edge_betweenness']
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
plt.savefig(OUTPUT_PATH + "Fig4d.png", dpi=500, bbox_inches='tight', transparent=True)



# SEPARATE COLORBAR PLOT
cbar_fig, cbar_axs = plt.subplots(1, 3, figsize=(25, 0.5))
fs = 24
# Create a colorbar for each colormap
cbar0 = plt.colorbar(plt.cm.ScalarMappable(cmap=plt.get_cmap('Purples'), norm=norm), cax=cbar_axs[0], orientation='horizontal')
cbar0.set_label('EBC (DJF)', color='black', fontsize=fs)
cbar0.ax.tick_params(labelsize=fs)  

cbar1 = plt.colorbar(plt.cm.ScalarMappable(cmap=plt.get_cmap('Greys'), norm=norm), cax=cbar_axs[1], orientation='horizontal')
cbar1.set_label('EBC (unconditioned)', color='black', fontsize=fs)
cbar1.ax.tick_params(labelsize=fs)  

cbar2 = plt.colorbar(plt.cm.ScalarMappable(cmap=plt.get_cmap('Greens'), norm=norm), cax=cbar_axs[2], orientation='horizontal')
cbar2.set_label('EBC (JJA)', color='black', fontsize=fs)
cbar2.ax.tick_params(labelsize=fs)  
# Adjust layout to avoid overlap
plt.subplots_adjust(wspace=0.1)
plt.show()
plt.savefig(OUTPUT_PATH + "Fig4d_cbar.png", dpi=500, bbox_inches='tight')



# %% PANEL E&F/H&I - PHASE 1 & 2

# Pick phase of oscillation & average IVTdiffs per edge for consensus
nph = 1
l_Gcons0 = [artn.average_networks_by_attributes(l_Gbetw_phases[nph][0], l_Gbetw_phases[nph][1], attr_name="IVTdiff") for nph in range(Lp)]
l_Gcons = [artn.complete_nodes(l_Gcons0[nph], res) for nph in range(Lp)]
Gplot = l_Gcons[nph].copy()


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
plt.savefig(OUTPUT_PATH + "Fig4f.png", dpi=500, bbox_inches='tight')

