# Copyright (C) 2025 by
# Tobias Braun

#------------------ PATHS ---------------------------#

# working directory
import sys
WDPATH = "/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/paper/scripts/ARnetlab"
sys.path.insert(0, WDPATH)
# input and output
INPUT_PATH = '/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/paper/data/'
SUPP_PATH = '/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/paper/suppl_figures/'


# %% IMPORT MODULES

# standard packages
import numpy as np
import pandas as pd
import matplotlib as mpl
#mpl.use('Agg')
from matplotlib import pyplot as plt

# specific packages
import networkx as nx
from cmcrameri import cm
from tqdm import tqdm

# my packages
import ARnet_sub as artn
import NETanalysis_sub as ana


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



# %% FUNCTION


# %% PARAMETERS

Nrealiz = 200
alpha = 0.1


# %% LOAD DATA

# PIKART
d_ars_pikart = pd.read_pickle(INPUT_PATH + 'PIKART' + '_hex.pkl')
# tARget v4
d_ars_target = pd.read_pickle(INPUT_PATH + 'target' + '_hex.pkl')
# we also import the untransformed one as it contains the lf_lons needed here (only)
d_ars_target_nohex = pd.read_pickle(INPUT_PATH + 'tARget_globalARcatalog_ERA5_1940-2023_v4.0_converted.pkl')
d_ars_target['lf_lon'] = d_ars_target_nohex['lf_lon']


l_Gbetw_rndm, l_Gbetw_genesis, l_Gbetw_term, l_Gbetw_rwd = [],[],[],[]
for n in tqdm(range(Nrealiz)):
   l_Gbetw_rndm.append(nx.read_gml(INPUT_PATH + 'output/ebc_graphs/l_Gcons_rndm_' + str(n) + '.gml'))
   l_Gbetw_genesis.append(nx.read_gml(INPUT_PATH + 'output/ebc_graphs/l_Gcons_genesis_' + str(n) + '.gml'))
   l_Gbetw_term.append(nx.read_gml(INPUT_PATH +  'output/ebc_graphs/l_Gcons_term_' + str(n) + '.gml'))
   l_Gbetw_rwd.append(nx.read_gml(INPUT_PATH +  'output/ebc_graphs/l_Gcons_rewired_' + str(n) + '.gml'))



# %% REAL EBC


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
Gcons0 = artn.average_networks_by_attributes(l_Gbetw_phases[0], l_Gbetw_phases[1], attr_name= "edge_betweenness")
# Complete nodes for plotting
Gcons = artn.complete_nodes(Gcons0, res)


# %% PANEL A- HISTOGRAMS


# REAL EBC
Greal = ana.invert_weights(Gcons)
d_ebetw = nx.edge_betweenness_centrality(Greal, weight='weight')
nx.set_edge_attributes(Greal, d_ebetw, "edge_betweenness")


# Collect EBC values from models in lists
l_allebc_rndm, l_allebc_genesis, l_allebc_term, l_allebc_rwd = [], [], [], []
for n in range(Nrealiz):
    l_allebc_rndm.extend([data['edge_betweenness'] for _, _, data in l_Gbetw_rndm[n].edges(data=True)])
    l_allebc_genesis.extend([data['edge_betweenness'] for _, _, data in l_Gbetw_genesis[n].edges(data=True)])
    l_allebc_term.extend([data['edge_betweenness'] for _, _, data in l_Gbetw_term[n].edges(data=True)])
    l_allebc_rwd.extend([data['edge_betweenness'] for _, _, data in l_Gbetw_rwd[n].edges(data=True)])
# Same for real EBCs
a_real_ebc = np.hstack([data['edge_betweenness'] for _, _, data in Greal.edges(data=True)])


# Quantiles
qrndm, qgen, qterm, qrwd = np.nanquantile(l_allebc_rndm, alpha), np.nanquantile(l_allebc_genesis, alpha), np.nanquantile(l_allebc_term, alpha), np.nanquantile(l_allebc_rwd, alpha)
print(np.nanmax([qrndm, qgen, qterm, qrwd]))

# Real histogram
a_realhist, a_bins = np.histogram(a_real_ebc, bins=100)

# Figure
%matplotlib
fig = plt.figure(figsize=(5,2))
plt.plot(a_bins[:-1], a_realhist, color='slategray', alpha=.75, linewidth=3)
plt.axvline(qrndm, color='darkorange', linestyle='dashed', linewidth=2)
plt.axvline(qgen, color='purple', linestyle='dashed', linewidth=2)
plt.axvline(qterm, color='royalblue', linestyle='dashed', linewidth=2)
plt.axvline(qrwd, color='darkgreen', linestyle='dashed', linewidth=2)
#plt.xlim(1e-4, np.nanquantile(a_real_ebc, .999))
plt.xscale('log')
plt.yscale('log')
plt.ylabel('log frequency'); plt.xlabel('log EBC')
plt.savefig(SUPP_PATH + "Fig4S1a.png", dpi=500, bbox_inches='tight')




# %% PANEL B - TRHESHOLDED MAP PLOT

Gplot =  Gcons.copy()

# CAUTION: only show edges that are significant for suppl. figure, OR SET ZERO!!!!
EBCTHRESH = 0.0028792171332354417

# Define colornorm based on ALL EBC values with log-scaling
l_allweights = [[data['edge_betweenness'] for _, _, data in Gplot.edges(data=True)]]
for nph in range(2):
    l_allweights.extend([data['edge_betweenness'] for _, _, data in l_Gbetw_phases[nph].edges(data=True)])
a_allweights = np.hstack(l_allweights)
wmax = np.nanmax(a_allweights)
# discard zeros in color scaling
norm = LogNorm(vmin=np.nanmin(a_allweights[a_allweights > 0]), vmax=wmax)


# Plot settings
proj = ccrs.EqualEarth(central_longitude=0)
d_position = {i: proj.transform_point(Gplot.nodes[i]['Longitude'], Gplot.nodes[i]['Latitude'],
                                      src_crs=ccrs.PlateCarree()) for i in Gplot.nodes}
l_colmaps = [plt.get_cmap('Purples'), plt.get_cmap('Greens')]
l_alphas = [.6, .6]

# FIGURE
%matplotlib 
fig, ax = plt.subplots(subplot_kw={'projection': proj}, figsize=(10, 10))
ax.set_global()
ax.coastlines(color='black', linewidth=0.5)
nplot.plot_nodes(ax, Gplot, d_position)
# Plot CONSENSUS
for node1, node2 in tqdm(Gplot.edges()):
    edge_weight = Gplot.edges[node1, node2]['edge_betweenness']
    if edge_weight == 0 or np.isnan(edge_weight):
        continue  # skip zero or NaN edges

    width = edge_weight / wmax

    # Use Greens for edges above threshold, Oranges for below
    if edge_weight >= EBCTHRESH:
        cmap = plt.get_cmap('Blues')
    else:
        cmap = plt.get_cmap('Reds')

    color = cmap(norm(edge_weight))

    lon1, lat1 = Gplot.nodes[node1]['Longitude'], Gplot.nodes[node1]['Latitude']
    lon2, lat2 = Gplot.nodes[node2]['Longitude'], Gplot.nodes[node2]['Latitude']
    segments = nplot.split_edges_at_meridian(lon1, lat1, lon2, lat2)

    for segment in segments:
        (lon1, lat1), (lon2, lat2) = segment
        nplot.draw_curved_edge_with_arrow(
            ax, lon1, lat1, lon2, lat2, color, width, ax.projection, 
            False, l0=10, curvature=0.3, alpha=1, arrow_size=0
        )

plt.show()
plt.savefig(SUPP_PATH + "Fig4S1b.png", dpi=500, bbox_inches='tight')



# SEPARATE COLORBAR PLOT
cbar_fig, cbar_axs = plt.subplots(1, 2, figsize=(18, 0.4))
fs = 20

# Colorbar for edges ABOVE threshold
cbar_above = plt.colorbar(
    plt.cm.ScalarMappable(cmap=plt.get_cmap('Blues'), norm=norm), 
    cax=cbar_axs[0], orientation='horizontal')
cbar_above.set_label('EBC ≥ FRW threshold', color='black', fontsize=fs)
cbar_above.ax.tick_params(labelsize=fs)

# Colorbar for edges BELOW threshold
cbar_below = plt.colorbar(
    plt.cm.ScalarMappable(cmap=plt.get_cmap('Reds'), norm=norm), 
    cax=cbar_axs[1], orientation='horizontal')
cbar_below.set_label('EBC < FRW threshold', color='black', fontsize=fs)
cbar_below.ax.tick_params(labelsize=fs)

# Adjust layout
plt.subplots_adjust(wspace=0.2)
plt.show()
plt.savefig(SUPP_PATH + "Fig4S1b_cbar.png", dpi=500, bbox_inches='tight')
