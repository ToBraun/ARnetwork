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
from matplotlib import cm as CMAP
from matplotlib.colors import Normalize

# specific packages
import networkx as nx
from tqdm import tqdm
import geopandas as gpd
from collections import defaultdict
import xarray as xr
from statsmodels.stats.multitest import multipletests
from cmcrameri import cm
import cartopy.feature as cfeature
import cartopy.crs as ccrs

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


# TOPOGRAPHY
dem_ds = xr.open_dataset('/Users/tbraun/Desktop/projects/#A_PIKART_PIK/ARcatalog_shared/scripts/detection&tracking/input_files/hyd_glo_dem_0_75deg.nc')
dem = dem_ds['dem'].isel(time=0)

# Parameters
Nrealiz = 200
loc = 'head'


l_Gcons_rndm, l_Gcons_genesis, l_Gcons_term, l_Gcons_rewired = [], [], [], []
for n in tqdm(range(Nrealiz)):
    l_Gcons_rndm.append(nx.read_gml(INPUT_PATH + 'random_graphs/l_Gwalk_rndm_' + str(n) + '._cons_' + loc + '.gml'))
    l_Gcons_rewired.append(nx.read_gml(INPUT_PATH + 'random_graphs/l_G_rewired_' + str(n) + '_cons_' + loc + '.gml'))
    l_Gcons_genesis.append(nx.read_gml(INPUT_PATH + 'random_graphs/l_Gwalk_genesis_' + str(n) + '_cons_' + loc + '.gml'))
    l_Gcons_term.append(nx.read_gml(INPUT_PATH + 'random_graphs/l_Gwalk_term_' + str(n) + '_cons_' + loc + '.gml'))


# %% REAL CATALOG

"""
Figure 3 S1: Clustering coefficients
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

# %% PARAMETERS
    
# Choose if multiple testing correciton should be applied
significance_mode = 'corrected'  # or 'raw'
# Choose panel
PANEL = 'b'
# Significance tests: centroid or head
l_signif_tests = [l_Gcons_rndm_head, l_Gcons_genesis_head, l_Gcons_term_head, l_Gcons_rewired_head]


# %% PANEL A -  REGULAR CLUSTERING COEFFICIENT

if PANEL == 'a':    
    # Prepare dataframe for plotting
    Gplot = Gcons.copy()
    l_hexID = list(nx.get_node_attributes(Gplot, "coordID").values())
    d_nodestr = gpd.GeoDataFrame({
        'nodestr': ana.clustering_coefficient(Gplot),
        'hex_idx': l_hexID,
        'geometry': [nplot.boundary_geom(hex_id) for hex_id in l_hexID]
    })
    
    # SIGNIFICANCE TESTING
    all_pvals = {}  # hex_idx -> list of 4 p-values (one per test)
    
    for ntest in tqdm(range(4)):
        l_Gnullm = l_signif_tests[ntest].copy()
    
        # Collect degree centralities per hex_idx
        nodestr_per_hex = defaultdict(list)
        geom_per_hex = {}
    
        for n in tqdm(range(Nrealiz), desc=f"Null model realizations for test {ntest+1}"):
            Gnullm = l_Gnullm[n]
            nodestr_arr = ana.clustering_coefficient(Gnullm)
            coordID_dict = nx.get_node_attributes(Gnullm, "coordID")
    
            for node, nodestr in zip(Gnullm.nodes(), nodestr_arr):
                hex_id = coordID_dict[node]
                nodestr_per_hex[hex_id].append(nodestr)
                if hex_id not in geom_per_hex:
                    geom_per_hex[hex_id] = nplot.boundary_geom(hex_id)
    
        # Merge real data with geometry
        d_nodestr_nullm = gpd.GeoDataFrame({
            'hex_idx': list(nodestr_per_hex.keys()),
            'geometry': [geom_per_hex[h] for h in nodestr_per_hex]
        })
    
        merged_df = d_nodestr.merge(
            d_nodestr_nullm[['hex_idx', 'geometry']],
            on='hex_idx', how='outer', suffixes=('_real', '_nullmodel')
        )
    
        teststring = f'pval_{ntest+1}'
    
        # Compute empirical one-sided p-values
        for idx, row in merged_df.iterrows():
            hex_id = row['hex_idx']
            nodestr_real = row['nodestr']
            null_values = nodestr_per_hex.get(hex_id, [])
    
            if not null_values:
                pval = np.nan
            else:
                # p-value = fraction of null >= real (empirical)
                pval = (np.sum(np.array(null_values) >= nodestr_real) + 1) / (len(null_values) + 1)
    
            merged_df.loc[idx, teststring] = pval
    
            # Store p-values in dictionary for later correction
            all_pvals.setdefault(hex_id, [np.nan]*4)
            all_pvals[hex_id][ntest] = pval
    
        # Merge p-values into main dataframe
        d_nodestr = d_nodestr.merge(merged_df[['hex_idx', teststring]], on='hex_idx', how='left')
    
    # Convert all_pvals into DataFrame for correction
    df_pvals = pd.DataFrame([
        {'hex_idx': h, 'pvals': pvals} for h, pvals in all_pvals.items()
    ])
    df_pvals[['pval_1', 'pval_2', 'pval_3', 'pval_4']] = pd.DataFrame(df_pvals['pvals'].to_list(), index=df_pvals.index)
    df_pvals = df_pvals.drop(columns='pvals')
    
    # FDR correction across the 4 tests per hex cell
    corrected_pvals = []
    for _, row in df_pvals.iterrows():
        pvals = row[['pval_1', 'pval_2', 'pval_3', 'pval_4']].values.astype(float)
    
        if np.all(np.isnan(pvals)):
            corrected_pvals.append([np.nan]*4)
        else:
            valid_mask = ~np.isnan(pvals)
            corrected = np.full(4, np.nan)
            corrected_vals = multipletests(pvals[valid_mask], alpha=alpha, method='fdr_bh')[1]
            corrected[valid_mask] = corrected_vals
            corrected_pvals.append(corrected)
    
    df_pvals[['pval_corr_1', 'pval_corr_2', 'pval_corr_3', 'pval_corr_4']] = pd.DataFrame(corrected_pvals, index=df_pvals.index)
    
    # Significance mask after correction
    for t in range(4):
        df_pvals[f'signif_corr_{t+1}'] = (df_pvals[f'pval_corr_{t+1}'] < alpha).astype(int)
    
    # Merge significance back into main dataframe
    d_nodestr = d_nodestr.merge(
        df_pvals[['hex_idx', 'signif_corr_1', 'signif_corr_2', 'signif_corr_3', 'signif_corr_4']],
        on='hex_idx', how='left'
    )


# %% PANEL B -  CYCLE CLUSTERING COEFFICIENT

if PANEL == 'b':    
    # Prepare dataframe for plotting
    Gplot = Gcons.copy()
    l_hexID = list(nx.get_node_attributes(Gplot, "coordID").values())
    d_nodestr = gpd.GeoDataFrame({
        'nodestr': ana.cycle_clustering_coefficient(Gplot),
        'hex_idx': l_hexID,
        'geometry': [nplot.boundary_geom(hex_id) for hex_id in l_hexID]
    })
    
    # SIGNIFICANCE TESTING
    all_pvals = {}  # hex_idx -> list of 4 p-values (one per test)
    
    for ntest in tqdm(range(4)):
        l_Gnullm = l_signif_tests[ntest].copy()
    
        # Collect degree centralities per hex_idx
        nodestr_per_hex = defaultdict(list)
        geom_per_hex = {}
    
        for n in tqdm(range(Nrealiz), desc=f"Null model realizations for test {ntest+1}"):
            Gnullm = l_Gnullm[n]
            nodestr_arr = ana.cycle_clustering_coefficient(Gnullm)
            coordID_dict = nx.get_node_attributes(Gnullm, "coordID")
    
            for node, nodestr in zip(Gnullm.nodes(), nodestr_arr):
                hex_id = coordID_dict[node]
                nodestr_per_hex[hex_id].append(nodestr)
                if hex_id not in geom_per_hex:
                    geom_per_hex[hex_id] = nplot.boundary_geom(hex_id)
    
        # Merge real data with geometry
        d_nodestr_nullm = gpd.GeoDataFrame({
            'hex_idx': list(nodestr_per_hex.keys()),
            'geometry': [geom_per_hex[h] for h in nodestr_per_hex]
        })
    
        merged_df = d_nodestr.merge(
            d_nodestr_nullm[['hex_idx', 'geometry']],
            on='hex_idx', how='outer', suffixes=('_real', '_nullmodel')
        )
    
        teststring = f'pval_{ntest+1}'
    
        # Compute empirical one-sided p-values
        for idx, row in merged_df.iterrows():
            hex_id = row['hex_idx']
            nodestr_real = row['nodestr']
            null_values = nodestr_per_hex.get(hex_id, [])
    
            if not null_values:
                pval = np.nan
            else:
                # p-value = fraction of null >= real (empirical)
                pval = (np.sum(np.array(null_values) >= nodestr_real) + 1) / (len(null_values) + 1)
    
            merged_df.loc[idx, teststring] = pval
    
            # Store p-values in dictionary for later correction
            all_pvals.setdefault(hex_id, [np.nan]*4)
            all_pvals[hex_id][ntest] = pval
    
        # Merge p-values into main dataframe
        d_nodestr = d_nodestr.merge(merged_df[['hex_idx', teststring]], on='hex_idx', how='left')
    
    # Convert all_pvals into DataFrame for correction
    df_pvals = pd.DataFrame([
        {'hex_idx': h, 'pvals': pvals} for h, pvals in all_pvals.items()
    ])
    df_pvals[['pval_1', 'pval_2', 'pval_3', 'pval_4']] = pd.DataFrame(df_pvals['pvals'].to_list(), index=df_pvals.index)
    df_pvals = df_pvals.drop(columns='pvals')
    
    # FDR correction across the 4 tests per hex cell
    corrected_pvals = []
    for _, row in df_pvals.iterrows():
        pvals = row[['pval_1', 'pval_2', 'pval_3', 'pval_4']].values.astype(float)
    
        if np.all(np.isnan(pvals)):
            corrected_pvals.append([np.nan]*4)
        else:
            valid_mask = ~np.isnan(pvals)
            corrected = np.full(4, np.nan)
            corrected_vals = multipletests(pvals[valid_mask], alpha=alpha, method='fdr_bh')[1]
            corrected[valid_mask] = corrected_vals
            corrected_pvals.append(corrected)
    
    df_pvals[['pval_corr_1', 'pval_corr_2', 'pval_corr_3', 'pval_corr_4']] = pd.DataFrame(corrected_pvals, index=df_pvals.index)
    
    # Significance mask after correction
    for t in range(4):
        df_pvals[f'signif_corr_{t+1}'] = (df_pvals[f'pval_corr_{t+1}'] < alpha).astype(int)
    
    # Merge significance back into main dataframe
    d_nodestr = d_nodestr.merge(
        df_pvals[['hex_idx', 'signif_corr_1', 'signif_corr_2', 'signif_corr_3', 'signif_corr_4']],
        on='hex_idx', how='left'
    )




# %% PLOTTING


# Colormaps
cmap0, cmap1, cmap2, cmap3, cmap4 = cm.grayC_r, cm.devon_r, CMAP.Purples, CMAP.Oranges, cm.imola#CMAP.Greens
vmin, vmax = np.nanmin(d_nodestr.nodestr), np.nanquantile(d_nodestr.nodestr, .99) # truncating

# POST-PROCESSING
d_nodestr.dropna(subset=['nodestr'], inplace=True)
d_nodestr = nplot.split_hexagons(d_nodestr)

# Choose significance columns depending on mode
if significance_mode == 'corrected':
    s1, s2, s3, s4 = 'signif_corr_1', 'signif_corr_2', 'signif_corr_3', 'signif_corr_4'
else:
    s1, s2, s3, s4 = 'pval_1', 'pval_2', 'pval_3', 'pval_4'
    # Convert raw p-values to binary significance
    d_nodestr[s1] = (d_nodestr[s1] < alpha).astype(int)
    d_nodestr[s2] = (d_nodestr[s2] < alpha).astype(int)
    d_nodestr[s3] = (d_nodestr[s3] < alpha).astype(int)
    d_nodestr[s4] = (d_nodestr[s4] < alpha).astype(int)

# Further processing for populations passing different numbers of tests
pop0 = d_nodestr[~((d_nodestr[s1] == 1) |
                   ((d_nodestr[[s1, s4]] == 1).sum(axis=1) == 2) |
                   ((d_nodestr[[s1, s4, s2]] == 1).sum(axis=1) == 3) |
                   ((d_nodestr[[s1, s4, s2, s3]] == 1).sum(axis=1) == 4))]
pop1 = d_nodestr[d_nodestr[s1] == 1]
pop2 = d_nodestr[(d_nodestr[[s1, s4]] == 1).sum(axis=1) == 2]
pop3 = d_nodestr[(d_nodestr[[s1, s4, s2]] == 1).sum(axis=1) == 3]
pop4 = d_nodestr[(d_nodestr[[s1, s4, s2, s3]] == 1).sum(axis=1) == 4]
# set geometry attributes
pop0 = pop0.set_geometry('geometry')
pop1 = pop1.set_geometry('geometry')
pop2 = pop2.set_geometry('geometry')
pop3 = pop3.set_geometry('geometry')
pop4 = pop4.set_geometry('geometry')



# F I G U R E 
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.EqualEarth()})
ax.set_global()
ax.add_feature(cfeature.COASTLINE, color='black')

# Normalization for color scaling
norm = Normalize(vmin=vmin, vmax=vmax)#LogNorm(vmin=vmin+1e-9, vmax=vmax)#

# Plot each population with its corresponding colormap
if not pop0.empty:
    pop0.plot(column='nodestr', cmap=cmap0, ax=ax,
               transform=ccrs.PlateCarree(), norm=norm, edgecolor='black', linewidth=0, legend=False, alpha=.85)

if not pop1.empty:
    pop1.plot(column='nodestr', cmap=cmap1, ax=ax,
               transform=ccrs.PlateCarree(), norm=norm, edgecolor='black', linewidth=0, legend=False, alpha=.85)

if not pop2.empty:
    pop2.plot(column='nodestr', cmap=cmap2, ax=ax,
               transform=ccrs.PlateCarree(), norm=norm, edgecolor='black', linewidth=0, legend=False, alpha=.9)

if not pop3.empty:
    pop3.plot(column='nodestr', cmap=cmap3, ax=ax,
               transform=ccrs.PlateCarree(), norm=norm, edgecolor='black', linewidth=0, legend=False, alpha=.95)

if not pop4.empty:
    pop4.plot(column='nodestr', cmap=cmap4, ax=ax,
               transform=ccrs.PlateCarree(), norm=norm, edgecolor='black', linewidth=.8, legend=False)

plt.show()
plt.savefig(OUTPUT_PATH + '/Fig3S1b.png', dpi=300, bbox_inches='tight')


# SEPARATE COLORBAR PLOT
cbar_fig, cbar_axs = plt.subplots(1, 5, figsize=(25, 0.4))
fs = 16
# Create a colorbar for each colormap
cbar0 = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap0, norm=norm), cax=cbar_axs[0], orientation='horizontal')
cbar0.set_label('no test passed', color='black', fontsize=fs)
cbar0.ax.tick_params(labelsize=fs)  # Set tick label size

cbar1 = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap1, norm=norm), cax=cbar_axs[1], orientation='horizontal')
cbar1.set_label('FRW', color='black', fontsize=fs)
cbar1.ax.tick_params(labelsize=fs)  # Set tick label size

cbar2 = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap2, norm=norm), cax=cbar_axs[2], orientation='horizontal')
cbar2.set_label('FRW+GCW', color='black', fontsize=fs)
cbar2.ax.tick_params(labelsize=fs)  # Set tick label size

cbar3 = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap3, norm=norm), cax=cbar_axs[3], orientation='horizontal')
cbar3.set_label('FRW+GCW+TCW', color='black', fontsize=fs)
cbar3.ax.tick_params(labelsize=fs)  # Set tick label size

cbar4 = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap4, norm=norm), cax=cbar_axs[4], orientation='horizontal')
cbar4.set_label('all tests passed', color='black', fontsize=fs)
cbar4.ax.tick_params(labelsize=fs)  # Set tick label size
# Adjust layout to avoid overlap
plt.subplots_adjust(wspace=0.1)
plt.show()
plt.savefig(OUTPUT_PATH + '/Fig3S1b_cbar.png', dpi=300, bbox_inches='tight')
