# Copyright (C) 2025 by
# Tobias Braun

#------------------ PATH ---------------------------#
# working directory
import sys
WDPATH = "/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/paper/scripts/ARnetlab"
sys.path.insert(0, WDPATH)
# input and output
INPUT_PATH = '/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/paper/data/'
OUTPUT_PATH = '/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/paper/data/output/'


# %% IMPORT MODULES

# standard packages
import numpy as np
import pandas as pd
import matplotlib as mpl
#mpl.use('Agg')
from matplotlib import pyplot as plt

# specific packages
import networkx as nx
from tqdm import tqdm
import geopandas as gpd
from collections import defaultdict
import xarray as xr
from statsmodels.stats.multitest import multipletests

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
loc = 'centroid'


l_Gcons_rndm, l_Gcons_genesis, l_Gcons_term, l_Gcons_rewired = [], [], [], []
for n in tqdm(range(Nrealiz)):
    l_Gcons_rndm.append(nx.read_gml(INPUT_PATH + 'random_graphs/l_Gwalk_rndm_' + str(n) + '._cons_' + loc + '.gml'))
    l_Gcons_rewired.append(nx.read_gml(INPUT_PATH + 'random_graphs/l_G_rewired_' + str(n) + '_cons_' + loc + '.gml'))
    l_Gcons_genesis.append(nx.read_gml(INPUT_PATH + 'random_graphs/l_Gwalk_genesis_' + str(n) + '_cons_' + loc + '.gml'))
    l_Gcons_term.append(nx.read_gml(INPUT_PATH + 'random_graphs/l_Gwalk_term_' + str(n) + '_cons_' + loc + '.gml'))


# Parameters
loc = 'head'
l_Gcons_rndm_head, l_Gcons_rewired_head, l_Gcons_genesis_head, l_Gcons_term_head = [], [], [], []
for n in tqdm(range(Nrealiz)):
    l_Gcons_rndm_head.append(nx.read_gml(INPUT_PATH + 'random_graphs/l_Gwalk_rndm_' + str(n) + '._cons_' + loc + '.gml'))
    l_Gcons_rewired_head.append(nx.read_gml(INPUT_PATH + 'random_graphs/l_G_rewired_' + str(n) + '_cons_' + loc + '.gml'))
    l_Gcons_genesis_head.append(nx.read_gml(INPUT_PATH + 'random_graphs/l_Gwalk_genesis_' + str(n) + '_cons_' + loc + '.gml'))
    l_Gcons_term_head.append(nx.read_gml(INPUT_PATH + 'random_graphs/l_Gwalk_term_' + str(n) + '_cons_' + loc + '.gml'))


# %% REAL CATALOG

"""
Figure 3: AR hubs by means of node strength, divergence and PageRank.
"""

## Network parameters
# spatiotemporal extent
T = None # no clipping
X = 'global'
# nodes
res = 2 # h3 system, corresponds to closest resolution to 2 degrees
grid_type = 'hexagonal'
# edges
weighing = 'absolute'
self_links = False
weighted = True
directed = True
ndec = 8.4 # number of decades
eps = int(ndec) # threshold: at least 1AR/decade
thresh = 1.25*eps
# conditioning
cond = None # any network conditioning
LC_cond = None # lifecycle conditioning
# analysis
alpha = 0.1
fdamp = .85
norm = True # normalization in PageRank


# Pick locator depending on panel
loc = 'head'


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
PANEL = 'a'
# Significance tests: centroid or head
l_signif_tests = [l_Gcons_rndm_head, l_Gcons_genesis_head, l_Gcons_term_head, l_Gcons_rewired_head]


# %% PANEL A -  NODE STRENGTH

if PANEL == 'a':    
    # Prepare dataframe for plotting
    Gplot = Gcons.copy()
    l_hexID = list(nx.get_node_attributes(Gplot, "coordID").values())
    d_nodestr = gpd.GeoDataFrame({
        'nodestr': ana.degree_centrality(Gplot),
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
            nodestr_arr = ana.degree_centrality(Gnullm)
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

    # SAVE
    d_nodestr = gpd.GeoDataFrame(d_nodestr, geometry='geometry', crs='EPSG:4326')  # or whatever CRS you're using
    d_nodestr.to_file(OUTPUT_PATH + "nodestr_head_consensus.gpkg", layer='nodestr', driver="GPKG")



# %% PANEL B - DIVERGENCE

elif PANEL == 'b':
    # --- Generate dataframe for real data ---
    Gplot = Gcons.copy()
    l_hexID = list(nx.get_node_attributes(Gplot, "coordID").values())
    d_ndiv = gpd.GeoDataFrame({
        'ndiv': ana.divergence(Gplot),  
        'hex_idx': l_hexID,
        'geometry': [nplot.boundary_geom(hex_id) for hex_id in l_hexID]
    })
    
    # --- SIGNIFICANCE TESTING ---
    all_pvals = {}  # hex_idx -> list of 4 p-values
    
    for ntest in tqdm(range(4), desc=f"Null model realizations for test {ntest+1}"):
        l_Gnullm = l_signif_tests[ntest]
        ndiv_per_hex = defaultdict(list)
        geom_per_hex = {}
    
        for n in tqdm(range(Nrealiz)):
            Gnullm = l_Gnullm[n]
            ndiv_arr = ana.divergence(Gnullm)
            coordID_dict = nx.get_node_attributes(Gnullm, "coordID")
    
            for node, ndiv_val in zip(Gnullm.nodes(), ndiv_arr):
                hex_id = coordID_dict[node]
                ndiv_per_hex[hex_id].append(ndiv_val)
                if hex_id not in geom_per_hex:
                    geom_per_hex[hex_id] = nplot.boundary_geom(hex_id)
    
        # Compute empirical p-values
        teststring = f'pval_{ntest+1}'
        d_ndiv[teststring] = np.nan
    
        for idx, row in d_ndiv.iterrows():
            hex_id = row['hex_idx']
            ndiv_real = row['ndiv']
            null_vals = ndiv_per_hex.get(hex_id, [])
    
            if not null_vals or pd.isna(ndiv_real):
                pval = np.nan
            else:
                null_vals_arr = np.array(null_vals)
                mean_null = np.mean(null_vals_arr)
                # Two-sided empirical p-value using mean-centered null
                pval = (np.sum(np.abs(null_vals_arr - mean_null) >= abs(ndiv_real - mean_null)) + 1) / (len(null_vals_arr) + 1)
    
            d_ndiv.at[idx, teststring] = pval
            all_pvals.setdefault(hex_id, [np.nan]*4)
            all_pvals[hex_id][ntest] = pval
    
    # --- MULTIPLE TESTING CORRECTION per hex cell across 4 tests ---
    df_pvals = pd.DataFrame([{'hex_idx': h, 'pvals': pvals} for h, pvals in all_pvals.items()])
    df_pvals[['pval_1', 'pval_2', 'pval_3', 'pval_4']] = pd.DataFrame(df_pvals['pvals'].to_list(), index=df_pvals.index)
    df_pvals.drop(columns='pvals', inplace=True)
    
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
    
    # Determine significance after correction
    for t in range(4):
        df_pvals[f'signif_corr_{t+1}'] = (df_pvals[f'pval_corr_{t+1}'] < alpha).astype(int)
    
    # Merge corrected significance back into d_ndiv
    d_ndiv = d_ndiv.merge(
        df_pvals[['hex_idx', 'signif_corr_1', 'signif_corr_2', 'signif_corr_3', 'signif_corr_4']],
        on='hex_idx', how='left'
    )
    
    
    # SAVE
    d_ndiv = gpd.GeoDataFrame(d_ndiv, geometry='geometry', crs='EPSG:4326')  # or whatever CRS you're using
    d_ndiv.to_file(OUTPUT_PATH + "divergence_head_consensus.gpkg", layer='ndiv', driver="GPKG")




# %% PANEL C - PAGERANK

#""" WORKS WITH AR HEADS!!! """ 

elif PANEL == 'c':
    """
    Key Takeaways
    PageRank emphasizes persistent, flow-retaining structures rather than just termination points.
    Your high PageRank regions align with moisture accumulation, recirculation, or persistent guidance by winds/topography.
    Degree centrality is more about sheer number of connections, while PageRank tells a deeper dynamical story about moisture transport stability and trapping.
    """
    # Parameters
    #fdamp = 0.9
    
    
    # Prepare dataframe for plotting (real graph)
    Gplot = Gcons.copy()
    l_hexID = list(nx.get_node_attributes(Gplot, "coordID").values())
    pagerank_real = ana.pagerank(Gplot, fdamp, normalised=norm)
    d_pagernk = gpd.GeoDataFrame({
        'pagernk': pagerank_real,
        'hex_idx': l_hexID,
        'geometry': [nplot.boundary_geom(hex_id) for hex_id in l_hexID]
    })
    
    # SIGNIFICANCE TESTING
    all_pvals = {}  # hex_idx -> list of 4 p-values
    
    for ntest in tqdm(range(4), desc="Null model realizations"):
        l_Gnullm = l_signif_tests[ntest]
        
        # Collect pagerank values per hex_idx
        pagernk_per_hex = defaultdict(list)
        geom_per_hex = {}
    
        for n in tqdm(range(Nrealiz), desc=f"Test {ntest+1} realizations"):
            Gnullm = l_Gnullm[n]
            pagerank_null = ana.pagerank(Gnullm, fdamp, normalised=norm)
            coordID_dict = nx.get_node_attributes(Gnullm, "coordID")
    
            for node, pr_value in zip(Gnullm.nodes(), pagerank_null):
                hex_id = coordID_dict[node]
                pagernk_per_hex[hex_id].append(pr_value)
                if hex_id not in geom_per_hex:
                    geom_per_hex[hex_id] = nplot.boundary_geom(hex_id)
    
        # Prepare merged dataframe
        d_null_geom = gpd.GeoDataFrame({
            'hex_idx': list(pagernk_per_hex.keys()),
            'geometry': [geom_per_hex[h] for h in pagernk_per_hex]
        })
    
        merged_df = d_pagernk.merge(d_null_geom, on='hex_idx', how='outer', suffixes=('_real', '_nullmodel'))
        teststring = f'pval_{ntest+1}'
    
        for idx, row in merged_df.iterrows():
            hex_id = row['hex_idx']
            pr_real = row['pagernk']
            null_values = pagernk_per_hex.get(hex_id, [])
    
            if not null_values:
                pval = np.nan
            else:
                pval = (np.sum(np.array(null_values) >= pr_real) + 1) / (len(null_values) + 1)
    
            merged_df.loc[idx, teststring] = pval
            all_pvals.setdefault(hex_id, [np.nan]*4)
            all_pvals[hex_id][ntest] = pval
    
        d_pagernk = d_pagernk.merge(merged_df[['hex_idx', teststring]], on='hex_idx', how='left')
    
    # Convert all pvals to DataFrame
    df_pvals = pd.DataFrame([
        {'hex_idx': h, 'pvals': pvals} for h, pvals in all_pvals.items()
    ])
    df_pvals[['pval_1', 'pval_2', 'pval_3', 'pval_4']] = pd.DataFrame(df_pvals['pvals'].to_list(), index=df_pvals.index)
    df_pvals.drop(columns='pvals', inplace=True)
    
    # FDR correction
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
    
    # Binary significance
    for t in range(4):
        df_pvals[f'signif_corr_{t+1}'] = (df_pvals[f'pval_corr_{t+1}'] < alpha).astype(int)
    
    # Merge back
    d_pagernk = d_pagernk.merge(
        df_pvals[['hex_idx', 'signif_corr_1', 'signif_corr_2', 'signif_corr_3', 'signif_corr_4']],
        on='hex_idx', how='left'
    )
    
    # SAVE
    d_pagernk = gpd.GeoDataFrame(d_pagernk, geometry='geometry', crs='EPSG:4326')  # or whatever CRS you're using
    d_pagernk.to_file(OUTPUT_PATH + "pagerank_head_consensus_085.gpkg", layer='pagernk', driver="GPKG")
    
    
