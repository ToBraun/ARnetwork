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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap
import random
import time
from scipy.stats import linregress, t
from collections import defaultdict, Counter


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
eps = 8 
thresh = 1.25*eps
# conditioning
cond = None # any network conditioning
LC_cond = None # lifecycle conditioning


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
Greal = nx.relabel_nodes(G, dict(zip(list(G.nodes()), range(len(list(G.nodes()))))))

# Random networks
Nrealiz = 200
Nrnets = 4
l_Gcons_rndm, l_Gcons_genesis, l_Gcons_term, l_Gcons_rewired = [], [], [], []
for n in tqdm(range(Nrealiz)):
    l_Gcons_rndm.append(nx.read_gml(INPUT_PATH + 'random_graphs/l_Gwalk_rndm_' + str(n) + '._cons_' + loc + '.gml'))
    l_Gcons_rewired.append(nx.read_gml(INPUT_PATH + 'random_graphs/l_G_rewired_' + str(n) + '_cons_' + loc + '.gml'))
    l_Gcons_genesis.append(nx.read_gml(INPUT_PATH + 'random_graphs/l_Gwalk_genesis_' + str(n) + '_cons_' + loc + '.gml'))
    l_Gcons_term.append(nx.read_gml(INPUT_PATH + 'random_graphs/l_Gwalk_term_' + str(n) + '_cons_' + loc + '.gml'))


# %% NON-HIERARCHICAL COMMUNITIES


# PARAMETERS
use_node_weights_as_flow = True
Nmin = 15
n_seeds = 100
seeds = np.random.uniform(0, 1000, n_seeds)  
# Apprehend all networks into a single list (messy)
l_allGs =  [Greal] + l_Gcons_rndm + l_Gcons_genesis + l_Gcons_term + l_Gcons_rewired


all_ami_scores = []  
all_ari_scores = []
all_flowrat_values = []
for ng in tqdm(range(len(l_allGs))):  # 0 = real graph, rest are surrogates
    G0 = l_allGs[ng].copy()
    
    # Complete nodes (add disconnected ones)
    Gc = artn.complete_nodes(G0, 2)

    ### Relabel the nodes, starting from 0
    G = nx.relabel_nodes(Gc, dict(zip(list(Gc.nodes()), range(len(list(Gc.nodes()))))))

    # Initialize
    d_coordID = nx.get_node_attributes(G, "coordID")
    node_list = list(G.nodes())
    d_coordID_sub = {node: d_coordID[node] for node in node_list}
    d_node_comm = pd.DataFrame()
    d_node_comm['hex_id'] = [d_coordID_sub[node] for node in node_list]
    l_flowrats = []
    # Iterate over seeds (stability)
    for i, seed in tqdm(enumerate(seeds)):
        communities, d_flow = ana.detect_non_hierarchical_communities(
            G,
            use_node_weights_as_flow=use_node_weights_as_flow,
            filename=None,
            return_flows=True,
            seed=seed
        )

        # Filter small communities
        community_sizes = {}
        for node, comm in communities.items():
            community_sizes[comm] = community_sizes.get(comm, 0) + 1
        filtered_communities = {
            node: (comm if community_sizes[comm] >= Nmin else -999)
            for node, comm in communities.items()
        }
        # stack
        colname = f"community_{i+1}"
        d_node_comm[colname] = [filtered_communities[node] for node in node_list]

        # Flow ratios
        d_flowrat = ana.compute_community_flow_ratio(d_flow, filtered_communities)
        l_flowrats.append(list(d_flowrat.values()))

    # Combine all flowrat values across seeds
    a_flowrats = np.hstack(l_flowrats)
    a_flowrats = a_flowrats[~np.isnan(a_flowrats)]
    all_flowrat_values.append(a_flowrats)

    # Compute AMI and ARI between all pairs of seeds
    n_seeds = len(seeds)
    ami_scores = []
    ari_scores = []
    for i in tqdm(range(n_seeds)):
        for j in range(i + 1, n_seeds):
            labels_i = d_node_comm[f"community_{i+1}"].values
            labels_j = d_node_comm[f"community_{j+1}"].values

            valid_mask = (labels_i != -999) & (labels_j != -999)
            filtered_i = labels_i[valid_mask]
            filtered_j = labels_j[valid_mask]

            if len(filtered_i) > 0:
                ami_scores.append(adjusted_mutual_info_score(filtered_i, filtered_j))
                ari_scores.append(adjusted_rand_score(filtered_i, filtered_j))

    all_ami_scores.append(ami_scores)
    all_ari_scores.append(ari_scores)


ami_real = all_ami_scores[0]
ari_real = all_ari_scores[0]
flowrat_real = all_flowrat_values[0]

# Split random results into their respective groups (assuming each group has 20 graphs)
ami_rndm_groups = [all_ami_scores[1+i*Nrealiz:1+(i+1)*Nrealiz] for i in range(Nrnets)]
ari_rndm_groups = [all_ari_scores[1+i*Nrealiz:1+(i+1)*Nrealiz] for i in range(Nrnets)]
flowrat_rndm_groups = [np.hstack(all_flowrat_values[1+i*Nrealiz:1+(i+1)*Nrealiz]) for i in range(Nrnets)]

# SAVE
np.save(INPUT_PATH + 'output/figs53_performance_boxplots_AMI.npy', ami_rndm_groups)
all_ami_scores.shape

# %% BOXPLOT


%matplotlib

labels = ['FRW', 'GCW', 'TCW', 'RWG']
colors = ['darkorange', 'purple', 'royalblue', 'darkgreen']

fig, axes = plt.subplots(1, 3, figsize=(9, 2.5))

# === AMI ===
axes[0].boxplot(ami_real, patch_artist=True,
                boxprops=dict(facecolor='slategrey', alpha=0.7),
                medianprops=dict(color='black', linewidth=2))
for group, label, color in zip(ami_rndm_groups, labels, colors):
    flat = [score for sublist in group for score in sublist]
    q95 = np.percentile(flat, 95)
    axes[0].axhline(q95, linestyle='--', color=color, label=f'{label}', linewidth=2)
axes[0].set_ylabel('Adjusted Mutual Information')
axes[0].grid(True, alpha=0.3)

# === ARI ===
axes[1].boxplot(ari_real, patch_artist=True,
                boxprops=dict(facecolor='slategrey', alpha=0.7),
                medianprops=dict(color='black', linewidth=2))
for group, label, color in zip(ari_rndm_groups, labels, colors):
    flat = [score for sublist in group for score in sublist]
    q95 = np.percentile(flat, 95)
    axes[1].axhline(q95, linestyle='--', color=color, label=f'{label}', linewidth=2)
axes[1].set_ylabel('Adjusted Rand Score')
axes[1].grid(True, alpha=0.3)

# === Flow Ratios ===
axes[2].boxplot(flowrat_real, patch_artist=True,
                boxprops=dict(facecolor='slategrey', alpha=0.7),
                medianprops=dict(color='black', linewidth=2))
for group, label, color in zip(flowrat_rndm_groups, labels, colors):
    flow_vals = group[group > 0]
    if len(flow_vals) > 0:
        q95 = np.percentile(flow_vals, 95)
        axes[2].axhline(q95, linestyle='--', color=color, label=f'{label}', linewidth=2)
axes[2].set_yscale('log')
axes[2].set_ylabel('Flow Ratio (log scale)')
axes[2].grid(True, alpha=0.3, which='both')

# Remove x-tick labels
for ax in axes:
    ax.set_xticklabels([])

# Create single shared legend on top# Get handles and labels from any axis (e.g., first one)
handles, labels_legend = axes[0].get_legend_handles_labels()

# Create legend with title
fig.legend(handles, labels_legend,
           title='Null Model (95%-quantile):',
           loc='upper center', ncol=len(labels_legend),
           bbox_to_anchor=(0.5, 1.12),  # adjust vertical position
           frameon=False, handletextpad=1.5)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for legend
plt.show()
plt.savefig(OUTPUT_PATH + "Fig5S3_boxplots.pdf", dpi=300, bbox_inches='tight')






# %% OVERALL COMPARISON/STABILITY

n_seeds = len(seeds)
ami_scores = []
ari_scores = []

for i in tqdm(range(n_seeds)):
    for j in range(i+1, n_seeds):  # upper triangle only, no repeats
        labels_i = d_node_comm[f"community_{i+1}"].values
        labels_j = d_node_comm[f"community_{j+1}"].values
        
        # Filter nodes where either label is -999
        valid_mask = (labels_i != -999) & (labels_j != -999)
        filtered_i = labels_i[valid_mask]
        filtered_j = labels_j[valid_mask]
        
        # Compute metrics only if some nodes remain
        if len(filtered_i) > 0:
            ami_scores.append(adjusted_mutual_info_score(filtered_i, filtered_j))
            ari_scores.append(adjusted_rand_score(filtered_i, filtered_j))

# Calculate statistics
ami_avg = np.mean(ami_scores)
ami_var = np.var(ami_scores)
ari_avg = np.mean(ari_scores)
ari_var = np.var(ari_scores)
flowrat_avg = np.mean(a_flowrats)
flowrat_var = np.var(a_flowrats)

print(f"AMI average: {ami_avg:.4f} ± {np.sqrt(ami_var):.4f}")
print(f"ARI average: {ari_avg:.4f} ± {np.sqrt(ari_var):.4f}")
print(f"FLOWRATIO average: {flowrat_avg:.4f} ± {np.sqrt(flowrat_var):.4f}")


# %% CREATE BOX PLOTS


# Avoid log of zero or negative (in case of NaNs or zeros)
valid_flowrats = np.array(a_flowrats)
valid_flowrats = valid_flowrats[np.isfinite(valid_flowrats) & (valid_flowrats > 0)]

median_flowrat = np.median(valid_flowrats)
iqr_flowrat = np.percentile(valid_flowrats, 75) - np.percentile(valid_flowrats, 25)


fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# AMI Box Plot
axes[0].boxplot(ami_scores, patch_artist=True, 
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
axes[0].set_ylabel('AMI Score')
axes[0].grid(True, alpha=0.3)
axes[0].text(0.02, 0.98, f'Mean: {ami_avg:.4f}\nStd: {np.sqrt(ami_var):.4f}', 
             transform=axes[0].transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ARI Box Plot
axes[1].boxplot(ari_scores, patch_artist=True,
                boxprops=dict(facecolor='lightgreen', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
axes[1].set_ylabel('ARI Score')
axes[1].grid(True, alpha=0.3)
axes[1].text(0.02, 0.98, f'Mean: {ari_avg:.4f}\nStd: {np.sqrt(ari_var):.4f}', 
             transform=axes[1].transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Flow Ratio Box Plot
axes[2].boxplot(valid_flowrats, patch_artist=True,
                boxprops=dict(facecolor='lightcoral', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
axes[2].set_yscale('log')
axes[2].set_ylabel('Flow Ratio (log scale)')
axes[2].grid(True, alpha=0.3, which='both')
axes[2].text(0.02, 0.98, f'Median: {median_flowrat:.4f}\nIQR: {iqr_flowrat:.4f}',
             transform=axes[2].transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()





# %% ALTERNATIVE: Using Seaborn for more aesthetic plots
# Create a combined dataframe for easier plotting
plot_data = pd.DataFrame({
    'AMI': ami_scores,
    'ARI': ari_scores
})

# Add compression data (repeat values to match AMI/ARI length if needed)
if len(compression_values) < len(ami_scores):
    # If we have fewer compression values, we might want to handle this differently
    # For now, let's create a separate plot
    pass

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Seaborn box plots
sns.boxplot(data=[ami_scores], ax=axes[0,0], palette=['lightblue'])
axes[0,0].set_title('AMI Stability')
axes[0,0].set_ylabel('AMI Score')
axes[0,0].set_xticklabels(['AMI'])

sns.boxplot(data=[ari_scores], ax=axes[0,1], palette=['lightgreen'])
axes[0,1].set_title('ARI Stability')
axes[0,1].set_ylabel('ARI Score')
axes[0,1].set_xticklabels(['ARI'])

sns.boxplot(data=[compression_values], ax=axes[1,0], palette=['lightcoral'])
axes[1,0].set_title('Compression Ratio')
axes[1,0].set_ylabel('Compression')
axes[1,0].set_xticklabels(['Compression'])

# Combined violin plot for AMI and ARI
combined_data = pd.melt(plot_data, var_name='Metric', value_name='Score')
sns.violinplot(data=combined_data, x='Metric', y='Score', ax=axes[1,1], palette=['lightblue', 'lightgreen'])
axes[1,1].set_title('AMI vs ARI Distribution')
axes[1,1].set_ylabel('Score')

plt.tight_layout()
plt.show()







