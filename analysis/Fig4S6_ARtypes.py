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
from numpy.polynomial.polynomial import Polynomial
import pandas as pd
import matplotlib as mpl
#mpl.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as mplcm
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors
from matplotlib.colors import to_rgba, ListedColormap, Normalize, LogNorm
from matplotlib.collections import PolyCollection, LineCollection
import time
from scipy.stats import ks_2samp, mannwhitneyu, wasserstein_distance
import seaborn as sns
from scipy.interpolate import splprep, splev
from scipy.stats import linregress


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

def remove_consecutive_duplicates(lons, lats):
    filtered = [(lons[0], lats[0])]
    for lon, lat in zip(lons[1:], lats[1:]):
        if (lon, lat) != filtered[-1]:
            filtered.append((lon, lat))
    return zip(*filtered)

def is_in_hemisphere(path, hemisphere='north'):
    coords = [h3.h3_to_geo(h) for h in path]
    lats, _ = zip(*coords)
    mean_lat = np.mean(lats)
    return mean_lat >= 15 if hemisphere == 'north' else mean_lat < -15


# %% LOAD DATA

# PIKART
d_ars_pikart = pd.read_pickle(INPUT_PATH + 'PIKART' + '_hex.pkl')
# tARget v4
d_ars_target = pd.read_pickle(INPUT_PATH + 'target' + '_hex.pkl')
# we also import the untransformed one as it contains the lf_lons needed here (only)
d_ars_target_nohex = pd.read_pickle(INPUT_PATH + 'tARget_globalARcatalog_ERA5_1940-2023_v4.0_converted.pkl')
d_ars_target['lf_lon'] = d_ars_target_nohex['lf_lon']


# %% CONDITIONAL AR NETWORK: SEASONAL

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
eps = np.array([5,5,5,5])
thresh = np.array(1.25*eps, dtype=int)
# conditioning
LC_cond = None # lifecycle conditioning
cond = 'season'


# PIKART
ARcat = d_ars_pikart.copy()
ARcat['time'] = pd.to_datetime(ARcat['time']).dt.floor('D')
ARcat['lf_lon'] = ARcat['lf_lon'].replace(0, np.nan)
# Convert landfall latitudes and longitudes to hexagon index
l_arcats_pikart, d_coord_dict = artn.preprocess_catalog(ARcat, T, loc, grid_type, X, res, cond, LC_cond)
Apik, t_idx_pikart, t_hexidx_pikart, t_ivt_pikart, t_grid_pikart = artn.generate_transport_matrix(l_arcats_pikart, grid_type, d_coord_dict, LC_cond)
Gpikart = artn.generate_network(Apik, t_grid_pikart, weighted, directed, eps, self_links, weighing)
# SEASONAL: only look at DJF and JJA
ns1, ns2 = 2, 3
Gpikart, t_idx_pikart, t_ivt_pikart = [Gpikart[ns1], Gpikart[ns2]], [t_idx_pikart[ns1], t_idx_pikart[ns2]], [t_ivt_pikart[ns1], t_ivt_pikart[ns2]]

# tARget
ARcat = d_ars_target.copy()
ARcat['time'] = pd.to_datetime(ARcat['time']).dt.floor('D')
ARcat['lf_lon'] = ARcat['lf_lon'].replace(0, np.nan)
# Convert landfall latitudes and longitudes to hexagon index
l_arcats_target, d_coord_dict = artn.preprocess_catalog(ARcat, T, loc, grid_type, X, res, cond, LC_cond)
Atarget, t_idx_target, t_hexidx_target, t_ivt_target, t_grid_target = artn.generate_transport_matrix(l_arcats_target, grid_type, d_coord_dict, LC_cond)
Gtarget = artn.generate_network(Atarget, t_grid_target, weighted, directed, eps, self_links, weighing)
# SEASONAL: only look at DJF and JJA
Gtarget, t_idx_target, t_ivt_target = [Gtarget[ns1], Gtarget[ns2]], [t_idx_target[ns1], t_idx_target[ns2]], [t_ivt_target[ns1], t_ivt_target[ns2]]
Lp = len(Gtarget)

# Check: did we set the thresholds right?
Gpikart[1].number_of_edges()
   

# %% EDGE BETWEENNESS: EBC CONSENSUS

# LOOP OVER REGIMES
l_Gbetw_phases = []
for nph in tqdm(range(Lp)):
    l_Gs = [Gpikart[nph], Gtarget[nph]]
    l_Gbetw_cat = []
    for n in range(2):
        # invert weights for shortest path identification
        G = ana.invert_weights(l_Gs[n])
        # EBC
        d_ebetw = nx.edge_betweenness_centrality(G)
        nx.set_edge_attributes(G, d_ebetw, "edge_betweenness")
        l_Gbetw_cat.append(G)
    l_Gbetw_phases.append(l_Gbetw_cat)

# Averaging of edge betweenness, edge weights and edge signs:
l_Gcons0 = [artn.average_networks_by_attributes(l_Gbetw_phases[nph][0], l_Gbetw_phases[nph][1], attr_name='edge_betweenness') for nph in range(Lp)]
# Complete nodes for plotting
l_Gcons = [artn.complete_nodes(l_Gcons0[nph], res) for nph in range(Lp)]



# %% STRAY INDEX

#low_ebc_quantile, high_ebc_quantile = .2, .8#0.2, 0.8
low_stray_quantile, high_stray_quantile = .33, .66#0.2, 0.5
#low_slope_quantile, high_slope_quantile = 0.33, 0.66
Lmin = 4

# Set to DJF (ns=3)
ns = 3

## EXTRACT tracks
# PIKART
l_artracks_pik = [group for name, group in l_arcats_pikart[ns].groupby('trackid')]
a_meanivt_pik = np.hstack([l_artracks_pik[i].mean_ivt.mean() for i in tqdm(range(len(l_artracks_pik)))])
a_ltime_pik = np.hstack([l_artracks_pik[i].shape[0] for i in tqdm(range(len(l_artracks_pik)))])/4
observed_paths_pik = [l_artracks_pik[i].coord_idx.values for i in range(len(l_artracks_pik))]
# tARget
l_artracks_target = [group for name, group in l_arcats_target[ns].groupby('trackid')]
a_meanivt_target = np.hstack([l_artracks_target[i].mean_ivt.mean() for i in tqdm(range(len(l_artracks_target)))])
a_ltime_target = np.hstack([l_artracks_target[i].shape[0] for i in tqdm(range(len(l_artracks_target)))])/4
observed_paths_target = [l_artracks_target[i].coord_idx.values for i in range(len(l_artracks_target))]


# network: DJF (ns=3)
G = l_Gcons[ns-2].copy()

label1, label2, label3 = 'conformists', 'straddlers', 'strays'

# Classification
classes_pikart = ana.classify_trajectories_simple(
    observed_paths_pik, G, 
    ebc_attr='edge_betweenness',
    min_length=4,
    low_stray_quantile=low_stray_quantile,
    high_stray_quantile=high_stray_quantile,
    scale='log')


classes_target = ana.classify_trajectories_simple(
    observed_paths_target, G, 
    ebc_attr='edge_betweenness',
    min_length=4,
    low_stray_quantile=low_stray_quantile,
    high_stray_quantile=high_stray_quantile,
    scale='log')


# Unpack
## PIKART
l_stray_indices_pikart, l_stray_scores_pikart = classes_pikart[label3]['indices'], classes_pikart[label3]['stray_scores']
l_conform_indices_pikart, l_conform_stray_scores_pikart = classes_pikart[label1]['indices'], classes_pikart[label1]['stray_scores']
l_strad_indices_pikart, l_strad_stray_scores_pikart = classes_pikart[label2]['indices'], classes_pikart[label2]['stray_scores']
## tARget
l_stray_indices_target, l_stray_scores_target = classes_target[label3]['indices'], classes_target[label3]['stray_scores']
l_conform_indices_target, l_conform_stray_scores_target = classes_target[label1]['indices'], classes_target[label1]['stray_scores']
l_strad_indices_target, l_strad_stray_scores_target = classes_target[label2]['indices'], classes_target[label2]['stray_scores']

# Fractions
print('Strays: ' + str(len(l_stray_indices_pikart)/np.sum(a_ltime_pik*4 > Lmin)))
print('Conformists: ' + str(len(l_conform_indices_pikart)/np.sum(a_ltime_pik*4 > Lmin)))
print('Straddlers: ' + str(len(l_strad_indices_pikart)/np.sum(a_ltime_pik*4 > Lmin)))
print('Strays: ' + str(len(l_stray_indices_target)/np.sum(a_ltime_target*4 > Lmin)))
print('Conformists: ' + str(len(l_conform_indices_target)/np.sum(a_ltime_target*4 > Lmin)))
print('Straddlers: ' + str(len(l_strad_indices_target)/np.sum(a_ltime_target*4 > Lmin)))


# %% SCATTER PLOT



# CHOOSE VARIABLE
VAR = 'ivt'  # or 'ltime'

# === CLASS-VARIABLE EXTRACTION ===

if VAR == 'ivt':
    DATA = pd.DataFrame({
        'group': (['conformists'] * (len(l_conform_indices_pikart)  + len(l_conform_indices_target))) + 
                 (['strays'] * (len(l_stray_indices_pikart)  + len(l_stray_indices_target))) + 
                 (['straddlers'] * (len(l_strad_indices_pikart)  + len(l_strad_indices_target))),
        'value': np.concatenate([
            np.hstack([a_meanivt_pik[l_conform_indices_pikart], a_meanivt_target[l_conform_indices_target]]),
            np.hstack([a_meanivt_pik[l_stray_indices_pikart], a_meanivt_target[l_stray_indices_target]]),
            np.hstack([a_meanivt_pik[l_strad_indices_pikart], a_meanivt_target[l_strad_indices_target]]),
        ])
    })
    ylabel = 'mean IVT (kg/ms)'
    overall_values = np.concatenate([a_meanivt_pik, a_meanivt_target])

elif VAR == 'ltime':
    DATA = pd.DataFrame({
        'group': (['conformists'] * (len(l_conform_indices_pikart)  + len(l_conform_indices_target))) + 
                 (['strays'] * (len(l_stray_indices_pikart)  + len(l_stray_indices_target))) + 
                 (['straddlers'] * (len(l_strad_indices_pikart)  + len(l_strad_indices_target))),
        'value': np.concatenate([
            np.hstack([a_ltime_pik[l_conform_indices_pikart], a_ltime_target[l_conform_indices_target]]),
            np.hstack([a_ltime_pik[l_stray_indices_pikart], a_ltime_target[l_stray_indices_target]]),
            np.hstack([a_ltime_pik[l_strad_indices_pikart], a_ltime_target[l_strad_indices_target]]),
        ])
    })
    ylabel = 'lifetime (days)'
    overall_values = np.concatenate([a_ltime_pik[a_ltime_pik >= 1], a_ltime_target[a_ltime_target >= 1]])


# === PLOT SETUP ===

group_order = ['conformists', 'straddlers', 'strays']
group_colors = {
    'conformists': '#729b57',  
    'straddlers': '#528ab7',  
    'strays': '#bb5f48'       
}
DATA['group'] = pd.Categorical(DATA['group'], categories=group_order, ordered=True)

# Create background dataframe (whole pop repeated for each group)
bg_data = pd.DataFrame({
    'group': np.repeat(group_order, len(overall_values)),
    'value': np.tile(overall_values, len(group_order))
})
bg_data['group'] = pd.Categorical(bg_data['group'], categories=group_order, ordered=True)




# === VIOLIN PLOT ===
plt.figure(figsize=(6, 5))
mpl.rcParams['font.size'] = 16

# Remove non-positive values if VAR is 'ltime'
if VAR == 'ltime':
    DATA = DATA[DATA['value'] > 0]
    bg_data = bg_data[bg_data['value'] > 0]

ax = sns.violinplot(
    x='group', y='value', data=bg_data, inner=None, cut=0, order=group_order,
    color='lightgrey', linewidth=1
)

sns.violinplot(
    x='group', y='value', data=DATA, inner=None, cut=0, order=group_order, ax=ax
)

# Recolor violins (only the second set needs recoloring)
from matplotlib.collections import PolyCollection
violin_bodies = [c for c in ax.collections if isinstance(c, PolyCollection)]
colored_violin_bodies = violin_bodies[len(group_order):]

for poly, group in zip(colored_violin_bodies, group_order):
    poly.set_facecolor(group_colors[group])
    poly.set_edgecolor("black")
    poly.set_alpha(0.7)
    poly.set_zorder(2)

# Add medians
medians = DATA.groupby('group')['value'].median()
for i, group in enumerate(group_order):
    plt.scatter(i, medians[group], color='black', zorder=3)

plt.ylabel(ylabel)
plt.xlabel('')
plt.grid(zorder=0)

# Set log scale
if VAR == 'ltime':
    ax.set_yscale('log')
    plt.ylim(0.9, DATA['value'].max() * 1.2)

    # Custom log ticks
    custom_ticks = [1, 2, 3, 7, 14, 21]
    ax.set_yticks(custom_ticks)
    ax.set_yticklabels([f"{d}d" for d in custom_ticks])
    
plt.tight_layout()
plt.show()
plt.savefig(OUTPUT_PATH + "Fig4S6d.png", dpi=500, bbox_inches='tight')



strays = DATA[DATA['group'] == 'strays']['value'].dropna()
conformists = DATA[DATA['group'] == 'conformists']['value'].dropna()
straddlers = DATA[DATA['group'] == 'straddlers']['value'].dropna()

# Wasserstein
print("Wasserstein strays vs conformists:",
      wasserstein_distance(strays, conformists) / np.sqrt((np.var(strays) + np.var(conformists)) / 2))
print("Wasserstein strays vs straddlers:",
      wasserstein_distance(strays, straddlers) / np.sqrt((np.var(strays) + np.var(straddlers)) / 2))
print("Wasserstein conformists vs straddlers:",
      wasserstein_distance(conformists, straddlers) / np.sqrt((np.var(conformists) + np.var(straddlers)) / 2))

# KS tests
print("KS strays vs conformists:", ks_2samp(strays, conformists))
print("KS strays vs straddlers:", ks_2samp(strays, straddlers))
print("KS conformists vs straddlers:", ks_2samp(conformists, straddlers))

# Mann-Whitney U tests
print("MWU strays vs conformists:", mannwhitneyu(strays, conformists, alternative='two-sided'))
print("MWU strays vs straddlers:", mannwhitneyu(strays, straddlers, alternative='two-sided'))
print("MWU conformists vs straddlers:", mannwhitneyu(conformists, straddlers, alternative='two-sided'))



# %% MAP PLOT

## Random seeds:
#A: N - 543, S - 234
#B: N - 567, S - 567
#C: N - 567, S - 454

# Parameters
tracktype = 'stray'
hemisphere = 'south'  # or 'south'
Nsmpl = 10
SEED = 567#476
l_cols = ["#003b76", "#ffab2f", "#ec48fa", "#6b9700", "#ac79ff", "#c89900", "#76006f",
          "#01864e", "#f80034", "#84b1ff", "#ff8f42", "#005757", "#de0053", "#eeb8c4",
          "#4b1303", "#fcade4", "#450f3c", "#ffa86d", "#910057", "#8b5f58"]


# BACKBONE: EBC
Gneutr = l_Gcons[ns-2].copy()
a_allweights = np.hstack([data['edge_betweenness'] for _, _, data in Gneutr.edges(data=True)])
wmax = np.nanmax(a_allweights)
norm = LogNorm(vmin=np.nanmin(a_allweights), vmax=wmax)#Normalize(vmin=0, vmax=wmax)#


# SELECT tracks & TYPE OF TRACKS
l_alltracks = observed_paths_pik.copy()
if tracktype == 'straddler':
    a_indices, a_scores = np.array(l_strad_indices_pikart), np.array(l_strad_stray_scores_pikart)
elif tracktype == 'stray':
    a_indices, a_scores = np.array(l_stray_indices_pikart), np.array(l_stray_scores_pikart)
elif tracktype == 'conformist':
    a_indices, a_scores = np.array(l_conform_indices_pikart), np.array(l_conform_stray_scores_pikart)
else:
    print("That's not a valid track type dude.")

# NH/SH
if hemisphere == 'north':
    minlat, maxlat = 10, 90
elif hemisphere == 'south':
    minlat, maxlat = -90, -10
    
          
## Generate a sample
#a_smpl = a_order[np.random.choice(np.arange(a_order.size-1), Nsmpl)]
#l_idx, l_scores = a_indices[a_smpl], a_scores[a_smpl]
l_plottracks = [l_alltracks[idx] for idx in a_indices]


# First, identify hemisphere of each trajectory
# Filter tracks based on hemisphere
valid_idx = [i for i, path in enumerate(l_plottracks) if is_in_hemisphere(path, hemisphere)]
filtered_scores = a_scores[valid_idx]
filtered_indices = a_indices[valid_idx]
# Sort filtered scores to ensure uniform sampling
sorted_order = np.argsort(filtered_scores)
sorted_scores = filtered_scores[sorted_order]
sorted_indices = filtered_indices[sorted_order]
# Now select Nsmpl samples uniformly across the sorted scores
np.random.seed(SEED)
quantile_bins = np.linspace(0, len(sorted_scores)-1, Nsmpl, dtype=int)
sampled_idx = np.random.choice(np.arange(len(sorted_scores)-1), Nsmpl)#sorted_order[quantile_bins]
l_idx, l_scores = sorted_indices[sampled_idx], sorted_scores[sampled_idx]#sorted_indices[quantile_bins], sorted_scores[quantile_bins]


# FIGURE
%matplotlib 
# Plot settings
proj = ccrs.EqualEarth(central_longitude=0)
d_position = {i: proj.transform_point(Gneutr.nodes[i]['Longitude'], Gneutr.nodes[i]['Latitude'],
                                      src_crs=ccrs.PlateCarree()) for i in Gneutr.nodes}

fig, ax = plt.subplots(subplot_kw={'projection': proj}, figsize=(10, 10))
#ax.set_global()#
ax.set_extent([-180, 180, minlat, maxlat], crs=ccrs.PlateCarree())
ax.coastlines(color='black', linewidth=0.5)

# # Plot HIGHWAYS
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
            False, l0=6, curvature=0, alpha=.5, arrow_size=0
        )


# TRAJECTORIES
for i, idx in enumerate(l_idx):
    path = l_alltracks[idx]
    coords = [h3.h3_to_geo(h) for h in path]
    lats, lons = zip(*coords)
    
    if np.abs(lats[0]) >= 0 and len(lats) >= 4:
        lons, lats = remove_consecutive_duplicates(lons, lats)
        
        if len(lons) >= 4:
            lons = np.unwrap(np.radians(lons))
            lons = np.degrees(lons)

            try:
                tck, _ = splprep([lons, lats], s=0.001)
                smooth_lons, smooth_lats = splev(np.linspace(0, 1, 100), tck)
            except Exception as e:
                print(f"Skipping smoothing for trajectory {i} due to error: {e}")
                smooth_lons, smooth_lats = lons, lats

            # Build segments
            points = np.array([smooth_lons, smooth_lats]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Create a line collection with gradient alpha
            n_segments = len(segments)
            alphas = np.linspace(0.2, 1.0, n_segments)
            
                        
            base_rgba = to_rgba(l_cols[i])  # Convert to (r, g, b, a)
            colors = [(base_rgba[0], base_rgba[1], base_rgba[2], a) for a in alphas]
            
            #colors = [(*l_cols[i][:3], a) for a in alphas]  # RGBA

            lc = LineCollection(
                segments,
                colors=colors,
                linewidths=5,
                transform=ccrs.Geodetic(), zorder=5
            )
            ax.add_collection(lc)
            
plt.tight_layout()
plt.show()
plt.savefig(OUTPUT_PATH + "Fig4S6c2.png", dpi=500, bbox_inches='tight')


