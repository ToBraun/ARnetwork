# Copyright (C) 2026 by
# Tobias Braun

#------------------ PATH ---------------------------#
# working directory
import sys
from pathlib import Path
import os

# Set root directory
REPO_ROOT = Path.cwd()
# Insert path to be able to find subroutines
sys.path.insert(0, str(REPO_ROOT))

# Set paths
INPUT_PATH  = Path(os.environ.get("ARNET_DATA",   REPO_ROOT / "data"))
OUTPUT_PATH = Path(os.environ.get("ARNET_FIGURES", REPO_ROOT / "figures"))

# %% IMPORT MODULES
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

# specific packages
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cmcrameri import cm
from scipy.stats import pearsonr


# local subroutines
import ARnet_sub as artn
import NETanalysis_sub as ana

# %% PLOT PARAMETERS
plt.style.use('default')
plt.rcParams.update({'xtick.direction': 'in', 'ytick.direction': 'in'})
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['font.size'] = 16

size_scale = 50
sign_color_map = {
    -3: 'darkred',
    -2: 'peru',
    -1: 'gold',
     0: '#aaaaaa',
     1: 'deepskyblue',
     2: 'dodgerblue',
     3: 'darkblue',
}
legend_colors = ['#B22222', '#E66100', '#FDB863', '#999999',
                 'deepskyblue', 'dodgerblue', 'navy']
legend_sizes = [12, 11, 10, 9, 10, 11, 12]
ivt_vals = [200, 400, 600, 800, 1000]

# %% FUNCTIONS

def build_network(ARcat):
    ARcat = ARcat.copy()
    ARcat['time'] = pd.to_datetime(ARcat['time']).dt.floor('D')
    ARcat['lf_lon'] = ARcat['lf_lon'].replace(0, np.nan)
    l_arcats, d_coord_dict = artn.preprocess_catalog(
        ARcat, T, loc, grid_type, X, res, cond, LC_cond
    )
    A, t_idx, t_hexidx, t_ivt, t_grid = artn.generate_transport_matrix(
        l_arcats, grid_type, d_coord_dict, LC_cond
    )
    G = artn.generate_network(A, t_grid, weighted, directed, eps, self_links, weighing)
    return G, t_hexidx, t_ivt


def plot_and_correlate(summary_df, color):
    x = summary_df['mean_score']
    y = summary_df['ref_ivt']
    mask = (~pd.isna(x)) & (~pd.isna(y))
    x, y = x[mask], y[mask]
    if len(x) == 0:
        print("No valid data")
        return None, None
    r, p = pearsonr(x, y)
    plt.figure(figsize=(2, 2))
    plt.scatter(x, y, alpha=0.05, color=color)
    plt.xlabel('mean trajectory score')
    plt.ylabel('IVT at landfall (kg m$^{-1}$ s$^{-1}$)')
    plt.title(f'Pearson R={r:.2f}')
    plt.xticks([-3, -2, -1, 0, 1, 2, 3])
    plt.show()
    return r, p


def regrid_coords(df, lon_col='lon', lat_col='lat', res=3.0):
    # Simple regrid
    df = df.copy()
    df['lon_bin'] = (df[lon_col] / res).round() * res
    df['lat_bin'] = (df[lat_col] / res).round() * res
    df['lonlat'] = df['lon_bin'].astype(str) + "_" + df['lat_bin'].astype(str)
    return df



# %% LOAD DATA

d_ars_pikart = pd.read_pickle(INPUT_PATH / 'PIKART_hex.pkl')
d_ars_target = pd.read_pickle(INPUT_PATH / 'target_hex.pkl')

# tARget needs landfall lon/lat from the untransformed catalog
d_ars_target_nohex = pd.read_pickle(INPUT_PATH / 'tARget_globalARcatalog_ERA5_1940-2023_v4.0_converted_lf.pkl')
d_ars_target['lf_lon'] = d_ars_target_nohex['lf_lon']
d_ars_target['lf_lat'] = d_ars_target_nohex['lf_lat']


# %% AR NETWORK

# Network parameters
T = None
X = 'global'
res = 2                  # H3 resolution ~ 2 degrees
grid_type = 'hexagonal'
loc = 'centroid'
weighing = 'absolute'
self_links = False
weighted = True
directed = True
ndec = 8.4
eps = int(2 * ndec)      # ≥ 2 ARs / decade
thresh = 1.25 * eps      # kept for parity with original (unused below)
cond = None
LC_cond = None

# Compute networks
Gpikart, t_hexidx_pikart, t_ivt_pikart = build_network(d_ars_pikart)
Gtarget, t_hexidx_target, t_ivt_target = build_network(d_ars_target)


# %% MOISTURE TRANSPORT — assign IVT-change classes to NODES 

qh1, qh2, qh3 = 0.60, 0.75, 0.90
ql1, ql2, ql3 = 1 - qh1, 1 - qh2, 1 - qh3

a_ivt_diffs_pikart = t_ivt_pikart[0][1] - t_ivt_pikart[0][0]
a_ivt_diffs_target = t_ivt_target[0][1] - t_ivt_target[0][0]
a_all_ivtdiffs = np.hstack([a_ivt_diffs_pikart, a_ivt_diffs_target])

a_IVTthresholds = np.hstack([
    np.nanquantile(a_all_ivtdiffs, ql3),
    np.nanquantile(a_all_ivtdiffs, ql2),
    np.nanquantile(a_all_ivtdiffs, ql1),
    np.nanquantile(a_all_ivtdiffs, qh1),
    np.nanquantile(a_all_ivtdiffs, qh2),
    np.nanquantile(a_all_ivtdiffs, qh3),
])

# Node-level moisture-transport classes
tmp_nodesigns_pikart = ana.compute_node_moisture_transport(
    t_hexidx_pikart[0], t_ivt_pikart[0],
    output='manual', thresholds=a_IVTthresholds
)
Gsigned_pikart = artn.add_node_attr_to_graph(
    Gpikart, tmp_nodesigns_pikart, attr_name='IVTdiff'
)

tmp_nodesigns_target = ana.compute_node_moisture_transport(
    t_hexidx_target[0], t_ivt_target[0],
    output='manual', thresholds=a_IVTthresholds
)
Gsigned_target = artn.add_node_attr_to_graph(
    Gtarget, tmp_nodesigns_target, attr_name='IVTdiff'
)


# Consensus network
Gcons0 = artn.average_networks_by_attributes(Gsigned_pikart, Gsigned_target,attr_name='IVTdiff')
Gcons = artn.complete_nodes(Gcons0, res)


# %% MOISTURE along NODES — Fig 3a

Gplot = Gcons.copy()

proj = ccrs.EqualEarth(central_longitude=0)
d_position = {
    i: proj.transform_point(Gplot.nodes[i]['Longitude'], Gplot.nodes[i]['Latitude'],
                            src_crs=ccrs.PlateCarree())
    for i in Gplot.nodes
}

signs = np.array([
    int(round(s)) if pd.notnull(s) else 0
    for s in (Gplot.nodes[i].get('sign', 0) for i in Gplot.nodes)
])
abs_signs = np.abs(signs)
max_sign = np.nanmax(abs_signs) if len(abs_signs) else 1

x_coords, y_coords = zip(*d_position.values())

colors = [sign_color_map.get(s, '#aaaaaa') for s in signs]
sizes = [
    size_scale * (abs(s) / max_sign) if not np.isclose(s, 0) else size_scale * 0.2
    for s in signs
]
alphas = [0.3 if s == 0 else 0.7 for s in signs]


# FIGURE
fig, ax = plt.subplots(subplot_kw={'projection': proj}, figsize=(10, 10))
ax.set_global()
ax.coastlines(color='black', linewidth=0.5)
ax.scatter(x_coords, y_coords, s=sizes, c=colors, alpha=alphas,
           linewidths=0.3, zorder=10)

tick_labels = (
    [f"< {a_IVTthresholds[0]:.0f}"] +
    [f"({lo:.0f},{hi:.0f})" for lo, hi in zip(a_IVTthresholds[:-1], a_IVTthresholds[1:])] +
    [f"{a_IVTthresholds[-1]:.0f} <"]
)

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=label,
           markerfacecolor=color, markersize=size)
    for label, color, size in zip(tick_labels, legend_colors, legend_sizes)
]
fig.legend(handles=legend_elements, title='Net IVT change (kg/ms)',
           loc='lower center', bbox_to_anchor=(0.5, 0.12),
           ncol=4, fontsize=14, title_fontsize=16, frameon=False)

plt.subplots_adjust(top=0.9)
plt.show()
plt.savefig(OUTPUT_PATH + "Fig3a.png", dpi=500, bbox_inches='tight')


# %% PREDICTABILITY — Fig 3b


# --- Compute trajectory scores (node mode) ---
summary_pikart = ana.compute_track_scores(d_ars_pikart, Gsigned_pikart, min_length=4)
plot_and_correlate(summary_pikart, color='darkcyan')
plt.savefig(OUTPUT_PATH + "scatter_nodeIVT_pikart.png", dpi=500, bbox_inches='tight')

summary_target = ana.compute_track_scores(d_ars_target, Gsigned_target, min_length=4)
plot_and_correlate(summary_target, color='mediumpurple')
plt.savefig(OUTPUT_PATH + "scatter_nodeIVT_target.png", dpi=500, bbox_inches='tight')


# --- Per-landfall predictability ---
lf_pikart = ana.compute_predictability(d_ars_pikart, summary_pikart, epsilon=1)
lf_target = ana.compute_predictability(d_ars_target, summary_target, epsilon=1)
# Wrap tARget longitudes to [-180, 180]
lf_target['lon'] = ((lf_target['lon'] + 180) % 360) - 180

# Aggregate per grid cell (mean over tracks that land in the same cell)
lf_pikart_agg = lf_pikart.groupby(['lon', 'lat'], as_index=False).agg(
    {'ivt': 'mean', 'pred': 'mean'}
)
lf_target_agg = lf_target.groupby(['lon', 'lat'], as_index=False).agg(
    {'ivt': 'mean', 'pred': 'mean'}
)

# Regrid to a 5° common grid and merge
lf_pikart_coarse = regrid_coords(lf_pikart_agg, res=5.0)
lf_target_coarse = regrid_coords(lf_target_agg, res=5.0)

merged = pd.merge(
    lf_pikart_coarse, lf_target_coarse,
    on='lonlat', suffixes=('_pikart', '_target'), how='outer'
)
merged['lon']  = merged[['lon_bin_pikart', 'lon_bin_target']].mean(axis=1, skipna=True)
merged['lat']  = merged[['lat_bin_pikart', 'lat_bin_target']].mean(axis=1, skipna=True)
merged['ivt']  = merged[['ivt_pikart',     'ivt_target']].mean(axis=1, skipna=True)
merged['pred'] = merged[['pred_pikart',    'pred_target']].mean(axis=1, skipna=True)

merged['has_pikart'] = ~merged['ivt_pikart'].isna()
merged['has_target'] = ~merged['ivt_target'].isna()
merged['both'] = merged['has_pikart'] & merged['has_target']

df_both   = merged[merged['both']]
df_unique = merged[~merged['both']]

# Plot: predictability map 
fig, ax = plt.subplots(figsize=(12.6, 12.6), subplot_kw={'projection': ccrs.EqualEarth()})
ax.set_global()
ax.add_feature(cfeature.COASTLINE, color='black', zorder=2)

sc2 = ax.scatter(
    df_unique['lon'], df_unique['lat'],
    c=df_unique['pred'], s=df_unique['ivt'] / 9,
    cmap=cm.batlow_r, transform=ccrs.PlateCarree(),
    alpha=0.6, edgecolors='none', label='Only one catalog',
)
sc1 = ax.scatter(
    df_both['lon'], df_both['lat'],
    c=df_both['pred'], s=df_both['ivt'] / 9,
    cmap=cm.batlow_r, transform=ccrs.PlateCarree(),
    alpha=0.9, edgecolors='black', linewidths=0.8, label='Both catalogs',
)

cb = plt.colorbar(sc1, ax=ax, shrink=0.5, orientation='vertical', pad=0.05)
cb.ax.tick_params(labelsize=16)
cb.set_label('Predictability of IVT at landfall', fontsize=16)

# Size legend
marker_sizes = [v / 9 for v in ivt_vals]
handles = [plt.scatter([], [], s=ms, color='gray', alpha=0.6, edgecolors='none')
           for ms in marker_sizes]
size_legend = ax.legend(
    handles, [f"{v}" for v in ivt_vals],
    title='IVT at landfall (kg m$^{-1}$ s$^{-1}$)',
    loc='lower left', fontsize=14, title_fontsize=14, frameon=True,
)
ax.legend(loc='upper left', fontsize=14, frameon=True)
ax.add_artist(size_legend)

plt.show()
plt.savefig(OUTPUT_PATH + "Fig3b_predict.png", dpi=500, bbox_inches='tight')


