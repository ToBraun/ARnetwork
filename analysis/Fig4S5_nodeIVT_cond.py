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
OUTPUT_PATH = '/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/paper/suppl_figures/'




# %% IMPORT MODULES

# standard packages
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

# specific packages
from tqdm import tqdm
import cartopy.crs as ccrs

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




def plot_change_map(G, attr='IVTchange',  output_path='change.png'):
    # Project positions
    proj = ccrs.EqualEarth(central_longitude=0)
    pos = {
        i: proj.transform_point(G.nodes[i]['Longitude'], G.nodes[i]['Latitude'], src_crs=ccrs.PlateCarree())
        for i in G.nodes
    }
    x, y = zip(*pos.values())
    values = np.array([G.nodes[i][attr] for i in G.nodes])

    # Discrete colormap setup
    bounds = np.arange(-6.5, 7.5, 1)  # bins centered on -6 to +6
    cmap = plt.cm.get_cmap('RdBu', len(bounds) - 1)  # 13 discrete colors
    norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=cmap.N)

    # Plot
    fig, ax = plt.subplots(subplot_kw={'projection': proj}, figsize=(8, 8))
    ax.set_global()
    ax.coastlines()

    sc = ax.scatter(
        x, y,
        c=values,
        s=np.clip(np.abs(values)*10, 10, 60),  # scale size with bounds
        cmap=cmap,
        norm=norm,
        alpha=0.8,
        edgecolor='k',
        linewidth=0.1,
        zorder=3
    )

    # Discrete colorbar
    cbar = fig.colorbar(sc, ax=ax, orientation='horizontal', pad=0.05, boundaries=bounds, ticks=np.arange(-6, 7, 1))
    cbar.set_label("IVT Class Change (El Niño - La Niña)", fontsize=16)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()



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



# %% CONDITIONAL AR NETWORKS

"""
SEASONS order: winter, summer
ENSO order: La Nina, neutral, El Nino
"""

# PICK THE INDEX
cond = 'teleconnection'

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



""" NODES """
# SEASONS: 0, 1 
# ENSO: 0, 2
n1, n2 = 0, 2

# PIKART
## regime #1
tmp_nodesigns_pik1 = ana.compute_node_moisture_transport(t_hexidx_pikart[n1],
                                                    t_ivt_pikart[n1],
                                                    output = 'manual',
                                                    thresholds = a_IVTthresholds)
Gpikart[n1] = artn.add_node_attr_to_graph(Gpikart[n1], tmp_nodesigns_pik1, attr_name = 'IVTdiff')
## regime #2
tmp_nodesigns_pik2 = ana.compute_node_moisture_transport(t_hexidx_pikart[n2],
                                                    t_ivt_pikart[n2],
                                                    output = 'manual',
                                                    thresholds = a_IVTthresholds)
Gpikart[n2] = artn.add_node_attr_to_graph(Gpikart[n2], tmp_nodesigns_pik2, attr_name = 'IVTdiff')


# tARget
## regime #1
tmp_nodesigns_target1 = ana.compute_node_moisture_transport(t_hexidx_target[n1],
                                                    t_ivt_target[n1],
                                                    output = 'manual',
                                                    thresholds = a_IVTthresholds)
Gtarget[n1] = artn.add_node_attr_to_graph(Gtarget[n1], tmp_nodesigns_target1, attr_name = 'IVTdiff')
## regime #2
tmp_nodesigns_target2 = ana.compute_node_moisture_transport(t_hexidx_target[n2],
                                                    t_ivt_target[n2],
                                                    output = 'manual',
                                                    thresholds = a_IVTthresholds)
Gtarget[n2] = artn.add_node_attr_to_graph(Gtarget[n2], tmp_nodesigns_target2, attr_name = 'IVTdiff')





# %% DIFFERENCES

# Compute class differences
ivt_change_pikart = {
    k: tmp_nodesigns_pik2[k] - tmp_nodesigns_pik1[k]
    for k in tmp_nodesigns_pik1.keys()
    if k in tmp_nodesigns_pik2 and pd.notnull(tmp_nodesigns_pik1[k]) and pd.notnull(tmp_nodesigns_pik2[k])
}
ivt_change_target = {
    k: tmp_nodesigns_target2[k] - tmp_nodesigns_target1[k]
    for k in tmp_nodesigns_target1.keys()
    if k in tmp_nodesigns_target2 and pd.notnull(tmp_nodesigns_target1[k]) and pd.notnull(tmp_nodesigns_target2[k])
}

# Add as node attributes
Gpikart_diff = artn.add_node_attr_to_graph(Gpikart[0].copy(), ivt_change_pikart, attr_name='IVTchange')
Gtarget_diff = artn.add_node_attr_to_graph(Gtarget[0].copy(), ivt_change_target, attr_name='IVTchange')


# %% FIGURE

# Plot for each
plot_change_map(Gpikart_diff, output_path=OUTPUT_PATH + "Fig4S5c.png")
plot_change_map(Gtarget_diff, output_path=OUTPUT_PATH + "Fig4S5d.png")
