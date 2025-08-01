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
from matplotlib.ticker import ScalarFormatter
from scipy import stats

# specific packages
import cartopy.crs as ccrs

# my packages
import ARnet_sub as artn
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
Fig 1 S5: EDGE DIRECTIONS
"""

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
ndec = 8.4
eps = int(ndec)
thresh = 1.25*eps
# conditioning
cond = None # any network conditioning
LC_cond = None # lifecycle conditioning


# %% PANEL A: PIKART


## Generate AR network 
ARcat = d_ars_pikart.copy()
l_arcats_pikart, d_coord_dict = artn.preprocess_catalog(ARcat, T, loc, grid_type, X, res, cond, LC_cond)
Apik, t_idx, t_hexidx, t_ivt, t_gridpik = artn.generate_transport_matrix(l_arcats_pikart, grid_type, d_coord_dict, LC_cond)
Gpikart = artn.generate_network(Apik, t_gridpik, weighted, directed, eps, self_links, weighing)
Gplot_pikart = artn.complete_nodes(Gpikart, res)


### PLOT
nplot.plot_network(Gplot_pikart, widths=None, colours='directions', layout='default', ndec=ndec, log=False,
                  arrowsize=0, linewidth=1, curvature=0.4, fontsize=14, ncolors=20, discard=180,
                  alpha=.4, show_nodes=True, proj = ccrs.EqualEarth(),
                  show_axes=False)
plt.savefig(OUTPUT_PATH + 'Fig1S5a.png', dpi=300, bbox_inches='tight')



# %% PANEL B: tARget

## Generate AR network 
ARcat = d_ars_target.copy()
l_arcats_target, d_coord_dict = artn.preprocess_catalog(ARcat, T, loc, grid_type, X, res, cond, LC_cond)
Atarget, t_idx, t_hexidx, t_ivt, t_gridtarget = artn.generate_transport_matrix(l_arcats_target, grid_type, d_coord_dict, LC_cond)
Gtarget = artn.generate_network(Atarget, t_gridtarget, weighted, directed, eps, self_links, weighing)
Gplot_target = artn.complete_nodes(Gtarget, res)

### PLOT
nplot.plot_network(Gplot_target, widths=None, colours='directions', layout='default', ndec=ndec, log=False,
                  arrowsize=0, linewidth=1, curvature=0.4, fontsize=14, ncolors=20, discard=180,
                  alpha=.4, show_nodes=True, proj = ccrs.EqualEarth(),
                  show_axes=False, vmax=17*ndec)
plt.savefig(OUTPUT_PATH + 'Fig1S5b.png', dpi=300, bbox_inches='tight')




# %% PANEL C: consensus


# Consensus
Gcons = artn.consensus_network([Gpikart, Gtarget], thresh, eps)
Gplot_cons = artn.complete_nodes(Gcons, res)

### PLOT
nplot.plot_network(Gplot_cons, widths=None, colours='directions', layout='default', ndec=ndec, log=False,
                  arrowsize=0, linewidth=1, curvature=0.4, fontsize=14, ncolors=20, discard=180,
                  alpha=.4, show_nodes=True, proj = ccrs.EqualEarth(),
                  show_axes=False, vmax=17*ndec)
plt.savefig(OUTPUT_PATH + 'Fig1S5c.png', dpi=300, bbox_inches='tight')


# %% PANEL D: KDEs

# Extract edge directions
a_edgedir_pik = np.hstack(list(nplot.edgedir_to_degrees(Gtarget).values()))
a_edgedir_target = np.hstack(list(nplot.edgedir_to_degrees(Gpikart).values()))
a_edgedir_cons = np.hstack(list(nplot.edgedir_to_degrees(Gcons).values()))

# Derive gaussian kernel density estimates
# PIKART
kde_pik = stats.gaussian_kde(a_edgedir_pik, bw_method='scott')  
xpik = np.linspace(min(a_edgedir_pik), max(a_edgedir_pik), 1000)
# TARGET
kde_target = stats.gaussian_kde(a_edgedir_target, bw_method='scott')  
xtarget = np.linspace(min(a_edgedir_target), max(a_edgedir_target), 1000)
# CONSENSUS
kde_pcons= stats.gaussian_kde(a_edgedir_cons, bw_method='scott')  
xcons = np.linspace(min(a_edgedir_cons), max(a_edgedir_cons), 1000)

# Plotting
fig = plt.figure(figsize=(8,4))
mpl.rcParams['font.size'] = 20
plt.grid()
plt.plot(xpik, kde_pik(xpik), label='PIKART', color='darkcyan', linewidth=2)
plt.plot(xtarget, kde_target(xtarget), label='tARget', color='goldenrod', linewidth=2)
plt.plot(xcons, kde_pcons(xcons), label='consensus', color='black', linewidth=3)
plt.xlabel('edge direction'); plt.ylabel('PDF')
plt.xticks([90, 180, 270, 360], labels=['E', 'S', 'W', 'N'])
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=3, fontsize=13)
formatter = ScalarFormatter()
formatter.set_scientific(True)
formatter.set_powerlimits((-3, 3)) 
plt.gca().yaxis.set_major_formatter(formatter)
plt.savefig(OUTPUT_PATH + 'Fig1S5d.png', dpi=300, bbox_inches='tight')

