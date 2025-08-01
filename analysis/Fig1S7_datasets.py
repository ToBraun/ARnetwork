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
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt

# specific packages
import cartopy.crs as ccrs


# my packages
import ARnet_sub as artn
import NETplots_sub as nplot


# %% PLOT PARAMETERS
plt.style.use('dark_background')
# Update Matplotlib parameters
colorbar_dir = 'horizontal'

# Change default tick direction
params = {'xtick.direction': 'in',
          'ytick.direction': 'in'}
plt.rcParams.update(params)
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['font.size'] = 18

# %% LOAD DATA

# PIKART
d_ars_pikart = pd.read_pickle(INPUT_PATH + 'PIKART' + '_hex.pkl')

# PIKART fromn MERRA2
d_ars_pikart2 = pd.read_pickle(INPUT_PATH + 'pikart_merra2' + '_hex.pkl')

# tARget v4
d_ars_target = pd.read_pickle(INPUT_PATH + 'target' + '_hex.pkl')


# %% PARAMETERS

"""
Figure 1 S7: generate networks for pre/post-1979 from ERA5 and from MERRA2.
"""

## Network parameters
# spatiotemporal extent
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
ndec_pre = (1979-1940)/10 # before 1979
ndec_post = (2023-1980)/10 # after 1979
ndec_merra = (2019-1980)/10 # MERRA2
eps_pre, eps_post, eps_merra = int(2*ndec_pre), int(2*ndec_post), int(2*ndec_merra) 
thresh_pre, thresh_post, thresh_merra = 1.25*eps_pre, 1.25*eps_post, 1.25*eps_merra
# conditioning
cond = None # any network conditioning
LC_cond = None # lifecycle conditioning



# %% A - PRE-1979

# Time clipping
T = (1940, 1979) 

# PIKART
ARcat = d_ars_pikart.copy()
l_arcats_pikart, d_coord_dict = artn.preprocess_catalog(ARcat, T, loc, grid_type, X, res, cond, LC_cond)
Apik, t_idx, t_hashidx, t_ivt, t_gridpik = artn.generate_transport_matrix(l_arcats_pikart, grid_type, d_coord_dict, LC_cond)
Gpre = artn.generate_network(Apik, t_gridpik, weighted, directed, eps_pre, self_links, weighing)
Gplot = artn.complete_nodes(Gpre, res)

### PLOT
nplot.plot_network(Gplot, widths='weights', colours='weights', layout='default', ndec=ndec_pre, log=True,
                  arrowsize=0, linewidth=2, curvature=0.4, fontsize=14, ncolors=20, discard=180,
                  alpha=.7, show_nodes=True, proj = ccrs.EqualEarth(), show_axes=False)
plt.savefig(OUTPUT_PATH + "Fig1S7a.png", dpi=300, bbox_inches='tight')


# %% B - POST-1979

# Time clipping
T = (1980, 2023) 

# PIKART
ARcat = d_ars_pikart.copy()
l_arcats_pikart, d_coord_dict = artn.preprocess_catalog(ARcat, T, loc, grid_type, X, res, cond, LC_cond)
Apik, t_idx, t_hashidx, t_ivt, t_gridpik = artn.generate_transport_matrix(l_arcats_pikart, grid_type, d_coord_dict, LC_cond)
Gpost = artn.generate_network(Apik, t_gridpik, weighted, directed, eps_post, self_links, weighing)
Gplot = artn.complete_nodes(Gpost, res)

### PLOT
nplot.plot_network(Gplot, widths='weights', colours='weights', layout='default', ndec=ndec_post, log=True,
                  arrowsize=0, linewidth=2, curvature=0.4, fontsize=14, ncolors=20, discard=180,
                  alpha=.7, show_nodes=True, proj = ccrs.EqualEarth(), show_axes=False)
plt.savefig(OUTPUT_PATH + "Fig1S7b.png", dpi=300, bbox_inches='tight')



# %% C - MERRA2

# Time clipping: full time period for MERRA2
T = None

# PIKART: MERRA2
ARcat = d_ars_pikart2.copy()
l_arcats_pikart, d_coord_dict = artn.preprocess_catalog(ARcat, T, loc, grid_type, X, res, cond, LC_cond)
Apik, t_idx, t_hashidx, t_ivt, t_gridpik = artn.generate_transport_matrix(l_arcats_pikart, grid_type, d_coord_dict, LC_cond)
Gmerra = artn.generate_network(Apik, t_gridpik, weighted, directed, eps_merra, self_links, weighing)
Gplot = artn.complete_nodes(Gmerra, res)

### PLOT
nplot.plot_network(Gplot, widths='weights', colours='weights', layout='default', ndec=ndec_merra, log=True,
                  arrowsize=0, linewidth=2, curvature=0.4, fontsize=14, ncolors=20, discard=180,
                  alpha=.7, show_nodes=True, proj = ccrs.EqualEarth(), show_axes=False)
plt.savefig(OUTPUT_PATH + "Fig1S7c.png", dpi=300, bbox_inches='tight')


