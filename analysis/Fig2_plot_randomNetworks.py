# Copyright (C) 2025 by
# Tobias Braun

#------------------ PATHS ---------------------------#

# working directory
import sys
WDPATH = "/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/paper/scripts/ARnetlab"
sys.path.insert(0, WDPATH)
# input and output
INPUT_PATH = '/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/paper/data/random_graphs/'
OUTPUT_PATH = '/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/paper/figures/'


# %% IMPORT MODULES

# standard packages
import matplotlib as mpl
from matplotlib import pyplot as plt

# specific packages
import networkx as nx
import cartopy.crs as ccrs

# my packages
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

%matplotlib 


# %% LOAD DATA

loc = 'centroid'


Gcons_rndm = nx.read_gml(INPUT_PATH + 'l_Gwalk_rndm_' + str(0) + '._cons_' + loc + '.gml')
Gcons_rewired = nx.read_gml(INPUT_PATH + 'l_G_rewired_' + str(0) + '_cons_' + loc + '.gml')
Gcons_genesis = nx.read_gml(INPUT_PATH + 'l_Gwalk_genesis_' + str(0) + '_cons_' + loc + '.gml')
Gcons_term = nx.read_gml(INPUT_PATH + 'l_Gwalk_term_' + str(0) + '_cons_' + loc + '.gml')


# %% PARAMETERS
ndec = 8.4

# %% FRW 

### PLOT
nplot.plot_network(Gcons_rndm, widths='weights', colours='weights', layout='default', ndec=ndec, log=False,
                  arrowsize=0, linewidth=1.5, curvature=0.4, fontsize=14, ncolors=20, discard=180,
                  alpha=.5, show_nodes=True, proj = ccrs.EqualEarth(), show_axes=False)
plt.savefig(OUTPUT_PATH + "Fig2a.png", dpi=300, bbox_inches='tight')



# %% RWG

### PLOT
nplot.plot_network(Gcons_rewired, widths='weights', colours='weights', layout='default', ndec=ndec, log=False,
                  arrowsize=0, linewidth=2, curvature=0.4, fontsize=14, ncolors=20, discard=180,
                  alpha=.5, show_nodes=True, proj = ccrs.EqualEarth(), show_axes=False)
plt.savefig(OUTPUT_PATH + "Fig2b.png", dpi=300, bbox_inches='tight')


# %% GCN

### PLOT
nplot.plot_network(Gcons_genesis, widths='weights', colours='weights', layout='default', ndec=ndec, log=False,
                  arrowsize=0, linewidth=1.5, curvature=0.4, fontsize=14, ncolors=20, discard=180,
                  alpha=.5, show_nodes=True, proj = ccrs.EqualEarth(), show_axes=False)
plt.savefig(OUTPUT_PATH + "Fig2c.png", dpi=300, bbox_inches='tight')


# %% TCN


### PLOT
nplot.plot_network(Gcons_term, widths='weights', colours='weights', layout='default', ndec=ndec, log=False,
                  arrowsize=0, linewidth=1.5, curvature=0.4, fontsize=14, ncolors=20, discard=180,
                  alpha=.5, show_nodes=True, proj = ccrs.EqualEarth(), show_axes=False)
plt.savefig(OUTPUT_PATH + "Fig2d.png", dpi=500, bbox_inches='tight')
