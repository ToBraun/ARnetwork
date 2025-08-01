# Copyright (C) 2025 by
# Tobias Braun

#------------------ PATH ---------------------------#
# working directory
import sys
WDPATH = "/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/paper/scripts/ARnetlab"
sys.path.insert(0, WDPATH)
# input and output
INPUT_PATH = '/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/paper/data/random_graphs/'
OUTPUT_PATH = '/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/paper/suppl_figures/'


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
mpl.rcParams['font.size'] = 16

# %% LOAD DATA

# Locator & number of realizations
loc = 'centroid'
Nrealiz = 4

l_Gcons_rndm, l_Gcons_rewired, l_Gcons_genesis, l_Gcons_term = [], [], [], []
for n in range(Nrealiz):
    l_Gcons_rndm.append(nx.read_gml(INPUT_PATH + 'l_Gwalk_rndm_' + str(n) + '._cons_' + loc + '.gml'))
    l_Gcons_rewired.append(nx.read_gml(INPUT_PATH + 'l_G_rewired_' + str(n) + '_cons_' + loc + '.gml'))
    l_Gcons_genesis.append(nx.read_gml(INPUT_PATH + 'l_Gwalk_genesis_' + str(n) + '_cons_' + loc + '.gml'))
    l_Gcons_term.append(nx.read_gml(INPUT_PATH + 'l_Gwalk_term_' + str(n) + '_cons_' + loc + '.gml'))


# %% PARAMETERS
ndec = 8.4
#%matplotlib

# %% FRW 

### PLOT
for n in range(Nrealiz):
    nplot.plot_network(l_Gcons_rndm[n], widths='weights', colours='weights', layout='default', ndec=ndec, log=False,
                      arrowsize=0, linewidth=1.5, curvature=0.4, fontsize=14, ncolors=20, discard=180,
                      alpha=.5, show_nodes=True, proj = ccrs.EqualEarth(), show_axes=False)
    plt.savefig(OUTPUT_PATH + "Fig2S1a" + str(n) + ".png", dpi=300, bbox_inches='tight')



# %% RWG

### PLOT
for n in range(Nrealiz):
    nplot.plot_network(l_Gcons_rewired[n], widths='weights', colours='weights', layout='default', ndec=ndec, log=False,
                      arrowsize=0, linewidth=2, curvature=0.4, fontsize=14, ncolors=20, discard=180,
                      alpha=.5, show_nodes=True, proj = ccrs.EqualEarth(), show_axes=False)
    plt.savefig(OUTPUT_PATH + "Fig2S1b" + str(n) + ".png", dpi=300, bbox_inches='tight')


# %% GCN

### PLOT
for n in range(Nrealiz):
    nplot.plot_network(l_Gcons_genesis[n], widths='weights', colours='weights', layout='default', ndec=ndec, log=False,
                      arrowsize=0, linewidth=1.5, curvature=0.4, fontsize=14, ncolors=20, discard=180,
                      alpha=.5, show_nodes=True, proj = ccrs.EqualEarth(), show_axes=False)
    plt.savefig(OUTPUT_PATH + "Fig2S1c" + str(n) + ".png", dpi=300, bbox_inches='tight')


# %% TCN


### PLOT
for n in range(Nrealiz):
    nplot.plot_network(l_Gcons_term[n], widths='weights', colours='weights', layout='default', ndec=ndec, log=False,
                      arrowsize=0, linewidth=1.5, curvature=0.4, fontsize=14, ncolors=20, discard=180,
                      alpha=.5, show_nodes=True, proj = ccrs.EqualEarth(), show_axes=False)
    plt.savefig(OUTPUT_PATH + "Fig2S1d" + str(n) + ".png", dpi=300, bbox_inches='tight')
