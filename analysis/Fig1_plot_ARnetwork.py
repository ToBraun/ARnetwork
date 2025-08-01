# Copyright (C) 2023 by
# Tobias Braun

#------------------ PATH ---------------------------#
# working directory
import sys
WDPATH = "/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/paper/scripts/ARnetlab"
sys.path.insert(0, WDPATH)
# input and output
INPUT_PATH = '/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/paper/data/'
OUTPUT_PATH = '/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/paper/figures/'


# %% IMPORT MODULES

# standard packages
import matplotlib as mpl
#mpl.use('Agg')
from matplotlib import pyplot as plt

# specific packages
import networkx as nx
from networkx.readwrite import gexf
import cartopy.crs as ccrs

# my packages
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

"""
Figure 1: plot the AR networks.
"""

# %% LOAD DATA

Gplot_pikart = gexf.read_gexf(INPUT_PATH + "arnet_pikart_centroid.gexf")
Gplot_target = gexf.read_gexf(INPUT_PATH + "arnet_target_centroid.gexf")
Gplot_cons = gexf.read_gexf(INPUT_PATH + "arnet_consensus_centroid.gexf")


# %% PANEL A -  PIKART

# PARAMETERS
ndec = 8.4
eps = int(2*ndec)


### PLOT
nplot.plot_network(Gplot_pikart, widths='weights', colours='weights', layout='default', ndec=ndec, log=False,
                  arrowsize=0, linewidth=3, curvature=0.4, fontsize=14, ncolors=20, discard=180,
                  alpha=.5, show_nodes=True, proj = ccrs.EqualEarth(), show_axes=False)
plt.savefig(OUTPUT_PATH + "Fig1a.png", dpi=300, bbox_inches='tight')



# %% PANEL B -  tARget

### PLOT
nplot.plot_network(Gplot_target, widths='weights', colours='weights', layout='default', ndec=ndec, log=False,
                  arrowsize=0, linewidth=3, curvature=0.4, fontsize=14, ncolors=20, discard=180,
                  alpha=.5, show_nodes=True, proj = ccrs.EqualEarth(), show_axes=False)
plt.savefig(OUTPUT_PATH + "Fig1b.png", dpi=300, bbox_inches='tight')




# %% PANEL C -  consensus


### PLOT
nplot.plot_network(Gplot_cons, widths='weights', colours='weights', layout='dark', ndec=ndec, log=False,
                  arrowsize=0, linewidth=3, curvature=0.4, fontsize=14, ncolors=20, discard=180,
                  alpha=.5, show_nodes=True, proj = ccrs.EqualEarth(), show_axes=False)
plt.savefig("/Users/tbraun/Desktop/" + "ARnetwork.png", dpi=500, bbox_inches='tight')
#plt.savefig(OUTPUT_PATH + "Fig1c.png", dpi=300, bbox_inches='tight')

