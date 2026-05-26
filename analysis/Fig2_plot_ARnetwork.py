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

# standard packages
import matplotlib as mpl
#mpl.use('Agg')
from matplotlib import pyplot as plt

# specific packages
from networkx.readwrite import gexf
import cartopy.crs as ccrs

# local subroutines
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
Figure 2: plot the AR network.
"""

# %% LOAD DATA

Gplot_pikart = gexf.read_gexf(INPUT_PATH + "arnet_pikart_centroid.gexf")
Gplot_target = gexf.read_gexf(INPUT_PATH + "arnet_target_centroid.gexf")
Gplot_cons = gexf.read_gexf(INPUT_PATH + "arnet_consensus_centroid.gexf")


# %% PIKART-1.0

# PARAMETERS
ndec = 8.4
eps = int(2*ndec)

### PLOT
nplot.plot_network(Gplot_pikart, widths='weights', colours='weights', layout='default', ndec=ndec, log=False,
                  arrowsize=0, linewidth=3, curvature=0.4, fontsize=14, ncolors=20, discard=180,
                  alpha=.5, show_nodes=True, proj = ccrs.EqualEarth(), show_axes=False)
plt.savefig(OUTPUT_PATH + "Fig2a.png", dpi=300, bbox_inches='tight')


# %% tARget-4

### PLOT
nplot.plot_network(Gplot_target, widths='weights', colours='weights', layout='default', ndec=ndec, log=False,
                  arrowsize=0, linewidth=3, curvature=0.4, fontsize=14, ncolors=20, discard=180,
                  alpha=.5, show_nodes=True, proj = ccrs.EqualEarth(), show_axes=False)

# %% CONSENSUS


### PLOT
nplot.plot_network(Gplot_cons, widths='weights', colours='weights', layout='dark', ndec=ndec, log=False,
                  arrowsize=0, linewidth=3, curvature=0.4, fontsize=14, ncolors=20, discard=180,
                  alpha=.5, show_nodes=True, proj = ccrs.EqualEarth(), show_axes=False)
plt.savefig("/Users/tbraun/Desktop/" + "ARnetwork.png", dpi=500, bbox_inches='tight', transparent=True)