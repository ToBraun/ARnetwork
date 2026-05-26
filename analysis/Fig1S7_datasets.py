# Copyright (C) 2025 by
# Tobias Braun

#------------------ PATHS ---------------------------#

# working directory
import sys
WDPATH = "/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/paper/Nature/scripts/ARnetlab"
sys.path.insert(0, WDPATH)
# input and output
INPUT_PATH = '/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/paper/Nature/data/'
OUTPUT_PATH = '/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/paper/suppl_figures/'


# %% IMPORT MODULES


# standard packages
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import spearmanr

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
Gpre = Gplot.copy()

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
Gpost = Gplot.copy()

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
Gmerra = Gplot.copy()

### PLOT
nplot.plot_network(Gplot, widths='weights', colours='weights', layout='default', ndec=ndec_merra, log=True,
                  arrowsize=0, linewidth=2, curvature=0.4, fontsize=14, ncolors=20, discard=180,
                  alpha=.7, show_nodes=True, proj = ccrs.EqualEarth(), show_axes=False)
plt.savefig(OUTPUT_PATH + "Fig1S7c.png", dpi=300, bbox_inches='tight')




# %% QUANTITATIVE ROBUSTNESS DIAGNOSTICS



def edge_weight_dict(G):
    """Return {(u,v): weight} for a weighted DiGraph."""
    return {(u, v): d.get('weight', 1.0) for u, v, d in G.edges(data=True)}

def jaccard(G1, G2):
    E1, E2 = set(G1.edges()), set(G2.edges())
    return len(E1 & E2) / len(E1 | E2) if (E1 | E2) else np.nan

def weighted_jaccard(G1, G2):
    w1, w2 = edge_weight_dict(G1), edge_weight_dict(G2)
    keys = set(w1) | set(w2)
    num = sum(min(w1.get(k, 0), w2.get(k, 0)) for k in keys)
    den = sum(max(w1.get(k, 0), w2.get(k, 0)) for k in keys)
    return num / den if den > 0 else np.nan

def spearman_on_common(G1, G2):
    w1, w2 = edge_weight_dict(G1), edge_weight_dict(G2)
    common = set(w1) & set(w2)
    if len(common) < 3:
        return np.nan, len(common)
    a = np.array([w1[k] for k in common])
    b = np.array([w2[k] for k in common])
    rho, _ = spearmanr(a, b)
    return rho, len(common)

def topk_jaccard(G1, G2, frac=0.05):
    """Jaccard on the top-frac strongest edges of each network."""
    w1, w2 = edge_weight_dict(G1), edge_weight_dict(G2)
    k1 = max(1, int(frac * len(w1)))
    k2 = max(1, int(frac * len(w2)))
    top1 = set(sorted(w1, key=w1.get, reverse=True)[:k1])
    top2 = set(sorted(w2, key=w2.get, reverse=True)[:k2])
    return len(top1 & top2) / len(top1 | top2) if (top1 | top2) else np.nan

# rate-normalise weights (edges per decade) so networks of different length are comparable
def rate_normalise(G, ndec):
    H = G.copy()
    for u, v, d in H.edges(data=True):
        d['weight'] = d.get('weight', 1.0) / ndec
    return H

Gpre_n   = rate_normalise(Gpre,   ndec_pre)
Gpost_n  = rate_normalise(Gpost,  ndec_post)
Gmerra_n = rate_normalise(Gmerra, ndec_merra)

pairs = [('pre vs post', Gpre_n, Gpost_n),
         ('post vs MERRA2', Gpost_n, Gmerra_n),
         ('pre vs MERRA2', Gpre_n, Gmerra_n)]

print(f"{'pair':<18} {'|E1|':>6} {'|E2|':>6} {'Jacc':>6} {'wJacc':>6} {'rho':>6} {'top5%':>6}")
for label, G1, G2 in pairs:
    J  = jaccard(G1, G2)
    wJ = weighted_jaccard(G1, G2)
    rho, n_common = spearman_on_common(G1, G2)
    Jt = topk_jaccard(G1, G2, frac=0.05)
    print(f"{label:<18} {G1.number_of_edges():>6d} {G2.number_of_edges():>6d} "
          f"{J:>6.3f} {wJ:>6.3f} {rho:>6.3f} {Jt:>6.3f}")
