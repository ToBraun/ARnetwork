# Copyright (C) 2023 by
# Tobias Braun

#------------------ PATH ---------------------------#
# working directory
import sys
WDPATH = "/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/paper/scripts/ARnetlab"
sys.path.insert(0, WDPATH)
# input and output
INPUT_PATH = '/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/paper/data/output/'
OUTPUT_PATH = '/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/paper/figures/'
SUPP_PATH = '/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/paper/suppl_figures/'


# %% IMPORT MODULES

# standard packages
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm as CMAP
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter

# specific packages
from cmcrameri import cm
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import geopandas as gpd
import xarray as xr

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
fs = 20

%matplotlib

# %% FUNCTIONS
def divide_by_ndec(x, pos):
    return f'{int(x / nyear)}'


# %% LOAD DATA

# TOPOGRAPHY (displayed for pagerank)
dem_ds = xr.open_dataset('/Users/tbraun/Desktop/projects/#A_PIKART_PIK/ARcatalog_shared/scripts/detection&tracking/input_files/hyd_glo_dem_0_75deg.nc')
dem = dem_ds['dem'].isel(time=0)

# %% PARAMETERS
    
# Choose if multiple testing correciton should be applied
significance_mode = 'corrected'  # or 'raw'
# Choose panel
PANEL = 'a'
# Confidence level
alpha = 0.1#05
# number of years
nyear = 83 


# %% PANEL A -  NODE STRENGTH

if PANEL == 'a':    
    # LOAD data
    d_nodestr = gpd.read_file(INPUT_PATH + "nodestr_centroid_consensus.gpkg", layer='nodestr')



    # Colormaps
    cmap0, cmap1, cmap2, cmap3, cmap4 = cm.grayC_r, cm.devon_r, CMAP.Purples, CMAP.Oranges, cm.imola#CMAP.Greens
    vmin, vmax = np.nanmin(d_nodestr.nodestr), np.nanquantile(d_nodestr.nodestr, .99)

    # POST-PROCESSING
    d_nodestr.dropna(subset=['nodestr'], inplace=True)
    d_nodestr = nplot.split_hexagons(d_nodestr)
    
    # Choose significance columns depending on mode
    if significance_mode == 'corrected':
        s1, s2, s3, s4 = 'signif_corr_1', 'signif_corr_2', 'signif_corr_3', 'signif_corr_4'
    else:
        s1, s2, s3, s4 = 'pval_1', 'pval_2', 'pval_3', 'pval_4'
        # Convert raw p-values to binary significance
        d_nodestr[s1] = (d_nodestr[s1] < alpha).astype(int)
        d_nodestr[s2] = (d_nodestr[s2] < alpha).astype(int)
        d_nodestr[s3] = (d_nodestr[s3] < alpha).astype(int)
        d_nodestr[s4] = (d_nodestr[s4] < alpha).astype(int)
    
    # Further processing for populations passing different numbers of tests
    pop0 = d_nodestr[~((d_nodestr[s1] == 1) |
                       ((d_nodestr[[s1, s4]] == 1).sum(axis=1) == 2) |
                       ((d_nodestr[[s1, s4, s2]] == 1).sum(axis=1) == 3) |
                       ((d_nodestr[[s1, s4, s2, s3]] == 1).sum(axis=1) == 4))]
    pop1 = d_nodestr[d_nodestr[s1] == 1]
    pop2 = d_nodestr[(d_nodestr[[s1, s4]] == 1).sum(axis=1) == 2]
    pop3 = d_nodestr[(d_nodestr[[s1, s4, s2]] == 1).sum(axis=1) == 3]
    pop4 = d_nodestr[(d_nodestr[[s1, s4, s2, s3]] == 1).sum(axis=1) == 4]
    # set geometry attributes
    pop0 = pop0.set_geometry('geometry')
    pop1 = pop1.set_geometry('geometry')
    pop2 = pop2.set_geometry('geometry')
    pop3 = pop3.set_geometry('geometry')
    pop4 = pop4.set_geometry('geometry')
    
    

    # F I G U R E 
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.EqualEarth()})
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, color='black')
    
    # Normalization for color scaling
    norm = Normalize(vmin=vmin, vmax=vmax)#LogNorm(vmin=vmin+1e-9, vmax=vmax)#
    
    # Plot each population with its corresponding colormap
    if not pop0.empty:
        pop0.plot(column='nodestr', cmap=cmap0, ax=ax,
                   transform=ccrs.PlateCarree(), norm=norm, edgecolor='black', linewidth=0, legend=False, alpha=.85)
    
    if not pop1.empty:
        pop1.plot(column='nodestr', cmap=cmap1, ax=ax,
                   transform=ccrs.PlateCarree(), norm=norm, edgecolor='black', linewidth=0, legend=False, alpha=.85)
    
    if not pop2.empty:
        pop2.plot(column='nodestr', cmap=cmap2, ax=ax,
                   transform=ccrs.PlateCarree(), norm=norm, edgecolor='black', linewidth=0, legend=False, alpha=.9)
    
    if not pop3.empty:
        pop3.plot(column='nodestr', cmap=cmap3, ax=ax,
                   transform=ccrs.PlateCarree(), norm=norm, edgecolor='black', linewidth=0, legend=False, alpha=.95)
    
    if not pop4.empty:
        pop4.plot(column='nodestr', cmap=cmap4, ax=ax,
                   transform=ccrs.PlateCarree(), norm=norm, edgecolor='black', linewidth=.8, legend=False)
    
    plt.show()
    plt.savefig('/Users/tbraun/Desktop/' + '/Fig3a.png', dpi=600, bbox_inches='tight', transparent=True)
#    plt.savefig(OUTPUT_PATH + '/Fig3a.png', dpi=300, bbox_inches='tight')
#    plt.savefig(SUPP_PATH + '/Fig3S2a.png', dpi=300, bbox_inches='tight')

    

    # SEPARATE COLORBAR PLOT
    formatter = FuncFormatter(divide_by_ndec)
    cbar_fig, cbar_axs = plt.subplots(1, 5, figsize=(25, 0.6))
    
    # Create and format each colorbar
    cbars = []
    cmaps = [cmap0, cmap1, cmap2, cmap3, cmap4]
    cbar_labels = ['no test passed', 'FRW', 'FRW+GCW', 'FRW+GCW+TCW', 'all tests passed']
    k=0
    for ax, cmap in zip(cbar_axs, cmaps):
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(cmap=cmap, norm=norm),
            cax=ax,
            orientation='horizontal',
            format=formatter,
            ticks=np.array([0, 5*nyear,10*nyear,15*nyear], dtype=int)
        )
        cbar.ax.tick_params(labelsize=fs)
        cbar.set_label(cbar_labels[k], color='black', fontsize=fs)
        cbars.append(cbar)
        k+=1
    
    # Add a single, centered label below all colorbars
    cbar_fig.text(0.5, -1.4, 'node strength (ARs/year)', ha='center', va='center', fontsize=fs, color='black')
    
    # Adjust layout to fit everything nicely
    plt.subplots_adjust(wspace=0.1, bottom=0.4)
    plt.show()
    plt.savefig('/Users/tbraun/Desktop/' + '/Fig3a_cbar.png', dpi=600, bbox_inches='tight', transparent=True)
#    plt.savefig(OUTPUT_PATH + '/Fig3a_cbar.png', dpi=300, bbox_inches='tight')
#    plt.savefig(SUPP_PATH + '/Fig3S2a_cbar.png', dpi=300, bbox_inches='tight')


# %% PANEL B - DIVERGENCE

elif PANEL == 'b':
    
    # LOAD data
    d_ndiv = gpd.read_file(INPUT_PATH + "divergence_centroid_consensus.gpkg", layer='ndiv')


    # --- POST-PROCESSING ---
    d_ndiv.dropna(subset=['ndiv'], inplace=True)
    d_ndiv = nplot.split_hexagons(d_ndiv)
    
    
    # Colormaps
    cmap0, cmap1, cmap2 = cm.grayC_r, cm.vik_r, cm.bam
    vmin, vmax = -np.nanmax(d_ndiv.ndiv), np.nanmax(d_ndiv.ndiv)
    
    # SIGNIFICANCE
    if significance_mode == 'corrected':
        s1, s2, s3, s4 = 'signif_corr_1', 'signif_corr_2', 'signif_corr_3', 'signif_corr_4'
    else:
        s1, s2, s3, s4 = 'pval_1', 'pval_2', 'pval_3', 'pval_4'
        d_ndiv[s1] = (d_ndiv[s1] < alpha).astype(int)
        d_ndiv[s2] = (d_ndiv[s2] < alpha).astype(int)
        d_ndiv[s3] = (d_ndiv[s3] < alpha).astype(int)
        d_ndiv[s4] = (d_ndiv[s4] < alpha).astype(int)
    
    # How many tests are passed?
    pop0 = d_ndiv[~((d_ndiv[s1] == 1) |
                    ((d_ndiv[[s1, s4]] == 1).sum(axis=1) == 2) |
                    ((d_ndiv[[s1, s4, s2]] == 1).sum(axis=1) == 3) |
                    ((d_ndiv[[s1, s4, s2, s3]] == 1).sum(axis=1) == 4))]
    pop1 = d_ndiv[d_ndiv[s1] == 1]
    pop4 = d_ndiv[(d_ndiv[[s1, s4, s2, s3]] == 1).sum(axis=1) == 4]
    
    # set geometry attributes
    pop0 = pop0.set_geometry('geometry')
    pop1 = pop1.set_geometry('geometry')
    pop4 = pop4.set_geometry('geometry')
    
    # F I G U R E 
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.EqualEarth()})
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, color='black')
    
    # Normalization for color scaling
    norm = Normalize(vmin=vmin, vmax=vmax)#LogNorm(vmin=vmin+1e-9, vmax=vmax)#
    
    # Plot each population with its corresponding colormap
    if not pop0.empty:
        pop0.plot(column='ndiv', cmap=cmap0, ax=ax,
                   transform=ccrs.PlateCarree(), norm=norm, edgecolor='black', linewidth=0, legend=False, alpha=.35)
    
    if not pop1.empty:
        pop1.plot(column='ndiv', cmap=cmap1, ax=ax,
                   transform=ccrs.PlateCarree(), norm=norm, edgecolor='black', linewidth=0, legend=False, alpha=.85)
    
    if not pop4.empty:
        pop4.plot(column='ndiv', cmap=cmap2, ax=ax,
                   transform=ccrs.PlateCarree(), norm=norm, edgecolor='black', linewidth=0.5, legend=False)
    plt.show()
#    plt.savefig(OUTPUT_PATH + '/Fig3b.png', dpi=300, bbox_inches='tight')
    plt.savefig(SUPP_PATH + '/Fig3S2b.png', dpi=300, bbox_inches='tight')



    # SEPARATE COLORBAR PLOT
    formatter = FuncFormatter(divide_by_ndec)
    cbar_fig, cbar_axs = plt.subplots(1, 3, figsize=(25, 0.6))
    
    # Create and format each colorbar
    cbars = []
    cmaps = [cmap0, cmap1, cmap2, cmap3, cmap4]
    cbar_labels = ['no test passed', 'FRW', 'all tests passed']
    k=0
    for ax, cmap in zip(cbar_axs, cmaps):
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(cmap=cmap, norm=norm),
            cax=ax,
            orientation='horizontal',
            format=formatter#,
            #ticks=np.array([0, 5*nyear,10*nyear,15*nyear], dtype=int)
        )
        cbar.ax.tick_params(labelsize=fs)
        cbar.set_label(cbar_labels[k], color='black', fontsize=fs)
        cbars.append(cbar)
        k+=1
    
    # Add a single, centered label below all colorbars
    cbar_fig.text(0.5, -1.2, 'node divergence (ARs/year)', ha='center', va='center', fontsize=fs, color='black')
    
    # Adjust layout to fit everything nicely
    plt.subplots_adjust(wspace=0.1, bottom=0.4)
    plt.show()
    plt.savefig(OUTPUT_PATH + '/Fig3b_cbar.png', dpi=300, bbox_inches='tight')
#    plt.savefig(SUPP_PATH + '/Fig3S2b_cbar.png', dpi=300, bbox_inches='tight')




# %% PANEL C - PAGERANK


elif PANEL == 'c':     
    """
    Key Takeaways
    PageRank emphasizes persistent, flow-retaining structures rather than just termination points.
    Your high PageRank regions align with moisture accumulation, recirculation, or persistent guidance by winds/topography.
    Degree centrality is more about sheer number of connections, while PageRank tells a deeper dynamical story about moisture transport stability and trapping.
    """
    
    # LOAD data
    d_pagernk = gpd.read_file(INPUT_PATH + "pagerank_head_consensus.gpkg", layer='pagernk')
    
    
    
    # POST-PROCESSING
    d_pagernk.dropna(subset=['pagernk'], inplace=True)
    d_pagernk = nplot.split_hexagons(d_pagernk)
    
    # Colormaps
    cmap0, cmap1, cmap2, cmap3, cmap4 = cm.grayC_r, cm.devon_r, CMAP.Purples, CMAP.Oranges, cm.imola#CMAP.Greens
    vmin, vmax = np.nanmin(d_pagernk.pagernk), np.nanquantile(d_pagernk.pagernk, .99)

    # Group by significance pattern
    pop0 = d_pagernk[~((d_pagernk['signif_corr_1'] == 1) |
                       ((d_pagernk[['signif_corr_1', 'signif_corr_4']] == 1).sum(axis=1) == 2))]
    pop1 = d_pagernk[(d_pagernk['signif_corr_1'] == 1) &
                     ((d_pagernk[['signif_corr_1', 'signif_corr_4']] == 1).sum(axis=1) != 2)]
    pop2 = d_pagernk[(d_pagernk[['signif_corr_1', 'signif_corr_4']] == 1).sum(axis=1) == 2]
    pop0 = pop0.set_geometry('geometry')
    pop1 = pop1.set_geometry('geometry')
    pop2 = pop2.set_geometry('geometry')
    
    
    # Choose significance columns depending on mode
    if significance_mode == 'corrected':
        s1, s2, s3, s4 = 'signif_corr_1', 'signif_corr_2', 'signif_corr_3', 'signif_corr_4'
    else:
        s1, s2, s3, s4 = 'pval_1', 'pval_2', 'pval_3', 'pval_4'
        # Convert raw p-values to binary significance
        d_pagernk[s1] = (d_pagernk[s1] < alpha).astype(int)
        d_pagernk[s2] = (d_pagernk[s2] < alpha).astype(int)
        d_pagernk[s3] = (d_pagernk[s3] < alpha).astype(int)
        d_pagernk[s4] = (d_pagernk[s4] < alpha).astype(int)
    
    # Further processing for populations passing different numbers of tests
    pop0 = d_pagernk[~((d_pagernk[s1] == 1) |
                       ((d_pagernk[[s1, s4]] == 1).sum(axis=1) == 2) |
                       ((d_pagernk[[s1, s4, s2]] == 1).sum(axis=1) == 3) |
                       ((d_pagernk[[s1, s4, s2, s3]] == 1).sum(axis=1) == 4))]
    pop1 = d_pagernk[d_pagernk[s1] == 1]
    pop2 = d_pagernk[(d_pagernk[[s1, s4]] == 1).sum(axis=1) == 2]
    pop3 = d_pagernk[(d_pagernk[[s1, s4, s2]] == 1).sum(axis=1) == 3]
    pop4 = d_pagernk[(d_pagernk[[s1, s4, s2, s3]] == 1).sum(axis=1) == 4]
    # set geometry attributes
    pop0 = pop0.set_geometry('geometry')
    pop1 = pop1.set_geometry('geometry')
    pop2 = pop2.set_geometry('geometry')
    pop3 = pop3.set_geometry('geometry')
    pop4 = pop4.set_geometry('geometry')

    
    
    # F I G U R E 
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.EqualEarth()})
    ax.set_global()
    
    # Normalization for color scaling
    norm = Normalize(vmin=vmin, vmax=vmax)#LogNorm(vmin=vmin+1e-9, vmax=vmax)#
    
    # Plot each population with its corresponding colormap
    if not pop0.empty:
        pop0.plot(column='pagernk', cmap=cmap0, ax=ax,
                   transform=ccrs.PlateCarree(), norm=norm, edgecolor='black', linewidth=0, legend=False, alpha=.85)
    
    if not pop1.empty:
        pop1.plot(column='pagernk', cmap=cmap1, ax=ax,
                   transform=ccrs.PlateCarree(), norm=norm, edgecolor='black', linewidth=0, legend=False, alpha=.85)
    
    if not pop2.empty:
        pop2.plot(column='pagernk', cmap=cmap2, ax=ax,
                   transform=ccrs.PlateCarree(), norm=norm, edgecolor='black', linewidth=0, legend=False, alpha=.9)
    
    if not pop3.empty:
        pop3.plot(column='pagernk', cmap=cmap3, ax=ax,
                   transform=ccrs.PlateCarree(), norm=norm, edgecolor='black', linewidth=0, legend=False, alpha=.95)
    
    # DEM
    contour_levels = np.linspace(dem.min(), dem.max(), 10)  # Adjust the number of contour levels as needed
    cs = ax.contour(dem.lon, dem.lat, dem, levels=contour_levels, colors='dimgray', linewidths = 1, transform=ccrs.PlateCarree(), zorder=1)
    ax.add_feature(cfeature.COASTLINE, color='black', zorder=2)

    if not pop4.empty:
        pop4.plot(column='pagernk', cmap=cmap4, ax=ax,
                   transform=ccrs.PlateCarree(), norm=norm, edgecolor='black', linewidth=.8, legend=False, alpha=.9)
    
    plt.show()
    plt.savefig(OUTPUT_PATH + '/Fig3c.png', dpi=300, bbox_inches='tight')

    
    # SEPARATE COLORBAR PLOT
    cbar_fig, cbar_axs = plt.subplots(1, 5, figsize=(25, 0.6))
    
    # Create and format each colorbar
    cbars = []
    cmaps = [cmap0, cmap1, cmap2, cmap3, cmap4]
    cbar_labels = ['no test passed', 'FRW', 'FRW+GCW', 'FRW+GCW+TCW', 'all tests passed']
    k=0
    for ax, cmap in zip(cbar_axs, cmaps):
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(cmap=cmap, norm=norm),
            cax=ax,
            orientation='horizontal'
        )
        cbar.ax.tick_params(labelsize=fs)
        cbar.set_label(cbar_labels[k], color='black', fontsize=fs)
        cbars.append(cbar)
        k+=1
    
    # Add a single, centered label below all colorbars
    cbar_fig.text(0.5, -1.2, 'PageRank score', ha='center', va='center', fontsize=fs, color='black')
    
    # Adjust layout to fit everything nicely
    plt.subplots_adjust(wspace=0.1, bottom=0.4)
    plt.show()
    plt.savefig(OUTPUT_PATH + '/Fig3c_cbar.png', dpi=300, bbox_inches='tight')
#    plt.savefig(SUPP_PATH + '/Fig3S2a_cbar.png', dpi=300, bbox_inches='tight')

