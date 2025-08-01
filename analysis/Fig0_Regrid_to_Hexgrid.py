# Copyright (C) 2025 by
# Tobias Braun

#------------------ PATHS ---------------------------#
import sys
WDPATH = "/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/paper/scripts/ARnetlab"
sys.path.insert(0, WDPATH)
# input and output
INPUT_PATH = '/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/paper/data/'
OUTPUT_PATH = '/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/paper/data/'
MERRAPATH = "/Users/tbraun/Desktop/projects/#A_PIKART_PIK/production_files/data/MERRA2/PIKARTV1_lagrangian/"



# %% IMPORT MODULES

# standard packages
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt

# specific packages
from tqdm import tqdm
from h3 import h3

# my packages
import ARnet_sub as artn


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

# %% LOAD DATA

# PIKART-1
#d_ars_pikart = pd.read_csv(INPUT_PATH + 'CATALOG_1940-2023_PIKARTV1_Vallejo-Bernal_Braun_etal_2025_ERA5_full.csv')
#d_ars_pikart['time'] = pd.to_datetime(d_ars_pikart['time'])

# PIKART-1 MERRA2
d_ars_pikart = pd.read_csv(MERRAPATH + '1980-2019_PIKARTV1_Vallejo-Bernal_Braun_etal_2025_MERRA2.csv')
d_ars_pikart['time'] = pd.to_datetime(d_ars_pikart['time'])

# tARget-4
d_ars_target = pd.read_pickle(INPUT_PATH + 'tARget_globalARcatalog_ERA5_1940-2023_v4.0_converted.pkl')
d_ars_target['time'] = pd.to_datetime(d_ars_target['time'])



# %% REGRIDDING

## NOTE: tARget's longitudes range between 0 and 360!
# But h3 is smart enough to automatically transform them correctly.

# PARAMETERS
catalog = 'pikart'


# Initialize
if catalog == 'pikart':
    ARcat = d_ars_pikart.copy()
    l_arloc = ['centroid', 'head', 'core']
elif catalog == 'target':
    ARcat = d_ars_target.copy()
    l_arloc = ['centroid', 'head']
# AR locators that will be regridded

l_arloc = ['centroid']

Nloc = len(l_arloc)
Nres = 3

# Drop contours, axes, and all columns starting with 'insu_' or 'conti_'
ARcat = ARcat.drop(columns=[col for col in ARcat.columns 
                      if col in ['contour_lon', 'contour_lat', 'axis_lon', 'axis_lat'] 
                      or col.startswith('insu_') 
                      or col.startswith('conti_')])



## LOOP over different AR locators
for nloc in tqdm(range(Nloc)): 
    ## Set locator
    loc = l_arloc[nloc]
    ### iterate over different spatial resolutions
    for h3_resolution in tqdm(range(Nres)):
        xcoord, ycoord = artn.ARlocator(loc)
        
        # Convert latitudes and longitudes to hexagon index
        idxcol = 'hex_idx_' + loc + '_res' + str(h3_resolution)
        ARcat[idxcol] = ARcat.apply(lambda row: pd.Series(h3.geo_to_h3(row[ycoord], row[xcoord], h3_resolution)), axis=1)

        # Convert hexagon indices to centroid coordinates of hexagon
        ARcat[['hex_' + ycoord + '_res' + str(h3_resolution), 'hex_' + xcoord + '_res' + str(h3_resolution)]] = ARcat.apply(lambda row: pd.Series(h3.h3_to_geo(row[idxcol])), axis=1)

        # Normalize longitudes from [0, 360) → [-180, 180)
        lon_col = 'hex_' + xcoord + '_res' + str(h3_resolution)
        ARcat[lon_col] = ((ARcat[lon_col] + 180) % 360) - 180



# %% SAVE

# sanity check
print(ARcat.columns)

# Write out
ARcat.to_pickle(OUTPUT_PATH + 'pikart_merra2_hex.pkl')  
