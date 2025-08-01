# Copyright (C) 2025 by
# Tobias Braun

#------------------ PATH ---------------------------#
import sys
PATH = "/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/scripts"
sys.path.insert(0, PATH)



# %% IMPORT MODULES

# standard packages
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import warnings
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
from scipy import stats

# specific packages
from h3 import h3
import hashlib
from tqdm import tqdm
import networkx as nx
import pytz
from sklearn.preprocessing import binarize
from timezonefinder import TimezoneFinder

# %% INTERNAL HELPER FUNCTIONS
# The functions below are not meant to be called from a script but are secondary
# functions that are called by the main functions below.


def _condition_on_feature(ARcat, feature, average_feature, bins=5, return_bins=True):
    """
    Groups ARcat based on a feature, either averaging over trajectories or using raw values.

    Parameters:
    - ARcat (pd.DataFrame): The AR catalog.
    - feature (str): The column name of the feature to condition on.
    - average_feature (bool): Whether to average the feature over trajectories.
    - bins (int or list): Either the number of bins (int) or explicit bin edges (list).
    - return_bins (bool): If True, return the bin edges alongside the groups.

    Returns:
    - l_windowed_dfs (list of pd.DataFrame): The list of dataframes for each bin.
    - bin_edges (list, optional): The edges of the bins, if return_bins is True.
    """
    # Generate a new column that is the track-averaged mean feature
    if average_feature:
        ARcat['feat_bin'] = ARcat.groupby(['trackid'])[feature].transform('mean')
    else:
        ARcat['feat_bin'] = ARcat[feature]
    
    # Apply binning based on bins argument
    if isinstance(bins, int):
        # Use pd.qcut for quantile-based bins
        ARcat['feat_bin'], bin_edges = pd.qcut(ARcat['feat_bin'], q=bins, labels=False, retbins=True)
    elif isinstance(bins, list):
        # Use pd.cut for user-defined bins
        ARcat['feat_bin'], bin_edges = pd.cut(ARcat['feat_bin'], bins=bins, labels=False, retbins=True)
    else:
        raise ValueError("The 'bins' parameter must be an integer or a list of bin edges.")
    
    ARcat['feat_bin'] += 1  # Increment bins to start from 1 instead of 0
    
    # Group by bins
    p_feature_groups = ARcat.groupby('feat_bin')
    l_windowed_dfs = [group for name, group in p_feature_groups]
    
    # Return groups and bin edges if requested
    if return_bins:
        return l_windowed_dfs, bin_edges
    else:
        return l_windowed_dfs
    
    
def _birth_death(AR):
    """
    Simple function that helps to extract the first (genesis) and last (termination) instance of the AR track
    for lifecycle-conditioned AR networks.

    Parameters:
    - AR (pd.DataFrame): An individual AR trajectory.

    Returns:
    - [0, -1] (list): A list that is used to extract the first and last instance of the AR track.
    """
    return [0, -1]


def _birth_lf(AR):
    """
    Simple function that helps to extract the first (genesis) and first landfall instance of the AR track
    for lifecycle-conditioned AR networks.

    Parameters:
    - AR (pd.DataFrame): An individual AR trajectory.

    Returns:
    - [0, idx] (list): A list that is used to extract the first and first landfall instance of the AR track.
    """
    tmp_cond = (~np.isnan(AR.lf_lon))
    if np.any(tmp_cond):
        idx = np.where(tmp_cond)[0][0]
        return [0, idx]


def _lf_death(AR):
    """
    Simple function that helps to extract the first landfall and last (termination) instance of the AR track
    for lifecycle-conditioned AR networks.

    Parameters:
    - AR (pd.DataFrame): An individual AR trajectory.

    Returns:
    - [0, idx] (list): A list that is used to extract the landfall and last instance of the AR track.
    """
    tmp_cond = (~np.isnan(AR.lf_lon))
    if np.any(tmp_cond):
        idx = np.where(tmp_cond)[0][0]
        return [idx, -1]


def _downsample_matrix(matrix, res):
    """
    Downsample the continental mask as a matrix to a coarser grid using the most frequent value (mode) 
    in each block. Assumes the native resolution is 0.5 degrees.

    :param matrix: 2D numpy array of integers representing the original high-resolution grid.
    :param res: Target resolution in degrees (must be a multiple of 0.5, e.g., 2.0).
    :return: 2D numpy array of the same type as `matrix`, downsampled to the target resolution.
    """
    def _most_frequent_int(arr):
        return stats.mode(arr.ravel(), axis=None, keepdims=False).mode

    factor = int(res / 0.5)
    original_shape = matrix.shape
    downsampled_shape = (original_shape[0] // factor, original_shape[1] // factor)
    
    downsampled_matrix = np.empty(downsampled_shape, dtype=matrix.dtype)
    for i in range(downsampled_shape[0]):
        for j in range(downsampled_shape[1]):
            block = matrix[i*factor:(i+1)*factor, j*factor:(j+1)*factor]
            downsampled_matrix[i, j] = _most_frequent_int(block)
    
    return downsampled_matrix



def _get_season(dt):
    """ Return meteorological season (DJF, MAM, JJA, SON) for a given numeric month. """
    month = dt.month
    if month in [12, 1, 2]:
        return 'DJF'
    elif month in [3, 4, 5]:
        return 'MAM'
    elif month in [6, 7, 8]:
        return 'JJA'
    elif month in [9, 10, 11]:
        return 'SON'
    
    
def _get_local_time(row, tf, xcoord, ycoord):
    """
    Convert UTC time in a DataFrame row to local time based on the geographic coordinates.

    Parameters:
    -----------
    row : pd.Series
        A row of the ARcat DataFrame representing one AR observation.
    tf : TimezoneFinder
        An instance of TimezoneFinder to determine the timezone from coordinates.
    xcoord : str
        Name of the longitude column in the DataFrame.
    ycoord : str
        Name of the latitude column in the DataFrame.

    Returns:
    --------
    datetime or None
        Localized datetime object if timezone is found; otherwise None.
    """
    # Find timezone string for the location given by longitude and latitude
    tz_str = tf.timezone_at(lng=row[xcoord], lat=row[ycoord])

    if tz_str:
        # Create timezone object from string
        tz = pytz.timezone(tz_str)

        # Localize the UTC time (assume input time is naive UTC)
        utc_time = row['time'].replace(tzinfo=pytz.utc)

        # Convert UTC time to local timezone
        local_time = utc_time.astimezone(tz)
        return local_time
    else:
        # Return None if no timezone could be determined for the location
        return None


def _is_daytime(dt):
    """ Classify time as 'day' (6–18h) or 'night' (else). """
    hour = dt.hour
    return 'day' if 6 <= hour < 18 else 'night'




def _lat_lon_to_hash(lat, lon):
    """
    Generate a stable integer hash from latitude and longitude coordinates.

    This function is used to uniquely identify rectangular grid cells
    by hashing their center coordinates. It ensures compatibility with
    indexing schemes like H3 used for hexagonal grids.

    Parameters:
    -----------
    lat : float
        Latitude value (usually regridded to fixed resolution).
    lon : float
        Longitude value (usually regridded to fixed resolution).

    Returns:
    --------
    int
        Integer hash value derived from SHA256 digest of formatted coordinates.
    """
    lat_lon_str = f"{lat:.6f}_{lon:.6f}"  # Ensure consistent formatting
    hash_bytes = hashlib.sha256(lat_lon_str.encode()).digest()
    return int.from_bytes(hash_bytes[:8], byteorder='big', signed=False)  # Use first 8 bytes for 64-bit int




def _invert_hash(h, grid_type, coord_dict=None):
    """
    Retrieve the original latitude and longitude coordinates from a hashed index.

    For rectangular grids, this uses a reverse-lookup dictionary.
    For hexagonal grids, the H3 library is used to recover the approximate center.

    Parameters:
    -----------
    h : int or str
        Hashed coordinate index (int for rectangular, str for hexagonal).
    grid_type : str
        Type of spatial grid: 'rectangular' or 'hexagonal'.
    coord_dict : dict, optional
        Dictionary mapping integer hashes to (lat, lon) tuples (required for rectangular grids).

    Returns:
    --------
    tuple
        Tuple of (lat, lon) if successful; otherwise "Not Found" (for rectangular grid).
    """
    if grid_type == 'rectangular':
        return coord_dict.get(h, "Not Found")  # Fallback in case hash is not in dictionary
    elif grid_type == 'hexagonal':
        return h3.h3_to_geo(h)  # Returns (lat, lon) tuple from hexagon index



def _set_diagonal_zero(A_sparse):
    """
    Set all diagonal elements of a sparse matrix to zero, discarding self-links in the AR network.

    Parameters:
    - A_sparse : scipy.sparse.coo_matrix
        A sparse (adjacency) matrix in COO (Coordinate) format whose diagonal entries will be set to zero.

    Returns:
    - A_sparse : scipy.sparse.coo_matrix
        A new COO matrix with the same shape and non-diagonal entries preserved,
        but with all diagonal entries set to zero, i.e., no self-links.
    """
    # Extract data, row, and column indices from the COO matrix
    data, row, col = A_sparse.data, A_sparse.row, A_sparse.col

    # Find boolean mask for diagonal elements (i.e., where row index == column index)
    diag_indices = row == col

    # Set the values of diagonal elements to zero in the data array
    data[diag_indices] = 0

    # Reconstruct the COO matrix with updated data
    A_sparse = coo_matrix((data, (row, col)), shape=A_sparse.shape)
    
    return A_sparse


# %% ARTN generators
# The functions below are the core essential functions to generate an AR network.


def ARlocator(loc, grid_type='rectangular'):
    """
    Selects the appropriate coordinate fields for locating an Atmospheric River (AR) feature 
    based on the chosen locator strategy and grid type.

    Parameters
    ----------
    loc : str
        Specifies the AR locator strategy. Options include:
            - 'centroid': geometric centroid of the AR feature.
            - 'head': "head" of the AR.
            - 'core': location of maximum Integrated Vapor Transport (IVT).
            - 'tail': t"tail" of the AR.
            - 'li': land intersection coordinates

    grid_type : str, optional (default='rectangular')
        Specifies the type of spatial grid used in the AR catalog.
        Options:
            - 'rectangular': standard latitude-longitude grid.
            - 'hexagonal': hexagon-based grid; prefixes 'hex_' to the coordinate names.

    Returns
    -------
    xcoord : str
        The column name for the x (longitude or grid-x) coordinate in the AR catalog.
    
    ycoord : str
        The column name for the y (latitude or grid-y) coordinate in the AR catalog.
    
    Notes
    -----
    This function is typically used to extract coordinate field names from a catalog 
    of ARs for spatial indexing or transport network generation. The column names returned 
    should correspond to those already present in the AR catalog DataFrame.
    
    Raises
    ------
    ValueError
        If an unsupported locator is provided.
    """

    # Match the AR locator to the corresponding coordinate columns
    if loc == 'core':
        xcoord, ycoord = 'core_lon', 'core_lat'
    elif loc == 'centroid':
        xcoord, ycoord = 'centroid_lon', 'centroid_lat'
    elif loc == 'head':
        xcoord, ycoord = 'head_lon', 'head_lat'
    elif loc == 'tail':
        xcoord, ycoord = 'tail_lon', 'tail_lat'
    # currently not included in the catalog
    elif loc == 'li':
        xcoord, ycoord = 'li_lon', 'li_lat'
    else:
        raise ValueError(f"Unsupported AR locator: '{loc}'. Must be one of "
                         "'max_ivt', 'li', 'centroid', 'head', or 'tail'.")

    # Prefix with 'hex_' if the hexagonal grid is used
    if grid_type == 'hexagonal':
        xcoord = 'hex_' + xcoord
        ycoord = 'hex_' + ycoord

    return xcoord, ycoord



def regrid(ARcat, res, loc):
    """
    Regrid AR coordinates to a coarser rectangular grid. Only for rectangular grid!

    Parameters:
    -----------
    ARcat : pd.DataFrame
        DataFrame AR catalog.
    res : float
        Spatial resolution (in degrees) for the coarse rectangular grid.
    loc : str
        Location string used by ARlocator() to identify column names for longitude and latitude.

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Arrays of regridded longitudes and latitudes (coarse grid centers) for each point in ARcat.
    """
    print('...regridding...')

    # Identify the names of the longitude and latitude columns from `loc`
    xcoord, ycoord = ARlocator(loc, grid_type='rectangular')

    # Create 1D arrays of coarse grid centers along longitude and latitude
    a_coarse_lons = np.arange(-180, 180 + res, res)
    a_coarse_lats = np.arange(-90, 90 + res, res)

    # For each point in ARcat, find the index of the closest coarse grid latitude and longitude
    lat_idx = np.abs(ARcat[ycoord].values[:, None] - a_coarse_lats).argmin(axis=1)
    lon_idx = np.abs(ARcat[xcoord].values[:, None] - a_coarse_lons).argmin(axis=1)

    # Return the matched coarse grid center coordinates
    return a_coarse_lons[lon_idx], a_coarse_lats[lat_idx]


def constrain(ARcat, xcoord, ycoord, T, X, fully_contained):
    """
    Apply temporal and spatial constraints to an AR catalog DataFrame.

    Parameters:
    -----------
    ARcat : pd.DataFrame
        AR catalog, including 'time', 'trackid', and spatial coordinates.
    xcoord : str
        Name of the longitude coordinate column in ARcat (AR locator).
    ycoord : str
        Name of the latitude coordinate column in ARcat (AR locator).
    T : list or None
        List of years (int) defining the time window (e.g. [1980, 2020]).
        If None, no temporal filtering is applied.
    X : str, list, or None
        Spatial constraint:
        - If str and not 'global', assumed to be continent name (e.g. 'Europe').
        - If 'global' or None, no spatial filtering.
        - Otherwise, expected to be a bounding box [lat_min, lat_max, lon_min, lon_max].
    fully_contained : bool
        If True and X is a continent, only keep tracks fully contained within the continent or sea.
        If False, keep tracks that intersect the continent at any point.

    Returns:
    --------
    pd.DataFrame
        Subset of ARcat filtered according to the temporal and spatial constraints.
    """
    print('...applying constraints...')

    ### CONSTRAIN TIME WINDOW ###
    if T is not None:
        # Define start and end timestamps for filtering (start of first year to end of last year)
        t0, tfin = str(T[0]) + '-01-01', str(T[-1]) + '-12-31'

        # Filter ARcat to keep only times within the range
        d_ars_subset0 = ARcat[(ARcat['time'] > t0) & (ARcat['time'] < tfin)]

        # Remove the first and last AR tracks (may be incomplete or truncated)
        l_filt = list(d_ars_subset0.groupby('trackid'))[1:-1]
        d_ars_subset0 = pd.concat([group for _, group in l_filt])

    else:
        # If no temporal constraint, keep all data
        d_ars_subset0 = ARcat.copy()

    ### CONSTRAIN SPATIAL EXTENT ###
    if isinstance(X, str) and X != 'global':
        # CONTINENTAL constraint

        # Check required 'continent' column exists in DataFrame
        assert 'continent' in d_ars_subset0.columns, \
            "Error: The required column 'continent' is missing in the AR catalog!"

        # Group by track to apply spatial cropping on full tracks
        d_ars_subset1 = d_ars_subset0.groupby('trackid')

        # Define local function for cropping tracks based on continent membership
        def _continental_cropper(AR, fully_contained):
            if fully_contained:
                # Keep track only if *all* points are on continent X or 'Sea' AND the track includes continent X somewhere
                cond = ((AR.continent == X) | (AR.continent == 'Sea')).all() and (X in AR.continent.values)
            else:
                # Keep track if it touches continent X anywhere
                cond = (X in AR.continent.values)
            if cond:
                return AR
            else:
                # Otherwise discard the track
                return pd.DataFrame()

        # Suppress pandas deprecation warnings during group apply
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="pandas")

        # Apply the continental cropper to each track, reset index after
        d_ars_subset = d_ars_subset1.apply(lambda AR: _continental_cropper(AR, fully_contained)).reset_index(drop=True)

    elif X == 'global' or (X is None):
        # GLOBAL constraint or no spatial constraint: keep all tracks
        d_ars_subset = d_ars_subset0

    else:
        # SPATIAL WINDOW constraint (bounding box)
        # Expect X as [lat_min, lat_max, lon_min, lon_max]

        # Group by trackid for cropping full tracks by bounding box
        d_ars_subset1 = d_ars_subset0.groupby('trackid')

        # Local function to check if all points in track are inside bounding box
        def _spatial_cropper(AR):
            cond1 = (AR[ycoord] > X[0]).all()  # all lat > lat_min
            cond2 = (AR[ycoord] < X[1]).all()  # all lat < lat_max
            cond3 = (AR[xcoord] > X[2]).all()  # all lon > lon_min
            cond4 = (AR[xcoord] < X[3]).all()  # all lon < lon_max

            if cond1 and cond2 and cond3 and cond4:
                return AR
            else:
                # Discard track if any point falls outside bounding box
                return pd.DataFrame()

        # Apply spatial cropper, reset index after
        d_ars_subset = d_ars_subset1.apply(_spatial_cropper).reset_index(drop=True)

    # Return the constrained subset DataFrame
    return d_ars_subset


def add_local_time(ARcat, loc, grid_type):
    """
    Adds a 'local_time' column to the AR catalog DataFrame by converting UTC times to local times 
    based on the geographic coordinates and grid type. 

    Parameters:
    -----------
    ARcat : pd.DataFrame
        AR catalog, including time and coordinates.
    loc : dict or similar
        AR locator.
    grid_type : str
        Type of grid ('rectangular', 'hexagonal', etc.) used by ARlocator to find coordinate columns.

    Returns:
    --------
    pd.DataFrame
        ARcat with an additional 'local_time' column containing localized timestamps.
    """
    # Get coordinate column names for longitude and latitude based on location and grid type
    xcoord, ycoord = ARlocator(loc, grid_type)

    # Initialize a TimezoneFinder instance to map lat/lon to timezones
    tf = TimezoneFinder()

    # Apply _get_local_time row-wise to convert UTC times to local times for each row
    ARcat['local_time'] = ARcat.apply(lambda row: _get_local_time(row, tf, xcoord, ycoord), axis=1)

    return ARcat




def condition(ARcat, cond, bins=5, return_bins=True):
    """
    Split AR catalog into subsets based on a specified condition.

    Parameters:
    - ARcat (pd.DataFrame): Atmospheric river catalog.
    - cond (str): Conditioning criterion (e.g., 'season', 'daytime', 'IVT_mean', 'lifetime', etc.).
    - bins (int or array-like): Number of bins or bin edges for some of the features which are continuous.
    - return_bins (bool): Whether to return bin edges (currently only printed).

    Returns:
    - List[pd.DataFrame]: List of ARcat subsets fulfilling each bin/condition.
    
    Notes:
    - Some conditions may fragment AR trajectories (e.g., 'daytime').
    - Bin edges are printed if applicable but not returned.
    """
    print('...applying conditions...')

    if cond is None:
        return [ARcat]

    # Group by season
    if cond == 'season':
        ARcat['season'] = ARcat['time'].apply(_get_season)
        seasonal_groups = ARcat.groupby('season')
        print(f"Seasonal bins: {seasonal_groups.groups.keys()}")
        l_windowed_dfs = [group for name, group in seasonal_groups]

    # Split tracks into day/night subtracks
    elif cond == 'daytime':
        ARcat['day_time'] = ARcat['local_time'].apply(_is_daytime)

        def _assign_subtracks(group):
            group = group.copy()
            group['subtracks'] = (group['day_time'] != group['day_time'].shift()).cumsum()
            return group

        ARcat = ARcat.groupby('trackid').apply(_assign_subtracks).reset_index(drop=True)
        ARcat['trackid'] = ARcat['trackid'].astype(str) + '_' + ARcat['subtracks'].astype(str)

        # Filter short subtracks
        l_windowed_dfs0 = [group for name, group in ARcat.groupby('trackid')]
        l_filtered_groups = [df for df in l_windowed_dfs0 if df.shape[0] > 1]
        d_filtered = pd.concat(l_filtered_groups, ignore_index=True)

        daynight_groups = d_filtered.groupby('day_time')
        print(f"Daytime bins: {daynight_groups.groups.keys()}")
        l_windowed_dfs = [group for name, group in daynight_groups]

    # Group by inland penetration
    elif cond == 'IP':
        ip_groups = ARcat.groupby('inland_pen')
        print(f"Inland penetration bins: {ip_groups.groups.keys()}")
        l_windowed_dfs = [group for name, group in ip_groups]

    # Group by some teleconnection regime (requires adding a 'teleconnection' column beforehand tho)
    elif cond == 'teleconnection':
        tele_groups = ARcat.groupby('teleconnection')
        print(f"Teleconnection bins: {tele_groups.groups.keys()}")
        l_windowed_dfs = [group for name, group in tele_groups]

    # Binning based on trajectory-averaged feature of an AR
    elif cond in ['IVT_mean', 'area', 'length', 'width', 'perimeter', 'LW_ratio', 'cent_vel', 'ocean', 'land']:
        l_windowed_dfs, bin_edges = _condition_on_feature(
            ARcat=ARcat, average_feature=True, feature=cond, bins=bins, return_bins=return_bins
        )
        print(f"Feature '{cond}' bins: {len(l_windowed_dfs)} groups")
        print(f"Bin edges for '{cond}': {bin_edges}")

    # Binning based on lifecycle instance of each AR
    elif cond in ['lifecycle']:
        l_windowed_dfs, bin_edges = _condition_on_feature(
            ARcat=ARcat, average_feature=False, feature=cond, bins=bins, return_bins=return_bins
        )
        print(f"Feature '{cond}' bins: {len(l_windowed_dfs)} groups")
        print(f"Bin edges for '{cond}': {bin_edges}")

    # Binning based on AR lifetimes (quantile binning)
    elif cond == 'lifetime':
        if isinstance(bins, int):
            nbins = bins
        else:
            nbins = bins.size

        # Compute lifetime per track
        lifetimes = ARcat.groupby('trackid')['time'].agg(lambda x: (x.max() - x.min()).total_seconds())
        ARcat['lifetime'] = ARcat['trackid'].map(lifetimes)

        # Assign lifetime bins
        ARcat = ARcat.sort_values('lifetime').reset_index(drop=True)
        total_rows = len(ARcat)
        rows_per_group = total_rows // nbins
        ARcat['lifetime_bin'] = pd.Series(ARcat.index // rows_per_group).clip(upper=nbins - 1)

        lifetime_groups = ARcat.groupby('lifetime_bin')
        print(f"Lifetime bins: {list(lifetime_groups.groups.keys())}")
        bin_edges = pd.Series(ARcat['lifetime']).quantile([i / nbins for i in range(nbins + 1)]).values
        print(f"Bin edges for 'lifetime': {bin_edges}")
        l_windowed_dfs = [group for name, group in lifetime_groups]

    else:
        print(f"Warning: Unrecognized condition '{cond}'. Returning original dataframe.")
        l_windowed_dfs = [ARcat]

    return l_windowed_dfs


def condition_backwards_to_entry_region(
    ARcat, d_node_comm, 
    id_col='trackid', node_col='coord_idx', comm_col='community', 
    comm_id=None, track_after_entry=True
):
    """
    Filter AR trajectories so that only the segments that feed into a specified community 
    region (provided via d_node_comm) are retained.

    Parameters:
    - ARcat : pd.DataFrame
        Atmospheric river catalog with at least `trackid`, `time`, and `head_hex_id` columns.
    - d_node_comm : pd.DataFrame
        DataFrame with at least columns 'hex_id' and a community assignment column (e.g., 'community').
    - id_col : str
        Column name for trajectory ID in ARcat (default: 'trackid').
    - node_col : str
        Column name for hex ID in ARcat (default: 'coord_idx').
    - comm_col : str
        Column in d_node_comm specifying the community assignment (default: 'community').
    - comm_id : int or str
        Community ID to use as the entry region. If None, uses all non-negative (unfiltered) communities.
    - track_after_entry : bool
        If True, keeps AR segments from the start until they leave the region after entering it.
        If False, keeps segments only up to and including the first entry point.

    Returns:
    - ARcat_truncated : pd.DataFrame
        Subset of ARcat containing only segments that reach the entry region.
    """
    if comm_id is None:
        entry_region_hexes = set(d_node_comm.loc[d_node_comm[comm_col] >= 0, 'hex_id'])
        print(f"...filtering ARs entering any unfiltered community region ({len(entry_region_hexes)} hexes)...")
    else:
        entry_region_hexes = set(d_node_comm.loc[d_node_comm[comm_col] == comm_id, 'hex_id'])
        print(f"...filtering ARs entering community {comm_id} ({len(entry_region_hexes)} hexes)...")

    kept_segments = []

    for tid, group in ARcat.groupby(id_col):
        group_sorted = group.sort_values('time')
        hex_sequence = group_sorted[node_col].tolist()

        try:
            entry_idx = next(i for i, hex_id in enumerate(hex_sequence) if hex_id in entry_region_hexes)
        except StopIteration:
            continue  # No entry into the community

        if track_after_entry:
            for exit_idx in range(entry_idx + 1, len(hex_sequence)):
                if hex_sequence[exit_idx] not in entry_region_hexes:
                    break
            else:
                exit_idx = len(hex_sequence)  # Never left

            truncated = group_sorted.iloc[:exit_idx]
            kept_segments.append(truncated)
        else:
            truncated = group_sorted.iloc[:entry_idx + 1]
            kept_segments.append(truncated)

    if not kept_segments:
        print("Warning: No trajectories enter the specified entry region.")
        return pd.DataFrame(columns=ARcat.columns)

    ARcat_truncated = pd.concat(kept_segments, ignore_index=True)
    print(f"Kept {ARcat_truncated[id_col].nunique()} unique ARs feeding into the entry region.")
    return ARcat_truncated






def origins_and_destinations(ARcat, LC_cond=None):
    """
    Extract pairs of origin and destination coordinate indices and their associated IVT values
    from AR tracks, optionally filtered by lifecycle conditions.

    Parameters:
    -----------
    ARcat : pd.DataFrame
        Atmospheric river catalog with columns including 'trackid', 'coord_idx', 'mean_ivt'.
    LC_cond : str or None, optional
        Lifecycle conditioning method:
        - None: use full tracks
        - 'birth-death', 'birth-landfall', 'landfall-death': apply lifecycle filters via helper functions.

    Returns:
    --------
    a_origins_coordID : np.ndarray
        Array of origin coordinate indices for edges.
    a_origins_ivt : np.ndarray
        Array of IVT values at origin points.
    a_dest_coordID : np.ndarray
        Array of destination coordinate indices for edges.
    a_dest_ivt : np.ndarray
        Array of IVT values at destination points.
    """
    # Get unique track IDs
    a_allids = np.unique(ARcat.trackid)

    # Preallocate lists to collect origin/destination coordinate IDs and IVT values
    l_origins_coordID, l_dest_coordID = [], []
    l_origins_ivt, l_dest_ivt = [], []

    # Remove rows with NaN coordinate indices (invalid spatial data)
    ARcat_non_na = ARcat[ARcat['coord_idx'].notna()]

    # Define lifecycle condition functions mapping if LC_cond is provided
    conditional_locations = {}
    if LC_cond:
        conditional_locations = {
            'birth-death': _birth_death,
            'birth-landfall': _birth_lf,
            'landfall-death': _lf_death
        }

    # Loop over each track
    for ID in a_allids:
        # Select AR data for this track
        AR = ARcat_non_na[ARcat_non_na['trackid'] == ID]

        # Only proceed if track length > 1 (needed for origin-destination pairs)
        if AR.shape[0] > 1:
            tmp_ivt = AR['mean_ivt'].values
            tmp_coordID = AR['coord_idx'].values

            if LC_cond:
                # Retrieve lifecycle function based on LC_cond string
                lc_func = conditional_locations.get(LC_cond)
                if lc_func is None:
                    raise ValueError(f"Unknown conditioning of lifecycle instances for edge construction: {LC_cond}")

                # Get indices corresponding to lifecycle condition (e.g., birth-death segment)
                t_indices = lc_func(AR)

                if t_indices is not None:
                    # Subset IVT and coordinate IDs according to lifecycle indices
                    tmp_ivt = tmp_ivt[t_indices]
                    tmp_coordID = tmp_coordID[t_indices]

                    # Append origin and destination coordinate IDs and IVT values (consecutive pairs)
                    l_origins_coordID.append(tmp_coordID[:-1])
                    l_origins_ivt.append(tmp_ivt[:-1])
                    l_dest_coordID.append(tmp_coordID[1:])
                    l_dest_ivt.append(tmp_ivt[1:])
            else:
                # No lifecycle conditioning: take all consecutive pairs from full track
                l_origins_coordID.append(tmp_coordID[:-1])
                l_origins_ivt.append(tmp_ivt[:-1])
                l_dest_coordID.append(tmp_coordID[1:])
                l_dest_ivt.append(tmp_ivt[1:])

    # Concatenate lists of arrays into single numpy arrays for output
    a_origins_coordID = np.concatenate(l_origins_coordID)
    a_dest_coordID = np.concatenate(l_dest_coordID)
    a_origins_ivt = np.concatenate(l_origins_ivt)
    a_dest_ivt = np.concatenate(l_dest_ivt)

    return a_origins_coordID, a_origins_ivt, a_dest_coordID, a_dest_ivt



def transport_matrix(orig_coordID, dest_coordID):
    """
    Build a sparse transport (adjacency) matrix from AR origin and destination coordinate IDs.
    Note that this works with the IDs which are h3 hashs for hexgrids. 

    Parameters:
    -----------
    orig_coordID : np.ndarray
        Array of origin coordinate hash IDs.
    dest_coordID : np.ndarray
        Array of destination coordinate hash IDs.

    Returns:
    --------
    orig_indices : np.ndarray
        Integer indices corresponding to origin coordinates in the matrix.
    dest_indices : np.ndarray
        Integer indices corresponding to destination coordinates in the matrix.
    a_uni_coordID : np.ndarray
        Unique coordinate hash IDs representing rows and columns of the matrix.
    A_sparse : scipy.sparse.coo_matrix
        Sparse adjacency matrix of size n_unique_coords x n_unique_coords,
        where entries represent transport from origin to destination.
    """
    # Get unique coordinate IDs across origins and destinations
    a_uni_coordID = np.unique(np.hstack([orig_coordID, dest_coordID]))
    n = a_uni_coordID.size

    # Create a mapping from coordinate hash to a zero-based index in the matrix
    hash_mapping = {hash_id: idx for idx, hash_id in enumerate(a_uni_coordID)}

    # Vectorize coordinate hash lookups to indices for origins and destinations
    orig_indices = np.array([hash_mapping[hash_id] for hash_id in orig_coordID])
    dest_indices = np.array([hash_mapping[hash_id] for hash_id in dest_coordID])

    # Create sparse adjacency matrix with ones at edges between origins and destinations
    a_ones = np.ones(len(orig_indices))
    A_sparse = coo_matrix((a_ones, (orig_indices, dest_indices)), shape=(n, n))

    return orig_indices, dest_indices, a_uni_coordID, A_sparse



def matrix_to_graph(A, weighted, directed, eps, self_links, weighing='absolute'):
    """
    Convert a sparse adjacency matrix to a NetworkX graph with optional weighting and direction.

    Parameters:
    - A : scipy sparse matrix
        Adjacency matrix representing the AR transport network (optionally weighted).
    - weighted : bool
        Whether to treat edges as weighted or unweighted.
    - directed : bool
        Whether the resulting graph should be directed or undirected.
    - eps : float
        Threshold to consider an edge (edges with weight < eps are discarded).
    - self_links : bool
        Whether self-links (edges from a node to itself) are allowed.
    - weighing : str, optional, default 'absolute'
        Method of weighting edges; can be 'absolute' or 'relative'.
        'relative' normalizes edge weights by row sums.

    Returns:
    - G : networkx.Graph or networkx.DiGraph
        The constructed graph.
    """

    if self_links == False:
        # Remove self-links by setting diagonal elements to zero
        A = _set_diagonal_zero(A)

    ## CASE 1: weighted and undirected
    if weighted and (not directed):
        # Symmetrize the matrix by averaging with its transpose
        Acsr0 = A.tocsr()
        Acsr = (Acsr0 + Acsr0.T) / 2

        G = nx.Graph()
        added_edges = set()  # track added edges to avoid duplicates

        rows, cols = Acsr.nonzero()
        
        # If relative weighting is requested, precompute row sums for normalization
        if weighing == 'relative':
            row_sums = np.array(Acsr.sum(axis=1)).flatten()

        # Iterate over non-zero entries in the symmetric matrix
        for row, col in zip(rows, cols):
            # Avoid self-loops and duplicated edges (undirected)
            if row != col and (row, col) not in added_edges and (col, row) not in added_edges and Acsr[row,col] >= eps:
                # Average the weights of symmetric entries
                weight = (Acsr[row, col] + Acsr[col, row]) / 2
                # Normalize weight by row sum if requested
                if weighing == 'relative':
                    weight /= row_sums[row]
                G.add_edge(row, col, weight=weight)
                # Mark edges as added in both directions
                added_edges.add((row, col))
                added_edges.add((col, row))

    ## CASE 2: unweighted and directed
    elif (not weighted) and directed:
        Acsr = A.tocsr()
        G = nx.DiGraph()
        rows, cols = Acsr.nonzero()

        # Add edges for entries above threshold, unweighted
        for row, col in zip(rows, cols):
            if Acsr[row, col] >= eps:
                G.add_edge(row, col)
        # Set default edge weight attribute to 1.0
        nx.set_edge_attributes(G, values=1.0, name='weight')

    ## CASE 3: weighted and directed
    elif weighted and directed:
        Acsr = A.tocsr()
        G = nx.DiGraph()
        rows, cols = Acsr.nonzero()

        # Add weighted edges for entries above threshold
        for row, col in zip(rows, cols):
            weight = Acsr[row, col]
            if weight >= eps:
                # Normalize weight by row sum if relative weighing requested
                if weighing == 'relative':
                    weight = weight / Acsr[row, :].sum()
                G.add_edge(row, col, weight=weight)

    ## CASE 4: unweighted and undirected
    else:
        # Binarize the adjacency matrix with threshold eps-1 to ensure thresholding
        A = binarize(A, threshold=eps-1)
        # Create undirected graph from binary adjacency matrix
        G = nx.from_scipy_sparse_array(A, create_using=nx.Graph)
        # Set all edge weights to 1.0 (unweighted)
        nx.set_edge_attributes(G, values=1.0, name='weight')

    return G



def generate_network(A, grid, weighted, directed, eps, self_links, weighing='absolute'):
    """
    Generate one or multiple NetworkX graphs from adjacency matrices and spatial grid data.

    Parameters:
    - A : list of scipy sparse matrices
        List of adjacency matrices (one for each class, if conditioned).
    - grid : list of tuples
        Each tuple contains (latitudes, longitudes, coordinate IDs) corresponding to nodes.
    - weighted : bool
        Whether to treat edges as weighted.
    - directed : bool
        Whether the graph should be directed.
    - eps : float or list of floats
        Threshold(s) for edge inclusion; can be a single value or one per adjacency matrix.
    - self_links : bool
        Whether self-links should be allowed in the graph.
    - weighing : str, optional, default 'absolute'
        Method for edge weight normalization: 'absolute' or 'relative'.

    Returns:
    - G or list of G:
        If only one adjacency matrix is provided, returns a single graph;
        otherwise returns a list of graphs.
    """

    L = len(A)
    l_G = []  # List to store generated graphs

    # If eps is a single number, convert it to a list for all matrices
    if isinstance(eps, (int, float)):
        eps = [eps] * L

    for l in range(L):
        print(f"...generating graph ({l+1}/{L})...")
        a_replats, a_replons, a_coordID = grid[l]

        # Create the graph from adjacency matrix using the previous function
        G = matrix_to_graph(A[l], weighted, directed, eps[l], self_links, weighing)

        # Map numeric node indices to coordinate IDs (e.g., H3 hex IDs)
        mapping = {i: a_coordID[i] for i in G.nodes}
        G = nx.relabel_nodes(G, mapping)

        # Add latitude, longitude, and coordinate ID as node attributes
        for i in G.nodes:
            # Find the index of the coordinate ID in a_coordID
            orig_idx = a_coordID.index(i) if isinstance(a_coordID, list) else np.where(a_coordID == i)[0][0]

            # Round lat/lon to 2 decimals for clarity and assign to node
            G.nodes[i]['Latitude'] = round(a_replats[orig_idx], 2)
            G.nodes[i]['Longitude'] = round(a_replons[orig_idx], 2)

            # Add coordID attribute for clarity (redundant with node label)
            G.nodes[i]['coordID'] = i

        l_G.append(G)

    # Return single graph if only one input; otherwise return list of graphs
    return l_G[0] if L == 1 else l_G




def preprocess_catalog(ARcat, T, loc, grid_type, X, res, cond, LC_cond=None, fully_contained=True, bins=5):
    """
    Preprocess an atmospheric river (AR) catalog by regridding, filtering in time/space,
    conditioning on features, and extracting relevant columns.

    Parameters:
    - ARcat (pd.DataFrame): Input AR catalog.
    - T (tuple): Time constraint, e.g. (start, end).
    - loc (str): AR locator, e.g. 'head', 'centroid'.
    - grid_type (str): Either 'rectangular' or 'hexagonal'.
    - X (dict): Spatial constraint bounds: {'lon': (min, max), 'lat': (min, max)}.
    - res (float or int): Grid resolution (for regridding or selecting hex resolution).
    - cond (str): Conditioning dictionary, e.g. {'lifetime': 'longest'}.
    - LC_cond (str, optional): Condition on AR life cycle, e.g. 'landfall-death'.
    - fully_contained (bool, optional): If True, only keep ARs fully within spatial bounds.
    - bins (int): Number of bins to discretize the conditioning variable.

    Returns:
    - l_ars_subset (list of pd.DataFrames): List of processed AR tracks.
    - coord_dict (dict or None): Mapping of hashed coordinates to lat-lon (only for rectangular grid).
    """

    # Get coordinate column names for the specified location type
    xcoord, ycoord = ARlocator(loc, grid_type)

    if grid_type == 'rectangular':
        # Regrid ARcat coordinates to fixed resolution
        ARcat[xcoord], ARcat[ycoord] = regrid(ARcat, res, loc)

        # Hash coordinates to unique IDs (for compatibility with hex grid indexing)
        print('...hashing...')
        ARcat['coord_idx'] = ARcat.apply(lambda row: _lat_lon_to_hash(row[ycoord], row[xcoord]), axis=1)

        # Create dictionary mapping coord_idx → (lat, lon) for reverse lookup or labeling
        coord_dict = dict(zip(ARcat['coord_idx'], zip(ARcat[ycoord], ARcat[xcoord])))

    elif grid_type == 'hexagonal':
        # Rename relevant columns to standard names based on resolution and location
        ARcat = ARcat.rename(columns={"hex_idx_" + loc + "_res" + str(res): "coord_idx"})
        ARcat = ARcat.rename(columns={"hex_" + loc + "_x_res" + str(res): "hex_" + loc + "_x"})
        ARcat = ARcat.rename(columns={"hex_" + loc + "_y_res" + str(res): "hex_" + loc + "_y"})

        # No coordinate dictionary needed for hex grid, as coordinates are implicit in hex index
        coord_dict = None
        print("...hexagonal coordinates provided: no regridding and no coordinate dictionary...")

    else:
        # Invalid grid type
        print("Specification error: grid_type must be either rectangular or hexagonal!")
        return

    # Apply temporal and spatial constraints (e.g., cropping ARs to region/time window)
    d_ars_constr = constrain(ARcat, xcoord, ycoord, T, X, fully_contained)

    # Check for empty result after constraints
    if d_ars_constr.empty:
        print('Error: constraints eliminated all AR trajectories!')
        return None, None
    else:
        # Apply feature-based filtering (e.g., top N lifetime, widest ARs)
        l_ars_subset = condition(d_ars_constr, cond, bins)
        L = len(l_ars_subset)

        # Drop unneeded columns based on life-cycle condition
        if LC_cond == 'landfall-death' or LC_cond == 'birth-landfall':
            l_ars_subset = [
                l_ars_subset[l][['time', 'trackid', 'coord_idx', 'mean_ivt', 'lf_lon']]
                for l in range(L)
            ]
        else:
            l_ars_subset = [
                l_ars_subset[l][['time', 'trackid', 'coord_idx', 'mean_ivt']]
                for l in range(L)
            ]

        print("Computations successful, output contains " + str(L) + " AR catalogs.")
        return l_ars_subset, coord_dict



def generate_transport_matrix(ARcats, grid_type, coord_dict, LC_cond=None):
    """
    Generate transport matrices from multiple subsets of atmospheric river catalogs.

    Parameters:
    -----------
    ARcats : list of pd.DataFrame
        List of AR catalogs, each a subset possibly conditioned by some criteria. Only a list of one if no conditioning has been applied!
    grid_type : str
        Type of grid used ('rectangular', 'hexagonal', etc.), used in coordinate hashing.
    coord_dict : dict
        Dictionary mapping coordinate hashes to lat-lon pairs, used to invert hashes.
    LC_cond : str or None, optional
        Lifecycle condition for origins_and_destinations (e.g. 'birth-death').

    Returns:
    --------
    l_A : list of sparse matrices
        Transport matrices (adjacency) for each subset.
    l_idx : list of tuples
        Lists of origin and destination indices in the transport matrices.
    l_hashidx : list of tuples
        Origin and destination coordinate IDs used in the transport matrices.
    l_ivt : list of tuples
        IVT values corresponding to origins and destinations.
    l_grid : list of tuples
        Latitude and longitude arrays plus unique coordinate IDs for spatial embedding.
    """
    L = len(ARcats)

    # Initialize lists to store outputs per subset
    l_A, l_idx, l_hashidx, l_ivt, l_grid = [], [], [], [], []

    for l in range(L):
        d_cat = ARcats[l]

        # Extract origins, destinations, and associated IVT values
        print(f'...extracting origins and destinations from AR trajectories ({l+1}/{L})...')
        a_origins_coordID, a_origins_ivt, a_dest_coordID, a_dest_ivt = origins_and_destinations(d_cat, LC_cond)

        # Compute transport matrix and index mappings
        print(f'...computing transport matrix ({l+1}/{L})...')
        a_orig_indices, a_dest_indices, a_uni_coordID, A = transport_matrix(a_origins_coordID, a_dest_coordID)

        # Reconstruct lat-lon coordinates from unique coordinate IDs for spatial embedding
        a_replats, a_replons = np.vstack(
            [_invert_hash(a_uni_coordID[i], grid_type, coord_dict) for i in range(a_uni_coordID.size)]
        ).T

        # Collect results
        l_A.append(A)
        l_idx.append((a_orig_indices, a_dest_indices))
        l_hashidx.append((a_origins_coordID, a_dest_coordID))
        l_ivt.append((a_origins_ivt, a_dest_ivt))
        l_grid.append((a_replats, a_replons, a_uni_coordID))

    return l_A, l_idx, l_hashidx, l_ivt, l_grid




# %% EDGE AND NODE ATTRIBUTES



def add_edge_attr_to_graph(G, edge_attr, attr_name):
    """
    Add a custom attribute to edges in a graph based on a given attribute dictionary.

    Parameters:
    - G : networkx.DiGraph or networkx.Graph
        The input graph whose edges will be annotated.
    - edge_attr : dict
        Dictionary mapping (origin, destination) edge tuples to values (e.g., class, score).
        Edges not in this dictionary will receive a NaN value.
    - attr_name : str
        Name of the edge attribute to add to the graph.

    Returns:
    - G : networkx.DiGraph or networkx.Graph
        The graph with the specified edge attribute added to each edge.
    """
    # Iterate through the graph edges and add skewness to the corresponding edges
    for (orig_index, dest_index) in G.edges:
        if (orig_index, dest_index) in edge_attr:
            G[orig_index][dest_index][attr_name] = edge_attr[(orig_index, dest_index)]
        else:
            G[orig_index][dest_index][attr_name] = np.nan
    return G


def add_node_attr_to_graph(G, node_attr, attr_name):
    """
    Add a custom attribute to nodes in a graph based on a given attribute dictionary.

    Parameters:
    - G : networkx.DiGraph or networkx.Graph
        The input graph whose nodes will be annotated.
    - node_attr : dict
        Dictionary mapping node IDs to values (e.g., class, score).
        Nodes not in this dictionary will receive a NaN value.
    - attr_name : str
        Name of the node attribute to add to the graph.

    Returns:
    - G : networkx.DiGraph or networkx.Graph
        The graph with the specified node attribute added to each node.
    """
    for node in G.nodes:
        G.nodes[node][attr_name] = node_attr.get(node, np.nan)
    return G



def assign_continent(G, continental_mask, res):
    """
    Assign continent labels to nodes in an AR network based on their latitude and longitude.
    The labels are determined using a (possibly downsampled) continental mask at the specified resolution.

    :param G: A NetworkX graph with node attributes 'Latitude' and 'Longitude'.
    :param continental_mask: 2D numpy array representing continent classifications 
                             at 0.5-degree resolution.
    :param res: Target resolution in degrees for downsampling (e.g., 2.0).
    :return: A copy of the input graph with a new node attribute 'continent' assigned.
    """
    Gcopy = G.copy()
    # downsampling
    a_contimask_down = _downsample_matrix(continental_mask, res)
    a_lons_lowres, a_lats_lowres = np.arange(-180, 180, res), np.arange(-90, 90, res)
    
    # Extract lats and lons
    a_nodeidx = np.array(Gcopy.nodes())
    a_lons = np.hstack(list(nx.get_node_attributes(Gcopy, "Longitude").values()))
    a_lats = np.hstack(list(nx.get_node_attributes(Gcopy, "Latitude").values()))
    Nnodes = a_nodeidx.size

    # Assign continent to each node
    for i in range(Nnodes):
        nodeidx = a_nodeidx[i]
        lat, lon = a_lats[i], a_lons[i]
        lat_idx = np.abs(a_lats_lowres - lat).argmin(axis=0)
        lon_idx = np.abs(a_lons_lowres - lon).argmin(axis=0)
        Gcopy.nodes[nodeidx].update({'continent': a_contimask_down[lat_idx, lon_idx]})
    
    return Gcopy




def reindex_edges(G0):
    """
    Relabel the nodes in the input graph (directed) using their coordID attributes.
    
    Parameters:
        G0 (nx.DiGraph): The original directed graph with 'coordID' attributes for nodes.
    
    Returns:
        nx.DiGraph: A new directed graph with relabeled nodes and attributes, excluding self-links.
    """
    # Relabel nodes using 'coordID' attribute
    node_coordIDs = nx.get_node_attributes(G0, "coordID")
    
    # Create a new directed graph
    G = nx.DiGraph()
    
    # Add edges with updated labels and exclude self-links
    for u, v, data in G0.edges(data=True):
        new_u, new_v = node_coordIDs[u], node_coordIDs[v]
        G.add_edge(new_u, new_v, **data)
    
    # Add node attributes using updated labels
    for node, data in G0.nodes(data=True):
        if node in node_coordIDs:  # Ensure the node has a coordID
            new_node = node_coordIDs[node]
            G.add_node(new_node, **data)
    
    return G





# %% CONSENSUS NETWORKS

def average_networks_by_attributes(G1, G2, attr_name):
    """
    Create an averaged graph from two input graphs (G1 and G2) by averaging node and edge attributes.

    The function matches nodes between G1 and G2 by their geographical coordinates (e.g., Latitude and Longitude),
    using 'coordID' or the node ID as a unique identifier (such as an H3 hex ID). A new graph is returned where:
      - Nodes are retained and their values for the specified attribute (`attr_name`) are averaged across both graphs.
      - Edges are averaged by 'weight' and 'edge_betweenness' if present.
      - The attribute `attr_name` is also averaged for edges, if found on both graphs.

    Parameters
    ----------
    G1 : networkx.DiGraph or networkx.Graph
        The first input graph. Nodes must contain 'Latitude' and 'Longitude' attributes.
    G2 : networkx.DiGraph or networkx.Graph
        The second input graph, assumed to follow the same structure as G1.
    attr_name : str
        The name of the node and/or edge attribute to be averaged between G1 and G2.

    Returns
    -------
    Gavg : networkx.DiGraph
        A new graph where:
          - Nodes are indexed by their coordinate-based ID (e.g., H3) and have averaged values for `attr_name`.
          - Edges are defined by matching coordinate pairs, with averaged 'weight', 'edge_betweenness',
            and `attr_name` (if available in both graphs).

    Notes
    -----
    - Only attributes found in both G1 and G2 are averaged; if `attr_name` is missing for a node or edge in either graph,
      it will be excluded from the averaged result for that element.
    - Node matching is based on coordinate IDs; ensure the spatial grid is consistent across both graphs.
    - Assumes the input graphs share the same topology and reference space (e.g., both use H3 indexing or equivalent).
    """

    # Check if all nodes in both graphs have attr_name
    has_node_sign = (
        all(attr_name in data for _, data in G1.nodes(data=True)) and
        all(attr_name in data for _, data in G2.nodes(data=True))
    )

    # Check if all edges in both graphs have attr_name
    has_edge_sign = (
        all(attr_name in data for _, _, data in G1.edges(data=True)) and
        all(attr_name in data for _, _, data in G2.edges(data=True))
    )

    # Build mapping from coordinates to H3 node IDs
    coords_to_hex = {}
    for G in (G1, G2):
        for node, data in G.nodes(data=True):
            coord = (data['Latitude'], data['Longitude'])
            coordID = data.get('coordID', node)
            coords_to_hex[coord] = coordID

    # Collect all unique edges based on coordinate pairs
    edge_data = {}
    for G in (G1, G2):
        for u, v, data in G.edges(data=True):
            cu = (G.nodes[u]['Latitude'], G.nodes[u]['Longitude'])
            cv = (G.nodes[v]['Latitude'], G.nodes[v]['Longitude'])
            edge_entry = {
                'weight': data.get('weight', 0),
                'edge_betweenness': data.get('edge_betweenness', 0)
            }
            if has_edge_sign:
                edge_entry[attr_name] = data[attr_name]
            edge_data.setdefault((cu, cv), []).append(edge_entry)

    # Initialize consensus graph with H3 node labels
    Gavg = nx.DiGraph()
    for coord, coordID in coords_to_hex.items():
        lat, lon = coord
        node_data = {
            'Latitude': lat,
            'Longitude': lon,
            'coordID': coordID
        }
        if has_node_sign:
            sign_vals = []
            for G in (G1, G2):
                for node, data in G.nodes(data=True):
                    if (data['Latitude'], data['Longitude']) == coord:
                        sign_vals.append(data[attr_name])
            if sign_vals:
                node_data[attr_name] = sum(sign_vals) / len(sign_vals)

        Gavg.add_node(coordID, **node_data)

    # Add averaged edges using H3 IDs
    for (cu, cv), vals in tqdm(edge_data.items()):
        u_hex = coords_to_hex[cu]
        v_hex = coords_to_hex[cv]

        avg_weight = sum(d['weight'] for d in vals) / len(vals)
        avg_betweenness = sum(d['edge_betweenness'] for d in vals) / len(vals)

        edge_attrs = {
            'weight': avg_weight,
            'edge_betweenness': avg_betweenness
        }
        if has_edge_sign:
            edge_attrs[attr_name] = sum(d[attr_name] for d in vals) / len(vals)

        Gavg.add_edge(u_hex, v_hex, **edge_attrs)

    return Gavg



def consensus_network(networks, thresh, eps, weight_variable=None):
    """
    Construct a consensus network from a list of input graphs by averaging edge weights
    and applying thresholding rules to retain only consistent connections.

    This function identifies spatially corresponding nodes across multiple input graphs 
    (based on their Latitude and Longitude), tracks the weights of edges between them, 
    and retains only those edges that are present and above a specified threshold in 
    **all** networks (if non-zero) or have zero weight. The result is a directed graph 
    representing the consensus of input networks.

    Parameters
    ----------
    networks : list of networkx.DiGraph
        A list of input graphs assumed to have nodes with 'Latitude', 'Longitude', and 'coordID' attributes.
    thresh : float
        A minimum value that all corresponding edge weights across networks must meet 
        (if non-zero) to be considered part of the consensus.
    eps : float
        A final cutoff threshold. After averaging, an edge is only added to the consensus graph
        if its consensus weight is strictly greater than `eps`.
    weight_variable : str or None, optional
        The edge attribute to be used as the weight. If None, defaults to 'weight'. 
        All edge weights are assumed to be numeric.

    Returns
    -------
    Gcons : networkx.DiGraph
        A consensus graph where:
          - Nodes represent unique spatial positions across the input graphs.
          - Edges represent consistent directional connections between nodes that 
            meet the threshold condition across all graphs.
          - Each retained edge has an averaged weight under the key `weight_variable`.

    Notes
    -----
    - Nodes are matched across graphs based on identical (Latitude, Longitude) tuples.
    - An edge is retained in the consensus graph if **all** of its weights across the input
      graphs are either zero or greater than or equal to `thresh`.
    - The final weight of each retained edge is the average across all input graphs.
    - Only edges with average weight strictly greater than `eps` are included in the output.
    - Designed for use cases where spatial coherence and multi-model agreement are essential,
      such as in climate network analysis or transport network ensembles.

    See Also
    --------
    networkx.DiGraph : Underlying graph structure used.
    """
    # Precompute node coordinates, mappings, and coordIDs for efficiency
    coords_to_node = {}
    node_counter = 0
    edge_weights = [{} for _ in networks]  # One dictionary for each graph

    # Collect all unique edges across networks and map coordinates to nodes in Gcons
    unique_edges = set()
    for i, G in enumerate(networks):
        for node, data in G.nodes(data=True):
            coord = (data['Latitude'], data['Longitude'])
            if coord not in coords_to_node:
                coords_to_node[coord] = (node_counter, data['coordID'])  # Store node ID and coordID
                node_counter += 1
        for u, v, data in G.edges(data=True):
            coord_u = (G.nodes[u]['Latitude'], G.nodes[u]['Longitude'])
            coord_v = (G.nodes[v]['Latitude'], G.nodes[v]['Longitude'])
            unique_edges.add((coord_u, coord_v))
            if weight_variable is None:
                edge_weights[i][(coord_u, coord_v)] = data.get('weight', 0)
                weight_variable = 'weight'
            else:
                edge_weights[i][(coord_u, coord_v)] = data.get(weight_variable, 0)

    # Create the consensus network graph
    Gcons = nx.DiGraph()

    # Add nodes to Gcons with coordID
    for coord, (node_id, coordID) in coords_to_node.items():
        Gcons.add_node(node_id, Latitude=coord[0], Longitude=coord[1], coordID=coordID)

    # Compute consensus edges and add them to Gcons
    for coord_u, coord_v in tqdm(unique_edges):
        weights = [edge_weights[i].get((coord_u, coord_v), 0) for i in range(len(networks))]
        avg_weight = sum(weights) / len(networks)

        # Consensus formation: 
        if all(w == 0 or w >= thresh for w in weights):
            weight_cons = avg_weight
        else:
            weight_cons = 0

        # Add edge with consensus weight if above threshold
        if weight_cons > eps:
            Gcons.add_edge(coords_to_node[coord_u][0], coords_to_node[coord_v][0], **{weight_variable: weight_cons})

    return Gcons




def normalize_outgoing_weights(G):
    """
    Normalize outgoing edge weights for each node so that their sum equals 1.

    This function computes the total weight of all outgoing edges for each node 
    in a directed graph and assigns a new attribute `'probability'` to each edge, 
    representing the normalized weight (i.e., a fraction of the total outgoing weight). 
    This is useful for a probabilistic interpretation of the AR network.

    Parameters
    ----------
    G : networkx.DiGraph
        A directed graph where each edge is expected to have a `'weight'` attribute.

    Returns
    -------
    None
        The function modifies the input graph `G` in place by adding a `'probability'` 
        attribute to each edge.

    Notes
    -----
    - If a node has zero total outgoing weight, its outgoing edge probabilities are 
      set to zero to avoid division by zero.
    - The function assumes all relevant edges already exist in the graph.
    """
    for node in G.nodes():
        total_weight = sum(G[node][neighbor]['weight'] for neighbor in G.successors(node))
        
        # Normalize each edge weight as a probability
        if total_weight > 0:
            for neighbor in G.successors(node):
                G[node][neighbor]['probability'] = G[node][neighbor]['weight'] / total_weight
        else:
            for neighbor in G.successors(node):
                G[node][neighbor]['probability'] = 0



def complete_nodes(G, res):
    """
   Add missing hexagonal grid nodes to a graph to ensure full spatial coverage at a given H3 resolution.

   This function identifies all H3 hexagon cells at a specified resolution that 
   cover the entire globe and ensures that every such cell is represented as a node 
   in the graph `G`. If a hexagon is not already present (based on the `'coordID'` 
   node attribute), it is added with default geographic coordinates and NaN values 
   for any additional node attributes already used in `G`.

   Parameters
   ----------
   G : networkx.Graph or networkx.DiGraph
       The graph to complete. Existing nodes must have a `'coordID'` attribute.
   res : int
       The H3 resolution to use when completing the hexagonal grid. Higher values 
       yield finer grids.

   Returns
   -------
   G : networkx.Graph or networkx.DiGraph
       The modified graph containing all expected nodes at resolution `res`, 
       including previously missing hexagons.

   Notes
   -----
   - Uses `h3-py` to generate hexagonal grids.
   - Any extra node attributes already present in `G` will be initialized with `np.nan`
     for newly added nodes.
   - Assumes nodes are uniquely identified by their `'coordID'` values.
    """
    # Get all hexagons covering the planet at resolution 0
    res0_hexes = h3.get_res0_indexes()
    
    # Subdivide each resolution 0 hexagon into finer resolution hexagons
    all_hex_ids = []
    for hex_id in res0_hexes:
        all_hex_ids.extend(h3.uncompact([hex_id], res))

    # Find existing node coordIDs in the graph
    existing_ids = set(nx.get_node_attributes(G, "coordID").values())

    # Identify all node-level attributes beyond the standard ones
    additional_attrs = set()
    for _, data in G.nodes(data=True):
        additional_attrs.update(data.keys())
    additional_attrs -= {'Latitude', 'Longitude', 'coordID'}

    # Add missing nodes with NaN for any additional attributes
    for hex_id in all_hex_ids:
        if hex_id not in existing_ids:
            lat, lon = h3.h3_to_geo(hex_id)
            attr = {'Latitude': lat, 'Longitude': lon, 'coordID': hex_id}
            for extra in additional_attrs:
                attr[extra] = np.nan
            G.add_node(hex_id, **attr)

    return G


