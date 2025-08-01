# Copyright (C) 2025 by
# Tobias Braun

#------------------ PATH ---------------------------#
import sys
PATH = "/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/scripts"
sys.path.insert(0, PATH)


# %% IMPORT MODULES

# standard packages
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd


# specific packages
from cmcrameri import cm
import cartopy.crs as ccrs
from h3 import h3
from tqdm import tqdm
import networkx as nx
from shapely.geometry import shape, Polygon, box


# %% INTERNAL HELPER FUNCTIONS
# The functions below are not meant to be called from a script but are secondary
# functions that are called by the main functions below.



def _get_edge_directions(G, ncolors=40):
    """
    Extracts normalized edge direction values from an ARTN for use in directional color encoding.

    Parameters:
    - G (networkx.DiGraph): The directed graph.
    - ncolors (int): Number of bins/colors to be used in the direction colormap.

    Returns:
    - a_colours (np.ndarray): Normalized edge directions in [0, 1] for colormap indexing.
    - COLMAP (matplotlib.colors.ListedColormap): Colormap for edge directions.
    - a_colbins (np.ndarray): Bin edges for the direction colormap.
    """

    COLMAP = cm.romaO_r  # Colormap for direction
    a_cmap = COLMAP(np.linspace(0, 1, ncolors))
    COLMAP = ListedColormap(a_cmap)
    
    d_edgedir = edgedir_to_degrees(G)  # Ensure this function is defined
    a_directions = np.array([d_edgedir[edge] for edge in G.edges()]) 
    a_colours = a_directions / 360  # Normalize to [0,1]
    a_colbins = np.linspace(np.min(a_colours), np.max(a_colours), ncolors)
    
    return a_colours, COLMAP, a_colbins



def _calculate_initial_bearing(lon1, lat1, lon2, lat2):
    """
    Calculates the initial bearing (forward azimuth) between two geographic coordinates.
    Needed to transform edge directions to degrees.

    Parameters:
    - lon1, lat1 (float): Longitude and latitude of the starting point (degrees).
    - lon2, lat2 (float): Longitude and latitude of the destination point (degrees).

    Returns:
    - bearing (float): Initial bearing from point 1 to point 2, in degrees [0, 360).
    """

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    dlon = lon2 - lon1
    
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    initial_bearing = np.arctan2(y, x)
    
    initial_bearing = np.degrees(initial_bearing)
    bearing = (initial_bearing + 360) % 360
    
    return bearing



def _transform_coordinates(lon, lat, projection):
    """
    Transforms coordinates from geographic (longitude, latitude) to a specified map projection.
    Needed to draw curved edge with arrow.

    Parameters:
    - lon, lat (float): Geographic coordinates in degrees.
    - projection (cartopy.crs.Projection): Target projection.

    Returns:
    - x, y (float): Projected coordinates in the target projection's units.
    """
    return projection.transform_point(lon, lat, ccrs.PlateCarree())


# %% NETWORK PLOT


def boundary_geom(hash_id, res = 2, rect=False, coord_dict = None):
    """
    Returns the shapely geometry representing the boundary of a cell (hexagon or rectangle) 
    in an ARTN visualization.

    Parameters:
    - hash_id (str or int): Identifier for the hexagon (H3 index) or rectangular cell.
    - res (float): Resolution or width/height of rectangular cells (in degrees). Ignored if `rect=False`.
    - rect (bool): If True, constructs a rectangular cell from center coordinates. If False, uses H3 hexagon.
    - coord_dict (dict): Mapping from hash_id to (lat, lon) center coordinates (required if `rect=True`).

    Returns:
    - poly_shape (shapely.Polygon): Polygon representing the cell boundary.
    """

    if rect:
        center_lat, center_lon = coord_dict[hash_id]
        half_size = res / 2
        min_lon = center_lon - half_size
        max_lon = center_lon + half_size
        min_lat = center_lat - half_size
        max_lat = center_lat + half_size
        poly_shape = box(min_lon, min_lat, max_lon, max_lat)
    else:
        # create a hexagonal shape in shapely to obtain contours of hexagon
        poly_shape = shape({'type': 'Polygon',
                            'coordinates': [h3.h3_to_geo_boundary(h=hash_id, geo_json=True)]})
    return poly_shape


def split_hexagons(gdf):
    """
    Splits hexagons that cross the anti-meridian (dateline) into separate polygons on either side 
    for correct rendering in ARTN plots. Otherwise, they all get squished up bad.
    
    Parameters:
    - gdf (GeoDataFrame): GeoDataFrame with a 'geometry' column containing hexagonal polygons.
    
    Returns:
    - gdf_split (pd.DataFrame): DataFrame with hexagons split at the dateline, ready for plotting.
    """

    # Store the split hexagons
    new_rows = []

    for idx, row in tqdm(gdf.iterrows()):
        tmp_lons, tmp_lats = np.array(row.geometry.exterior.coords).T
        
        # Check if the hexagon crosses the dateline
        if (np.any(tmp_lons > 170)) and (np.any(tmp_lons < -170)):
            # Split into two polygons for each side of the dateline
            tmp_newlons1 = np.where(tmp_lons < 0, 180, tmp_lons)
            poly1 = Polygon(np.vstack([tmp_newlons1, tmp_lats]).T)
            
            tmp_newlons2 = np.where(tmp_lons > 0, -180, tmp_lons)
            poly2 = Polygon(np.vstack([tmp_newlons2, tmp_lats]).T)
            
            # Create two new rows with updated geometries
            row1, row2 = row.copy(), row.copy()
            row1.geometry, row2.geometry = poly1, poly2
            new_rows.extend([row1, row2])
        else:
            # Keep the row as is if it doesn't cross the dateline
            new_rows.append(row)

    # Reassemble the DataFrame with only the new rows
    return pd.DataFrame(new_rows)



def edgedir_to_degrees(G):
    """
    Computes the directional angle (in degrees) of each edge in the ARTN based on node coordinates.

    Parameters:
    - G (networkx.DiGraph): The directed ARTN with nodes having 'Latitude' and 'Longitude' attributes.

    Returns:
    - edge_directions (dict): Mapping from edge (u, v) to its bearing angle in degrees.
    """

    d_position = {i: (G.nodes[i]['Longitude'], G.nodes[i]['Latitude']) for i in G.nodes}
    
    edge_directions = {}
    
    for u, v in G.edges():
        lon1, lat1 = d_position[u]
        lon2, lat2 = d_position[v]
        
        angle_deg = _calculate_initial_bearing(lon1, lat1, lon2, lat2)
        
        edge_directions[(u, v)] = angle_deg
        
    return edge_directions



def split_edges_at_meridian(lon1, lat1, lon2, lat2):
    """
    Splits edges that cross the anti-meridian into two segments for proper plotting in global maps.

    Parameters:
    - lon1, lat1 (float): Coordinates of the edge's start node.
    - lon2, lat2 (float): Coordinates of the edge's end node.

    Returns:
    - segments (list of tuple): One or two coordinate segments representing the original edge.
    """
    
    # Normalize longitudes to be within the range of -180 to 180
    def _normalize_longitude(lon):
        return (lon + 180) % 360 - 180

    lon1 = _normalize_longitude(lon1)
    lon2 = _normalize_longitude(lon2)
    
    lon_diff = abs(lon1 - lon2)
    
    # Check if the edge visually crosses the anti-meridian or wraps around the map
    if lon_diff > 180:
        # Edge crosses the anti-meridian
        if lon1 > lon2:
            # Crosses from +longitudes to -longitudes
            lon1_new = 180
            lon2_new = -180
            if lon1_new == lon1:
                lon1_new -= 0.5
            if lon2_new == lon2:
                lon2_new += 0.5
        else:
            # Crosses from -longitudes to +longitudes
            lon1_new = -180
            lon2_new = 180
            if lon1_new == lon1:
                lon1_new += 0.5
            if lon2_new == lon2:
                lon2_new -= 0.5
        
        segment1 = ((lon1, lat1), (lon1_new, lat1))
        segment2 = ((lon2_new, lat2), (lon2, lat2))
        
        return [segment1, segment2]
    
    else:
        # Edge is within one hemisphere, return as is
        return [((lon1, lat1), (lon2, lat2))]
    
    

    
def draw_curved_edge_with_arrow(ax, lon1, lat1, lon2, lat2, color, width,
                                projection, plot_arrows=True, l0=1.0, curvature=0.2,
                                alpha=0.5, arrow_size=15):
    """
    Draws a curved edge (Bezier arc) between two nodes on a map projection, with optional arrowhead.
    Arrowhead scaling tends to get messy...
    
    Parameters:
    - ax (matplotlib.axes): The map axis to draw on.
    - lon1, lat1 (float): Start coordinates of the edge.
    - lon2, lat2 (float): End coordinates of the edge.
    - color (color): Edge color.
    - width (float): Edge line width.
    - projection (cartopy.crs.Projection): Map projection.
    - plot_arrows (bool): Whether to draw an arrowhead.
    - l0 (float): Base line width scaling factor.
    - curvature (float): Amount of curvature (positive = convex).
    - alpha (float): Transparency of the edge.
    - arrow_size (float): Base size of arrowhead.
    
    Returns:
    - Draws a patch when called from the plot_network function.
    """

    # Transform coordinates
    x1, y1 = _transform_coordinates(lon1, lat1, projection)
    x2, y2 = _transform_coordinates(lon2, lat2, projection)
    
    # Calculate the midpoint and control point for the Bezier curve
    midpoint_x = (x1 + x2) / 2
    midpoint_y = (y1 + y2) / 2
    control_x = midpoint_x
    control_y = midpoint_y + curvature * (x2 - x1)
    
    # Create the curved path using a Bezier curve
    path = patches.Path([
        (x1, y1),
        (control_x, control_y),
        (x2, y2)
    ], [
        patches.Path.MOVETO,
        patches.Path.CURVE3,
        patches.Path.CURVE3
    ])
    
    # Calculate the linewidth based on the input width and l0
    linewidth = l0 * width
    
    # Draw the curved edge
    edge = patches.PathPatch(path, facecolor='none', edgecolor=color, linewidth=linewidth, alpha=alpha, zorder=3)
    ax.add_patch(edge)
    
    if plot_arrows:
        # Calculate the position for the arrowhead
        arrow_position = path.interpolated(11).vertices[-1]
        
        # Calculate the direction for the arrowhead
        dx = arrow_position[0] - path.interpolated(11).vertices[-2][0]
        dy = arrow_position[1] - path.interpolated(11).vertices[-2][1]
        
        # Create and add the arrowhead
        arrow_size_adjusted = max(0.1, arrow_size * linewidth)  # Adjust arrow size based on linewidth
        arrow = patches.FancyArrowPatch(
            (arrow_position[0] - 0.001*dx, arrow_position[1] - 0.001*dy),
            arrow_position,
            color=color,
            alpha=alpha,
            arrowstyle=f'->,head_length={arrow_size_adjusted},head_width={0.6*arrow_size_adjusted}',
            mutation_scale=12
        )
        ax.add_patch(arrow)



def get_edge_weights(G, log=False, linewidth=0):
    """
    Extracts edge weights from an ARTN and normalizes them for proper use in plotting (e.g., edge thickness).

    Parameters:
    - G (networkx.DiGraph): The ARTN with edge weights stored in the 'weight' attribute.
    - log (bool): Whether to log-transform the weights.
    - linewidth (float): Base line width for scaling.

    Returns:
    - a_weights (np.ndarray): Raw edge weights.
    - a_widths (np.ndarray): Scaled edge widths for plotting.
    - weighted (bool): Whether the network contains non-uniform edge weights.
    """

    # Check if the attribute's there at all:
    if 'weight' not in next(iter(G.edges(data=True)), {})[2]:
        return None, False
    
    # get those weights
    a_weights = np.array([G.edges[edge]['weight'] for edge in G.edges()])
    weighted = not np.all(a_weights == 1.0) # is it weighted though?
    
    # log-scale?
    a_plotweights = np.log10(a_weights) if log else np.copy(a_weights)
    # normalize
    a_plotweights /= a_plotweights.max()
    
    return a_weights, linewidth * a_plotweights, weighted


def get_edge_signs(G, attr, linewidth=0):
    """
    Extracts an edge attribute (not really a sign) from the ARTN to use as a color or width encoding.

    Intended for visualizing categorical quantities (e.g., discretized IVT diffs).

    Parameters:
    - G (networkx.DiGraph): The ARTN containing edge attributes.
    - attr (str): The name of the edge attribute to extract ('IVTdiff').
    - linewidth (float): Base line width for scaling.

    Returns:
    - a_colors (np.ndarray): Attribute values (e.g., for coloring).
    - a_widths (np.ndarray): Scaled line widths.
    """
    
    # Check if the attribute's there at all:
    if attr not in next(iter(G.edges(data=True)), {})[2]:
        print('No edge attribute called like that could be found.')
        return None
    
    # get those attributes and make them colours
    a_signs = np.array([G.edges[edge][attr] for edge in G.edges()])
    a_colors = a_signs
    # scale widths
    a_widths = linewidth * np.where(np.abs(a_colors) == 0, 0.5, np.abs(a_colors))
    return a_colors, a_widths


def plot_nodes(ax, G, d_position):
    """
    Plots nodes of the ARTN on the map, with transparency encoding whether a node is actually linked at all.
    
    Parameters:
    - ax (matplotlib.axes): The axis to plot on.
    - G (networkx.DiGraph): The ARTN with nodes having positional attributes.
    - d_position (dict): Dictionary mapping node IDs to projected coordinates.
    
    Returns:
    - Draws nodes when called by plot_network function.
    """
    node_alpha = {node: 0.6 if G.degree(node) > 0 else 0.1 for node in G.nodes}
    nx.draw_networkx_nodes(G, d_position, node_size=.5, node_color='gray', alpha=[node_alpha[n] for n in G.nodes], ax=ax)




def plot_network(G, widths, colours, layout, ndec, log=False, arrowsize=10, linewidth=0, curvature=0.2,
                 fontsize=16, ncolors=40, discard=120, alpha=.5, show_nodes=True, show_arrows=False,
                 show_axes=True, proj=ccrs.PlateCarree(), vmin=None, vmax=None):
    """
    Plot an AR network graph on a map with curved edges and various edge and color encoding options.

    Parameters
    ----------
    G : networkx.DiGraph
        Directed graph representing the network. Nodes must have 'Latitude' and 'Longitude' attributes.

    widths : str or None
        Determines edge width encoding. 
        - 'weights': width reflects edge weights.
        - 'classifier': width reflects moisture classes.
        - None: uniform width based on `linewidth`.

    colours : str
        Determines edge color encoding.
        - 'weights': color reflects edge weights.
        - 'classifier': color reflects some quantitative attribute with discrete color bins.
        - 'directions': color reflects edge direction.

    layout : str
        Visualization theme. 
        - 'default' for light background.
        - 'dark' for dark mode.

    ndec : int
        Normalization factor for decade-scaling of weights (used in tick labels).

    log : bool, default=False
        Whether to log-transform edge weights before plotting (if `widths='weights'`).

    arrowsize : int, default=10
        Size of arrows at the end of directed edges.

    linewidth : float, default=0
        Base linewidth used when `widths` is None.

    curvature : float, default=0.2
        Curvature of the edges for drawing parabolic arcs.

    fontsize : int, default=16
        Font size used for labels and colorbar.

    ncolors : int, default=40
        Number of colors in the color map.

    discard : float, default=120
        Discards edges with longitudinal distance above this threshold (to avoid map-wrapping issues).

    alpha : float, default=0.5
        Transparency of the edges.

    show_nodes : bool, default=True
        Whether to show network nodes on the map.

    show_arrows : bool, default=False
        Whether to draw arrows at the end of each edge.

    show_axes : bool, default=True
        Whether to display map gridlines and labels.

    proj : cartopy.crs, default=ccrs.PlateCarree()
        Projection used for plotting.

    vmin : float, optional
        Minimum value for color normalization.

    vmax : float, optional
        Maximum value for color normalization.

    Returns
    -------
    None
        Displays the plot.
    """
    
    # Set dark or light background
    plt.style.use('dark_background' if layout == 'dark' else 'default')
    linecolor = 'whitesmoke' if layout == 'dark' else 'black'
    
    # Project node positions to the map projection
    d_position = {i: proj.transform_point(G.nodes[i]['Longitude'], G.nodes[i]['Latitude'],
                                          src_crs=ccrs.PlateCarree()) for i in G.nodes}
    
    # Determine edge widths and weights
    if widths == 'weights':
        a_weights, a_widths, weighted = get_edge_weights(G, log, linewidth)
    elif widths == 'classifier':
        a_colours, a_widths = get_edge_signs(G, linewidth)
    elif widths is None:
        a_widths = linewidth * np.ones(len(G.edges))

    # Determine edge coloring
    if colours == 'weights':
        a_colours = a_weights if weighted else np.ones(len(G.edges))
        COLMAP = cm.lapaz_r if layout == 'default' else cm.imola

        # Use provided or derived color range
        vmin = vmin if vmin is not None else int(np.round(np.nanmin(a_weights)))
        vmax = vmax if vmax is not None else int(np.round(np.nanquantile(a_weights, .999)))
        
        # Generate color bins and ticks
        a_cmap = COLMAP(np.linspace(0.2, 1, ncolors))
        a_colbins = np.linspace(vmin, vmax, ncolors)
        tick_positions = np.linspace(vmin, vmax, 5)
        tick_labels = [str(int(round(t / ndec))) for t in tick_positions]
        cbar_label = 'ARs/decade (edge weight)'

    elif colours == 'classifier':
        COLMAP = ListedColormap(['#D73027', '#F46D43', 'yellow', 'dimgray', 'aqua', 'deepskyblue', 'mediumblue'])
        a_cmap = COLMAP(np.linspace(0, 1, 7))
        vmin, vmax = np.nanmin(a_colours), np.nanmax(a_colours)
        a_colbins = np.linspace(vmin, vmax, 7)
        tick_positions = np.linspace(vmin, vmax, 7)
        tick_labels = [str(int(round(t))) for t in tick_positions]
        cbar_label = 'q'

    elif colours == 'directions':
        a_colours, COLMAP, a_colbins = _get_edge_directions(G, ncolors)
        a_cmap = COLMAP(a_colbins)
        vmin, vmax = np.nanmin(a_colours), np.nanmax(a_colours)
        tick_positions = np.array([0.25, 0.5, 0.75, 1])
        tick_labels = np.array(['E', 'S', 'W', 'N'])
        cbar_label = 'edge direction'

    else:
        raise ValueError("Invalid colour attribute.")

    # Initialize figure and axis
    fig, ax = plt.subplots(subplot_kw={'projection': proj})
    ax.set_global()
    ax.coastlines(color=linecolor, linewidth=.5)

    # Plot nodes if requested
    if show_nodes:
        plot_nodes(ax, G, d_position)

    # Normalize values for consistent color mapping
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # Draw all edges with curvature and color
    for k, (node1, node2) in enumerate(tqdm(G.edges)):
        colourval, width = a_colours[k], a_widths[k]

        # Normalize and clamp color values
        if colours == 'weights':
            colourval = min(colourval, vmax)
            colour = COLMAP(norm(colourval))
        else:
            colour = a_cmap[np.digitize(colourval, a_colbins) - 1]
        
        lon1, lat1 = G.nodes[node1]['Longitude'], G.nodes[node1]['Latitude']
        lon2, lat2 = G.nodes[node2]['Longitude'], G.nodes[node2]['Latitude']
        
        # Handle map wrapping by splitting edge segments
        segments = split_edges_at_meridian(lon1, lat1, lon2, lat2)
        for segment in segments:
            (lon1, lat1), (lon2, lat2) = segment
            if abs(lon1 - lon2) > discard:
                continue
            draw_curved_edge_with_arrow(
                ax, lon1, lat1, lon2, lat2, colour, width, ax.projection, 
                show_arrows, l0=1, curvature=curvature, alpha=alpha, arrow_size=arrowsize
            )

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=COLMAP, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.08, aspect=30, shrink=0.8)
    cbar.set_label(cbar_label, color=linecolor, fontsize=fontsize)
    cbar.ax.tick_params(colors=linecolor)
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels(tick_labels, fontsize=fontsize - 2)

    # Add gridlines and labels if requested
    if show_axes:
        gl = ax.gridlines(draw_labels=True, linewidth=0.1, color=linecolor, alpha=0, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'color': linecolor, 'size': fontsize}
        gl.ylabel_style = {'color': linecolor, 'size': fontsize}

    plt.tight_layout()




