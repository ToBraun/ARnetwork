# Copyright (C) 2026 by
# Tobias Braun



# %% IMPORT MODULES

# standard packages
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import warnings
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
from scipy import stats
import itertools

# specific packages
from itertools import combinations
from datetime import time
from dask import delayed, compute
from h3 import h3
from tqdm import tqdm
import networkx as nx
import pytz
from sklearn.preprocessing import binarize
from timezonefinder import TimezoneFinder

# %% RANDOM WALK MODELS

def build_hex_graph(res, dem=None, elevation_scaling=0.001):
    """
    Build a global directed H3-hexagonal graph at the given resolution.

    All resolution-0 hexagons are subdivided to the target resolution, each
    cell is added as a node carrying its centroid latitude/longitude and H3
    index, and a directed edge is created to every immediate (k=1) neighbour.
    Edge weights are uniform by default; the (currently disabled) `dem`
    pathway leaves a hook for elevation-based weighting.

    Args:
    res (int): H3 resolution at which to construct the global hexagonal grid.
    dem (pandas.DataFrame, optional): Digital elevation model with columns
        'hex_idx' and 'elevation', used by the elevation-weighting branch.
        Currently unused (the relevant block is commented out). Defaults to
        None.
    elevation_scaling (float): Decay constant applied to the absolute
        elevation difference between neighbouring cells when DEM weighting
        is active. Defaults to 0.001.

    Returns:
    networkx.DiGraph: Directed graph whose nodes are H3 hexagons (with
        'Latitude', 'Longitude', and 'coordID' attributes) and whose edges
        connect each cell to its six neighbours with attribute 'weight'.
    """
    G = nx.DiGraph()

    # # Add hexagons as nodes
    # for hex_id in hex_ids:
    #     lat, lon = h3.h3_to_geo(str(hex_id))
    #     G.add_node(hex_id, Latitude = lat, Longitude = lon, coordID = hex_id)
    # Get all hexagons covering the planet at resolution 0
    res0_hexes = h3.get_res0_indexes()
    
    # Subdivide each resolution 0 hexagon into finer resolution hexagons
    all_hex_ids = []
    for hex_id in res0_hexes:
        all_hex_ids.extend(h3.uncompact([hex_id], res))
    
    # Find existing node coordIDs in the graph
    existing_ids = set(nx.get_node_attributes(G, "coordID").values())
    
    # Add missing nodes to the graph
    for hex_id in all_hex_ids:
        if hex_id not in existing_ids:
            lat, lon = h3.h3_to_geo(hex_id)
            G.add_node(hex_id, Latitude=lat, Longitude=lon, coordID=hex_id)


    # Add edges between neighboring hexagons
    for hex_id in np.array(all_hex_ids):
        neighbors = list(h3.k_ring_distances(str(hex_id), 1)[1])  # Get neighbors (distance = 1)

        for neighbor in neighbors:
            # Optional: Assign weights based on elevation difference if DEM is provided
            # if dem is not None:
            #     elev_hex = dem.loc[dem.hex_idx == hex_id, 'elevation'].values[0]
            #     elev_neighbor = dem.loc[dem.hex_idx == neighbor, 'elevation'].values[0]
            #     weight = np.exp(-elevation_scaling * np.abs(elev_hex - elev_neighbor))  # Example of weighting
            # else:
            weight = 1.0  # Default weight
            
            if G.has_node(hex_id) and G.has_node(neighbor):  # Ensure both nodes exist before adding an edge
                G.add_edge(hex_id, neighbor, weight=weight)
            

    return G




def haversine_distance_vectorized(lat1, lon1, lat2, lon2):
    """
    Compute the great-circle distance between two sets of points on Earth.

    All four arguments are converted to radians and broadcast against each
    other, so the function works element-wise on scalars, 1-D arrays, or any
    broadcastable combination.

    Args:
    lat1, lon1 (float or numpy.ndarray): Latitude and longitude of the first
        point(s), in degrees.
    lat2, lon2 (float or numpy.ndarray): Latitude and longitude of the second
        point(s), in degrees.

    Returns:
    numpy.ndarray: Great-circle distance(s) in kilometres, using an Earth
        radius of 6371 km.
    """
    R = 6371  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c  # Distance in kilometers

def precompute_distances(G):
    """
    Build a full pairwise great-circle distance matrix over the nodes of `G`.

    Latitude and longitude are read from each node's 'Latitude' and
    'Longitude' attributes, and rows of the matrix are filled by vectorised
    Haversine evaluations against all other nodes.

    Args:
    G (networkx.DiGraph): Graph whose nodes carry 'Latitude' and 'Longitude'
        attributes in degrees.

    Returns:
    numpy.ndarray: Square distance matrix of shape (N, N) in kilometres,
        ordered according to `list(G.nodes())`.
    """
    nodes = list(G.nodes())
    num_nodes = len(nodes)
    
    # Extract coordinates for each node
    latitudes = np.array([G.nodes[node]['Latitude'] for node in nodes])
    longitudes = np.array([G.nodes[node]['Longitude'] for node in nodes])
    
    # Initialize distance matrix
    distance_matrix = np.zeros((num_nodes, num_nodes))
    
    # Compute distances in a vectorized manner
    for i in tqdm(range(num_nodes)):
        distance_matrix[i, :] = haversine_distance_vectorized(
            latitudes[i], longitudes[i], latitudes, longitudes
        )
    
    return distance_matrix


def random_walk_on_graph(G, current_node, length, final_node=None, distance_matrix=None, node_indices=None):
    """
    Generate a single random walk on a directed graph, optionally biased
    toward a target node via precomputed great-circle distances.

    At each step the walker moves to one of the current node's successors.
    When `final_node` is supplied, transition probabilities are weighted
    inversely with the Haversine distance from each candidate (the walker
    is also allowed to stay in place), so the walk drifts toward the target.
    Otherwise neighbours are sampled uniformly. The walk terminates when the
    requested length is reached, when the target is reached, or when the
    walker hits a node with no successors.

    Args:
    G (networkx.DiGraph): Directed graph to walk on.
    current_node: Node ID at which the walk starts.
    length (int): Desired walk length (number of nodes, including the start).
    final_node: Target node ID for guided walks. If None, the walk is
        unbiased. Defaults to None.
    distance_matrix (numpy.ndarray, optional): Pairwise distance matrix as
        returned by `precompute_distances`. Required when `final_node` is
        given.
    node_indices (dict, optional): Mapping from node ID to its row/column
        index in `distance_matrix`. Required when `final_node` is given.

    Returns:
    list[str]: Sequence of node IDs visited along the walk.
    """
    walk = [str(current_node)]
    i = 0
    while i < length - 1 or (final_node is not None and (current_node != final_node or current_node in list(G.successors(final_node)))):
        neighbors = list(G.successors(current_node))
        
        if not neighbors:  # If there are no neighbors, break the walk
            break

        if final_node:  # If a guided node is provided, use it to influence the next step
            # Get the indices for the current node and neighbors
            current_index = node_indices[current_node]
            neighbor_indices = [node_indices[neighbor] for neighbor in neighbors]
            # ARs can also stay where they are:
            neighbor_indices.append(current_index)
            
            # Look up precomputed distances
            distances = np.array([distance_matrix[current_index, idx] for idx in neighbor_indices])
            probabilities = 1 / (distances + 1e-6)  # Add a small constant to avoid division by zero
            probabilities /= probabilities.sum()  # Normalize probabilities
            
            next_node = np.random.choice(neighbors, p=probabilities)  # Choose next node based on calculated probabilities
        else:
            next_node = np.random.choice(neighbors)  # Randomly choose the next node

        walk.append(next_node)
        current_node = next_node
        i += 1

    return walk


def random_walker_ensemble(G_blank, traj_lengths, eps=0, start_nodes=None, term_nodes=None, 
                           Nrealiz=1, LC_cond=None, return_paths=True):
    """
    Run an ensemble of random walks on a directed graph and accumulate their
    traversals as edge weights.

    For each realisation a copy of `G_blank` is taken, `Ntraj = len(traj_lengths)`
    walks are generated (with starts and/or targets fixed by `start_nodes` and
    `term_nodes`, or drawn at random), and each step increments the weight of
    the corresponding edge. Under `LC_cond='birth-death'` only the start–end
    edge is incremented. After all walks, every edge weight is decremented by
    one and edges with weight ≤ `eps` are removed, so the returned graphs
    contain only transitions that recurred more often than the threshold.

    Args:
    G_blank (networkx.DiGraph): Base graph providing the transition topology
        and node coordinates.
    traj_lengths (list[int]): Desired length of each trajectory.
    eps (int): Edge-weight threshold for keeping an edge in the final graph.
        Defaults to 0.
    start_nodes (list, optional): Starting node IDs for each trajectory; if
        None, starts are drawn uniformly at random (unless `term_nodes` is
        given alone, in which case it is used as the starting point).
        Defaults to None.
    term_nodes (list, optional): Target node IDs for each trajectory. When
        both `start_nodes` and `term_nodes` are given, walks are guided from
        start to target. Defaults to None.
    Nrealiz (int): Number of independent realisations. The RNG is reseeded
        with the realisation index for reproducibility. Defaults to 1.
    LC_cond (str, optional): Lifecycle conditioning mode. None increments
        every traversed edge; 'birth-death' increments only the start–end
        edge. Defaults to None.
    return_paths (bool): If True, also return the realised walks. Defaults
        to True.

    Returns:
    list[networkx.DiGraph] or tuple: List of realisation graphs with
        thresholded edge weights. When `return_paths` is True, returns
        (realisations, paths), where `paths` is a list (per realisation) of
        lists of walks.
    """
    Ntraj = len(traj_lengths)
    distance_matrix = precompute_distances(G_blank)
    node_indices = {str(node): idx for idx, node in enumerate(G_blank.nodes())}

    realizations = []      # To store all realization graphs
    all_realizations_paths = [] if return_paths else None

    for nrealiz in tqdm(range(Nrealiz)):
        G = G_blank.copy()
        np.random.seed(nrealiz)
        walks_for_realization = []

        for n in tqdm(range(Ntraj)):
            L = traj_lengths[n]

            # Determine start and final nodes
            if term_nodes is not None and start_nodes is None:
                start_node = str(term_nodes[n])
                walk = random_walk_on_graph(G_blank, start_node, L, distance_matrix=distance_matrix, node_indices=node_indices)
            elif start_nodes is not None and term_nodes is None:
                start_node = str(start_nodes[n])
                walk = random_walk_on_graph(G_blank, start_node, L, distance_matrix=distance_matrix, node_indices=node_indices)
            elif start_nodes is None and term_nodes is None:
                start_node = np.random.choice(list(G_blank.nodes()))
                walk = random_walk_on_graph(G_blank, start_node, L, distance_matrix=distance_matrix, node_indices=node_indices)
            else:  # both start_nodes and term_nodes provided
                start_node = str(start_nodes[n])
                final_node = str(term_nodes[n])
                walk = random_walk_on_graph(G_blank, start_node, L, final_node, distance_matrix=distance_matrix, node_indices=node_indices)

            walks_for_realization.append(walk)  # store the path

            # Update graph weights
            if LC_cond is None:
                for i in range(len(walk) - 1):
                    u, v = walk[i], walk[i + 1]
                    if G.has_edge(u, v):
                        G[u][v]['weight'] += 1
                    else:
                        G.add_edge(u, v, weight=1)
            elif LC_cond == 'birth-death':
                u, v = walk[0], walk[-1]
                if G.has_edge(u, v):
                    G[u][v]['weight'] += 1
                else:
                    G.add_edge(u, v, weight=1)
            else:
                raise ValueError(f"LC_cond {LC_cond} not implemented.")

        # Adjust weights and remove edges below threshold
        for u, v, data in G.edges(data=True):
            data['weight'] -= 1
        edges_to_remove = [(u, v) for u, v, w in G.edges(data="weight") if w <= eps]
        G.remove_edges_from(edges_to_remove)

        # Store results
        realizations.append(G.copy())
        if return_paths:
            all_realizations_paths.append(walks_for_realization)

    if return_paths:
        return realizations, all_realizations_paths
    else:
        return realizations


def random_walker_paths(G_blank, traj_lengths, 
                        start_nodes=None, term_nodes=None, 
                        Nrealiz=1):
    """
    Generate an ensemble of unbiased random walks without accumulating edge
    weights.

    Each realisation is reseeded with its index, and walks are generated by
    sampling uniformly from the successors of the current node until the
    requested length is reached or a dead end is hit. Unlike
    `random_walker_ensemble`, this function only returns the walks
    themselves; no graph is constructed from them and `term_nodes` is
    accepted for signature compatibility but not used for guidance.

    Args:
    G_blank (networkx.DiGraph): Base graph providing the transition topology.
    traj_lengths (list[int]): Desired length of each trajectory.
    start_nodes (list, optional): Starting node IDs for each trajectory; if
        None, starts are drawn uniformly at random. Defaults to None.
    term_nodes (list, optional): Accepted for signature compatibility with
        `random_walker_ensemble`; currently unused. Defaults to None.
    Nrealiz (int): Number of independent realisations. Defaults to 1.

    Returns:
    list[list[list[str]]]: Outer list over realisations, then over
        trajectories, then node IDs within each walk.
    """
    walks_ensemble = []
    for r in range(Nrealiz):
        np.random.seed(r)
        walks = []
        for i, L in tqdm(enumerate(traj_lengths)):
            if start_nodes is not None:
                current_node = str(start_nodes[i])
            else:
                current_node = np.random.choice(list(G_blank.nodes()))

            walk = [current_node]
            for _ in range(L - 1):
                neighbors = list(G_blank.successors(current_node))
                if not neighbors:
                    break
                next_node = np.random.choice(neighbors)
                walk.append(next_node)
                current_node = next_node
            walks.append(walk)
        walks_ensemble.append(walks)
    return walks_ensemble



# %% RANDOM REWIRING


def h3_distance_distribution(G):
    """
    Compute the H3 grid distance for every edge in a graph.

    Each edge's endpoints are mapped to their 'coordID' attribute and the H3
    grid distance between the two hexagons is recorded. Edges whose endpoints
    are not connected on the H3 grid (e.g. across pentagon distortions)
    contribute NaN.

    Args:
    G (networkx.Graph): Graph whose nodes carry the H3 hexagon ID in their
        'coordID' attribute.

    Returns:
    list[float]: One H3 distance per edge in `G.edges()`, with NaN where
        the distance could not be computed.
    """
    distances = []
    
    # Iterate over all edges in the graph
    for edge in G.edges():
        node1, node2 = edge
        
        # Get the H3 coordID (hex ID) of the nodes
        hex_id1 = G.nodes[node1]['coordID']
        hex_id2 = G.nodes[node2]['coordID']
        
        # Compute the H3 distance between the two hexagons
        try:
            h3_dist = h3.h3_distance(hex_id1, hex_id2)
        except:
            h3_dist = np.nan
        distances.append(h3_dist)
    
    return distances


def rewire_network(G, max_dist, Nrealiz):
    """
    Rewire a graph by resampling edge targets within an H3 neighbourhood.

    The empirical distribution of H3 distances among the original edges is
    used as a sampling distribution: for each edge, candidate replacements
    are taken from the H3 k-ring of radius `max_dist` around the source, and
    a new target is drawn with probability proportional to the frequency of
    its H3 distance in the original network. The edge's weight is preserved
    on rewiring.

    Args:
    G (networkx.DiGraph): Input graph whose nodes carry a 'coordID' (H3
        hexagon ID) attribute and whose edges carry a 'weight' attribute.
    max_dist (int): Maximum H3 ring radius for candidate target hexagons.
    Nrealiz (int): Number of independent rewired graphs to produce.

    Returns:
    list[networkx.DiGraph]: One rewired graph per realisation, preserving
        the original out-degree of each source node.
    """
    a_hdist = np.hstack(h3_distance_distribution(G))
    a_hdistdistr = np.histogram(a_hdist[~np.isnan(a_hdist)], bins=np.arange(1, max_dist+2))[0]
    
    l_G = []
    for _ in range(Nrealiz):
        G_rewired = G.copy()  # Copy of the original graph
        
        for edge in tqdm(list(G_rewired.edges())):
            source, target = edge
            source_hexID = G_rewired.nodes[source]['coordID']
            potential_neighbors = [n for n in itertools.chain.from_iterable(h3.k_ring_distances(source_hexID, max_dist))
                                   if n != source_hexID]
            
            if not potential_neighbors:
                continue
            
            a_prob = np.array([a_hdistdistr[h3.h3_distance(source_hexID, n)-1] 
                               if h3.h3_distance(source_hexID, n) else 0 for n in potential_neighbors])
            if a_prob.sum() == 0:
                continue  # Skip if no valid neighbors are found

            a_prob = a_prob / a_prob.sum()  # Normalize probabilities
            new_target_hexID = np.random.choice(potential_neighbors, p=a_prob)

            # Find the node in G corresponding to new_target_hexID
            new_target_node = next((node for node, data in G_rewired.nodes(data=True) if data['coordID'] == new_target_hexID), None)
            if new_target_node and new_target_node != target:
                weight = G_rewired.edges[edge]['weight']
                G_rewired.remove_edge(source, target)
                G_rewired.add_edge(source, new_target_node, weight=weight)

        l_G.append(G_rewired)

    return l_G



def rewire_edges(G, Nrealiz, max_dist=3):
    """
    Rewire graph edges by sampling new targets from a precomputed H3
    distance distribution.

    Variant of `rewire_network` that uses an explicit candidate loop and
    integer probability accumulator. The empirical histogram of H3 edge
    distances drives the sampling weights, so the rewired ensemble has the
    same overall distance distribution as the input while randomising
    which specific targets each source connects to. Edge weights are
    preserved.

    Args:
    G (networkx.DiGraph): Input graph whose nodes carry a 'coordID' (H3
        hexagon ID) attribute and whose edges carry a 'weight' attribute.
    Nrealiz (int): Number of independent rewired graphs to produce.
    max_dist (int): Maximum H3 ring radius for candidate target hexagons.
        Defaults to 3.

    Returns:
    list[networkx.DiGraph]: One rewired graph per realisation.
    """    
    a_hdist = np.hstack(h3_distance_distribution(G))
    a_hdistdistr = np.histogram(a_hdist[~np.isnan(a_hdist)], bins=np.arange(1,max_dist+2))[0]
    
    l_G = []
    for n in tqdm(range(Nrealiz)):
        G_rewired = G.copy()  # Copy of the original graph
        # Iterate through all edges
        for edge in tqdm(list(G_rewired.edges())):
            source, target = edge
            
            # Get the current hexagon IDs for source and target nodes
            source_hexID = G_rewired.nodes[source]['coordID']
            
            # Find all potential neighbors of the source node within a certain k-ring radius
            potential_neighbors = list(itertools.chain.from_iterable(h3.k_ring_distances(source_hexID, max_dist)))
            # Remove the original target node from potential candidates
            potential_neighbors = [n for n in potential_neighbors if n != source_hexID]
            
            # If no valid neighbors are found, nothing to rewire
            if not potential_neighbors:
                continue
            
            # Calculate H3 distances to potential neighbors and normalize using the precomputed distribution
            k=0
            a_prob = np.zeros(len(potential_neighbors), dtype=int)
            for n in potential_neighbors:
                try:
                    d = h3.h3_distance(source_hexID, n)
                    a_prob[k] = a_hdistdistr[d-1]
                except: 
                    a_prob[k] = 0
                k+=1
                
            a_prob = a_prob / a_prob.sum()
    
            # Randomly select a new target node based on the distance distribution
            new_target_hexID = np.random.choice(potential_neighbors, p=a_prob)
            
            # Find the node in the graph corresponding to this new target hexID
            new_target_node = None
            for node, data in G.nodes(data=True):
                if data['coordID'] == new_target_hexID:
                    new_target_node = node
                    break
            
            if new_target_node:
                # Rewire: remove the original edge and add a new edge to the chosen node
                weight = G_rewired.edges[edge]['weight']
                G_rewired.remove_edge(source, target)
                G_rewired.add_edge(source, new_target_node, weight=weight)
            
        l_G.append(G_rewired)
    
    return l_G
