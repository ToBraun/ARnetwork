# Copyright (C) 2023 by
# Tobias Braun

#------------------ PATH ---------------------------#
import sys
#PATH = "/home/tobraun/Desktop/Postdoc/projects/#1_ClimXtreme/ARcatalog_shared/scripts/postprocessing"
PATH = "/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/scripts"
sys.path.insert(0, PATH)



# %% IMPORT MODULES

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import warnings
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
from scipy import stats
import itertools

#from dask import delayed
#import dask.array as da
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
    """Build a networkx graph from a list of hex IDs with optional elevation-based weights."""
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
    """Calculate the Haversine distance between two arrays of points on the Earth."""
    R = 6371  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c  # Distance in kilometers

def precompute_distances(G):
    """Precompute Haversine distances between all pairs of nodes based on their Latitude and Longitude attributes."""
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
    """Perform a random walk on a directed graph, optionally guided towards a node using Haversine distances."""
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


# def random_walker_ensemble(G_blank, traj_lengths, eps=0, start_nodes=None, term_nodes=None, Nrealiz=1, LC_cond=None):
#     Ntraj = len(traj_lengths)
#     """Perform random walks on a directed graph with different models, adjusting edge directions based on walker movement."""

#     # Precompute distances using Latitude and Longitude
#     distance_matrix = precompute_distances(G_blank)
#     node_indices = {str(node): idx for idx, node in enumerate(G_blank.nodes())}

#     realizations = []  # To store all realizations
#     for nrealiz in tqdm(range(Nrealiz)):
#         # Start with an empty directed graph to capture the walker's directions
#         G = G_blank.copy()#nx.DiGraph()

#         np.random.seed(nrealiz)
#         for n in tqdm(range(Ntraj)):
#             L = traj_lengths[n]
#             if term_nodes is not None and start_nodes is None:
#                 start_node = str(term_nodes[n])
#                 walk = random_walk_on_graph(G_blank, start_node, L, distance_matrix=distance_matrix, node_indices=node_indices)
#             elif start_nodes is not None and term_nodes is None:
#                 start_node = str(start_nodes[n])
#                 walk = random_walk_on_graph(G_blank, start_node, L, distance_matrix=distance_matrix, node_indices=node_indices)
#             elif start_nodes is None and term_nodes is None:
#                 start_node = np.random.choice(list(G_blank.nodes()))
#                 walk = random_walk_on_graph(G_blank, start_node, L, distance_matrix=distance_matrix, node_indices=node_indices)
#             elif start_nodes is not None and term_nodes is not None:
#                 start_node = str(start_nodes[n])
#                 final_node = str(term_nodes[n])
#                 walk = random_walk_on_graph(G_blank, start_node, L, final_node, distance_matrix=distance_matrix, node_indices=node_indices)

#             if LC_cond is None:
#                 # Traverse the walk and update edge directions and weights
#                 for i in range(len(walk) - 1):
#                     current_node = walk[i]
#                     next_node = walk[i + 1]

#                     # Set the direction and update the weight based on the actual walk
#                     if G.has_edge(current_node, next_node):
#                         G[current_node][next_node]['weight'] += 1  # Increment the edge weight
#                     else:
#                         G.add_edge(current_node, next_node, weight=1)  # Create edge with weight 1 if it does not exist
            
#             elif LC_cond == 'birth-death':
#                 # Update the edge direction only for genesis - termination locations
#                 current_node, next_node = walk[0], walk[-1]
#                 if G.has_edge(current_node, next_node):
#                     G[current_node][next_node]['weight'] += 1  # Increment the edge weight
#                 else:
#                     G.add_edge(current_node, next_node, weight=1)  # Create edge with weight 1 if it does not exist
            
#             else:
#                 print('Error: this function is currently not equipped to deal with the specified LC_cond.')
#                 return

#         # After finishing all walks, subtract 1 from each edge weight to account for the initial weight
#         for u, v, data in G.edges(data=True):
#             data['weight'] -= 1  # Adjust for initial edge weight of 1

#         # Remove edges with weight 0
#         edges_to_remove = [(u, v) for u, v, weight in G.edges(data="weight") if weight <= eps]
#         G.remove_edges_from(edges_to_remove)

#         # Store the realization of the graph
#         realizations.append(G.copy())

#     return realizations  # Return a list of Nrealiz graphs


def random_walker_ensemble(G_blank, traj_lengths, eps=0, start_nodes=None, term_nodes=None, 
                           Nrealiz=1, LC_cond=None, return_paths=True):
    """
    Perform random walks on a directed graph with different models, updating edge directions 
    based on walker movement. Optionally return the realized paths of the walkers.
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



# %% RANDOM REWIRING




def h3_distance_distribution(G):
    """
    Compute and return the distribution of H3 distances between connected nodes in the graph.
    
    Args:
    - G (nx.Graph): The graph, with each node having a 'coordID' attribute storing its H3 hexagon ID.
    
    Returns:
    - distances (list): A list of H3 distances between connected nodes.
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


# Function to preserve degree during rewiring
def rewire_network(G, max_dist, Nrealiz):
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
    """Rewire graph edges, sampling rewiring candidates based on precomputed distance distribution."""
    
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
            
            # If no valid neighbors are found, skip rewiring for this edge
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





