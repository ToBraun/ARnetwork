# Copyright (C) 2025 by
# Tobias Braun

#------------------ PATH ---------------------------#
# working directory
import sys
WDPATH = "/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/paper/scripts/ARnetlab"
sys.path.insert(0, WDPATH)


# %% IMPORT MODULES

# standard packages
from collections import defaultdict
import numpy as np
import pandas as pd
import warnings
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

# specific packages
from itertools import combinations
from tqdm import tqdm
import networkx as nx
from infomap import Infomap


# %% INTERNAL HELPER FUNCTIONS
# The functions below are not meant to be called from a script but are secondary
# functions that are called by the main functions below.



# %% NODE-BASED MEASURES

def clustering_coefficient(G):
    """
    Calculate clustering coefficients for all nodes in the graph.
    
    Args:
    G (networkx.DiGraph): The input graph.
    
    Returns:
    numpy.ndarray: Array of clustering coefficient values for each node.
    """
    return np.array(list(dict(nx.clustering(G)).values()))



def cycle_clustering_coefficient(G):
    """
    Calculate cycle-clustering coefficients for all nodes in the graph.
    
    Args:
    G (networkx.DiGraph): The input directed graph.
    
    Returns:
    numpy.ndarray: Array of cycle-clustering coefficient values for each node.
    """
    clustering = {}
    for node in G:
        neighbors = set(G.successors(node)) & set(G.predecessors(node))
        
        if len(neighbors) < 2:
            clustering[node] = 0.0
            continue
        
        possible_triangles = combinations(neighbors, 2)
        count = 0
        for u, v in possible_triangles:
            if G.has_edge(u, v) and G.has_edge(v, node):
                count += 1
            if G.has_edge(v, u) and G.has_edge(u, node):
                count += 1
        
        clustering[node] = count / (len(neighbors) * (len(neighbors) - 1))
    
    return np.array(list(clustering.values()))



def degree_centrality(G, correct_for_area=False):
    """
    Calculate degree centralities for all nodes in the graph.
    
    Parameters:
    - G (networkx.Graph): The input graph.
    - directed (bool): Whether the graph is directed. Default is True.
    - correct_for_area (bool): Whether to apply area correction. Default is True.
    
    Returns:
    numpy.ndarray: Array of degree centrality values for each node.
    """
    if correct_for_area:
        # Get latitude values for nodes
        lats = np.array(list(nx.get_node_attributes(G, "Latitude").values()))
        # Get adjacency matrix
        a_adj = nx.adjacency_matrix(G).toarray()
        # Calculate latitude correction factors
        a_latcorr = np.cos(np.radians(lats))
        # Calculate degree with area correction
        a_deg = np.sum(a_adj * a_latcorr, axis=1) / np.sum(a_latcorr)
    else:
        # Calculate degree without area correction
        a_deg = np.array(list(dict(G.degree(weight='weight')).values()))

    return a_deg


def divergence(G, weight='weight'):
    """
    Calculate divergence (out-degree minus in-degree) for all nodes in the graph.
    
    Parameters:
    - G (networkx.Graph): The input graph.
    - weight (str): The edge attribute to use as weight. Default is 'weight'.
    
    Returns:
    numpy.ndarray: Array of divergence values for each node.
    """
    # Calculate in-degrees and out-degrees
    a_in_degrees = np.array(list(dict(G.in_degree(weight=weight)).values()))
    a_out_degrees = np.array(list(dict(G.out_degree(weight=weight)).values()))
    # Calculate divergence
    a_div = a_out_degrees - a_in_degrees
    return a_div




def pagerank(G, fdamp=0.85, normalised=False):
    """
    Calculate PageRank for all nodes in the graph.
    
    Parameters:
    G (networkx.Graph): The input graph.
    
    Returns:
    numpy.ndarray: Array of PageRank values for each node.
    """
    # Calculate PageRank
    a_pagerank = np.array(list(nx.pagerank(G, weight='weight', alpha=fdamp).values()))
    if normalised:
        a_pagerank = a_pagerank / sum(a_pagerank) * 100  

    return a_pagerank


def compute_node_moisture_transport(indices, ivt, output='quantile', quantiles=None, thresholds=None):
    """
    Classify nodes by their net moisture gain or loss, based on incoming and outgoing IVT.

    Parameters:
    - indices : tuple of arrays
        Tuple (a_orig_indices, a_dest_indices) containing origin and destination node indices.
    - ivt : tuple of arrays
        Tuple (a_orig_ivt, a_dest_ivt) containing IVT values at the origin and destination nodes.
    - output : str
        Either 'quantile' or 'manual'. Determines how thresholds are defined.
    - quantiles : tuple of floats, optional
        If `output` is 'quantile', specifies the upper quantiles (e.g., (0.75, 0.85, 0.95)). Lower quantiles are inferred.
    - thresholds : list of floats, optional
        If `output` is 'manual', a list of exactly 6 thresholds separating the 7 node transport classes.

    Returns:
    - node_class : dict
        Dictionary mapping each node to a class in [-3, 3] representing net moisture gain/loss.
    """
    # unpack
    a_orig_indices, a_dest_indices = indices
    a_orig_ivt, a_dest_ivt =  ivt

    # Step 1: Collect all incoming and outgoing IVT values per node
    incoming_ivt = defaultdict(list)
    outgoing_ivt = defaultdict(list)

    for o_idx, d_idx, o_ivt, d_ivt in zip(a_orig_indices, a_dest_indices, a_orig_ivt, a_dest_ivt):
        outgoing_ivt[o_idx].append(d_ivt)  # node as origin: AR leaving node
        incoming_ivt[d_idx].append(o_ivt)  # node as destination: AR approaching node

    # Step 2: Compute net IVT gain/loss per node
    node_diffs = {}
    for node in set(list(incoming_ivt.keys()) + list(outgoing_ivt.keys())):
        avg_out = np.nanmean(outgoing_ivt.get(node, [np.nan]))
        avg_in = np.nanmean(incoming_ivt.get(node, [np.nan]))
        if np.isnan(avg_in) or np.isnan(avg_out):
            node_diffs[node] = np.nan
        else:
            node_diffs[node] = avg_out - avg_in  # gain/loss at this node

    # Step 3: Determine thresholds
    diffs_array = np.array([v for v in node_diffs.values() if not np.isnan(v)])
    if output == 'quantile':
        if quantiles is None:
            qh1, qh2, qh3 = .75, .85, .95
        else:
            qh1, qh2, qh3 = quantiles
        ql1, ql2, ql3 = 1 - qh1, 1 - qh2, 1 - qh3
        pthresh1, pthresh2, pthresh3 = np.nanquantile(diffs_array, qh1), np.nanquantile(diffs_array, qh2), np.nanquantile(diffs_array, qh3)
        nthresh1, nthresh2, nthresh3 = np.nanquantile(diffs_array, ql1), np.nanquantile(diffs_array, ql2), np.nanquantile(diffs_array, ql3)
        all_thresholds = [nthresh3, nthresh2, nthresh1, pthresh1, pthresh2, pthresh3]

    elif output == 'manual':
        if thresholds is None or len(thresholds) != 6:
            raise ValueError("Manual output requires a 'thresholds' argument with exactly 6 values.")
        all_thresholds = sorted(thresholds)

    else:
        raise ValueError('Specification error: "output" must be either "quantile" or "manual".')

    # Step 4: Assign class per node
    node_class = {}
    for node, diff in node_diffs.items():
        if np.isnan(diff):
            node_class[node] = np.nan
        else:
            node_class[node] = np.digitize(diff, all_thresholds) - 3  # range [-3, 3]

    return node_class



# %% EDGE-BASED MEASURES


def invert_weights(G, weight_attr='weight'):
    """
    Returns a copy of the input graph with inverted edge weights. 
    Apply BEFORE edge betweenness calculations to MAXIMIZE AR frequencies along high-EBC edges.

    Parameters:
    - G (networkx.Graph or networkx.DiGraph): The input graph with weighted edges.
    - weight_attr (str): The name of the edge attribute to invert (default is 'weight').

    Returns:
    - G_inverted (networkx.Graph or networkx.DiGraph): A copy of the graph with inverted weights.
      Edges with zero or negative weights are assigned infinite weight (np.inf).
    """
    G_inverted = G.copy()

    # Invert weights: 1 / weight if positive, else set to infinity
    for u, v, data in G_inverted.edges(data=True):
        original_weight = data[weight_attr]
        if original_weight > 0:
            data[weight_attr] = 1 / original_weight
        else:
            data[weight_attr] = np.inf

    return G_inverted



def compute_edge_moisture_transport(indices, ivt, output, quantiles=None, thresholds=None):
    """
    Classify moisture transport across edges based on IVT differences between origin and destination nodes.
    
    Parameters:
    - indices : tuple of arrays
        Tuple (a_orig_indices, a_dest_indices) containing arrays of origin and destination node indices for each AR step.
    - ivt : tuple of arrays
        Tuple (a_orig_ivt, a_dest_ivt) containing IVT values at the origin and destination nodes for each AR step.
    - output : str
        Either 'quantile' or 'manual'. Determines how thresholds are defined.
    - quantiles : tuple of floats, optional
        If `output` is 'quantile', specifies the upper quantiles (e.g., (0.75, 0.85, 0.95)). Lower quantiles are inferred.
    - thresholds : list of floats, optional
        If `output` is 'manual', a list of exactly 6 thresholds separating the 7 IVT difference classes.
    
    Returns:
    - sign_dict : dict
        Dictionary mapping (origin, destination) edge tuples to a class in [-3, 3] representing average IVT difference.
    """
    # Calculate the IVT differences
    a_orig_indices, a_dest_indices = indices
    a_orig_ivt, a_dest_ivt = ivt
    a_ivt_diffs = a_dest_ivt - a_orig_ivt

    # Determine thresholds
    if output == 'quantile':
        if quantiles is None:
            qh1, qh2, qh3 = .75, .85, .95
        else:
            qh1, qh2, qh3 = quantiles
        ql1, ql2, ql3 = 1 - qh1, 1 - qh2, 1 - qh3
        pthresh1, pthresh2, pthresh3 = np.nanquantile(a_ivt_diffs, qh1), np.nanquantile(a_ivt_diffs, qh2), np.nanquantile(a_ivt_diffs, qh3)
        nthresh1, nthresh2, nthresh3 = np.nanquantile(a_ivt_diffs, ql1), np.nanquantile(a_ivt_diffs, ql2), np.nanquantile(a_ivt_diffs, ql3)
        all_thresholds = [nthresh3, nthresh2, nthresh1, pthresh1, pthresh2, pthresh3]

    elif output == 'manual':
        if thresholds is None or len(thresholds) != 6:
            raise ValueError("Manual output requires a 'thresholds' argument with exactly 6 values.")
        all_thresholds = sorted(thresholds)  # Ensure thresholds are in increasing order

    else:
        raise ValueError('Specification error: "output" must be either "quantile" or "manual".')

    # Build dictionary of IVT differences
    ivt_diff_dict = {}
    for o_idx, d_idx, ivt_diff in zip(a_orig_indices, a_dest_indices, a_ivt_diffs):
        if (o_idx, d_idx) not in ivt_diff_dict:
            ivt_diff_dict[(o_idx, d_idx)] = []
        ivt_diff_dict[(o_idx, d_idx)].append(ivt_diff)

    # Assign class per edge
    sign_dict = {}
    for key, values in ivt_diff_dict.items():
        if not values:
            sign_dict[key] = np.nan
            continue
        avg_ivt = np.mean(values)
        edge_class = np.digitize(avg_ivt, all_thresholds) - 3  # maps to range -3 to +3
        sign_dict[key] = edge_class

    return sign_dict



# %% COMMUNITY DETECTION


def detect_non_hierarchical_communities(G, filename=None, use_node_weights_as_flow=False, return_flows=False, seed=None):
    """
    Detects flat (non-hierarchical) communities in a directed network using the Infomap algorithm.

    Parameters:
    - G (networkx.DiGraph): The input directed graph with edge weights.
    - filename (str, optional): If provided, saves the Infomap tree to this file.
    - use_node_weights_as_flow (bool): If True, node weights are used for flow calculations.
    - return_flows (bool): If True, returns the flow values between nodes.
    - seed (int, optional): Random seed for reproducibility.

    Returns:
    - communities (dict): A mapping from node to community label.
    - d_flow (pd.DataFrame, optional): Flow values between nodes (only if return_flows is True).
    """
    # Initialize Infomap with non-hierarchical (two-level) option
    infomap = Infomap(
        no_self_links=True,
        directed=True,
        two_level=True,
        verbosity_level=1,
        silent=False,
        use_node_weights_as_flow=use_node_weights_as_flow,
        seed=seed
    )

    # Normalize and add edge weights
    total_weight = sum(data['weight'] for u, v, data in G.edges(data=True))
    for u, v, data in G.edges(data=True):
        weight = data['weight'] / total_weight
        infomap.add_link(u, v, weight)

    # Add isolated nodes to the map
    for node in G.nodes():
        if G.degree(node) == 0:
            infomap.add_node(node)

    # Run the Infomap algorithm
    infomap.run()
    communities = infomap.get_modules()

    # Optionally write the flow tree to a file
    if filename is not None:
        infomap.write_flow_tree(filename=filename)

    # Return community assignment and optionally the flow values
    if return_flows:
        l_flows = []
        for link in infomap.get_links(data="flow"):
            node_i, node_j, flow_value = link
            l_flows.append((int(node_i), int(node_j), flow_value))
        d_flow = pd.DataFrame(l_flows, columns=["Node1", "Node2", "Flow"])
        return communities, d_flow
    else:
        return communities




def detect_hierarchical_communities(G, filename=None, use_node_weights_as_flow=False, return_flows=False):
    """
    Detects hierarchical (multi-level) communities in a directed network using the Infomap algorithm.

    Parameters:
    - G (networkx.DiGraph): The input directed graph with edge weights.
    - filename (str, optional): If provided, saves the Infomap tree to this file.
    - use_node_weights_as_flow (bool): If True, node weights are used for flow calculations.
    - return_flows (bool): If True, returns the flow values between nodes.

    Returns:
    - a_lvlcomms (np.ndarray): Community labels for each node at each hierarchy level,
                                shape = (N_levels, N_nodes), starting from level 0.
    - d_flow (pd.DataFrame, optional): Flow values between nodes (only if return_flows is True).
    """
    # Initialize Infomap with hierarchical detection enabled
    infomap = Infomap(
        no_self_links=True,
        directed=True,
        verbosity_level=1,
        silent=False,
        use_node_weights_as_flow=use_node_weights_as_flow
    )

    # Normalize and add weighted edges
    total_weight = sum(data['weight'] for u, v, data in G.edges(data=True))
    for u, v, data in G.edges(data=True):
        weight = data['weight'] / total_weight
        infomap.add_link(u, v, weight)

    # Add isolated nodes
    for node in G.nodes():
        if G.degree(node) == 0:
            infomap.add_node(node)

    # Run the Infomap algorithm
    infomap.run()

    # Retrieve community assignments for all levels
    level_1_communities = infomap.get_modules()
    level_2_communities = infomap.get_multilevel_modules()

    # Optionally write the flow tree to a file
    if filename is not None:
        infomap.write_flow_tree(filename=filename)

    # Optionally extract flow values between nodes
    if return_flows:
        l_flows = []
        for link in infomap.get_links(data="flow"):
            node_i, node_j, flow_value = link
            l_flows.append((int(node_i), int(node_j), flow_value))
        d_flow = pd.DataFrame(l_flows, columns=["Node1", "Node2", "Flow"])

    # Create a community label matrix with shape (N_levels, N_nodes)
    Nlvl = len(level_2_communities[1])
    a_lvlcomms = np.zeros((Nlvl + 1, len(G.nodes())), dtype=int)
    a_lvlcomms[0, :] = [level_1_communities[v] for v in G.nodes()]
    lab0 = a_lvlcomms[0, :].max()

    for lvl in range(1, Nlvl):
        a_lvlcomms[lvl, :] = [lab0 + level_2_communities[v][lvl] for v in G.nodes()]
        lab0 = a_lvlcomms[lvl, :].max()

    if return_flows:
        return a_lvlcomms, d_flow
    else:
        return a_lvlcomms





def filter_small_communities(lvlcomms, min_size=10):
    """
    Filters out small communities across hierarchical levels by marking nodes belonging to 
    communities smaller than a specified minimum size.
    
    For each level, if a community has fewer than `min_size` members, all its nodes are flagged.
    These flags propagate to higher levels, ensuring that once a node is part of a small community,
    it is considered invalid at all subsequent levels.
    
    Parameters:
    - lvlcomms (np.ndarray): Community labels for each node at each hierarchy level,
                             shape = (N_levels, N_nodes), starting from level 0.
    - min_size (int): Minimum community size to be considered valid.
    
    Returns:
    - mask (np.ndarray): Boolean array of shape (N_levels, N_nodes) indicating which nodes
                         are part of small communities (True means invalid).
    """
    # Get the number of levels and nodes
    num_levels, num_nodes = lvlcomms.shape
    
    # Initialize an array to track the mask for nodes to be flagged
    mask = np.zeros_like(lvlcomms, dtype=bool)
    
    # Iterate over levels
    for lvl in range(num_levels):
        # Get unique communities and their sizes at this level
        unique_comms, counts = np.unique(lvlcomms[lvl, :], return_counts=True)
        
        # Iterate over communities
        for comm, count in zip(unique_comms, counts):
            if count < min_size:
                # Find nodes in the small community
                small_comm_nodes = np.where(lvlcomms[lvl, :] == comm)[0]
                
                # Mark these nodes in the mask for this level and higher levels
                mask[lvl:, small_comm_nodes] = True
    
    return mask




def module_flow(commlabs_by_level, flows):
    """
    Aggregates node-to-node flow values into module-to-module flows at each hierarchical level 
    of community detection.
    
    For each level, this function computes a flow matrix representing the total flow between 
    detected communities (modules), excluding unassigned nodes (marked as -999, part of a community that is too small).
    It also computes the total incoming flow to each module.
    
    Parameters:
    - commlabs_by_level (np.ndarray): Community labels for each node at each hierarchy level,
                                      shape = (N_levels, N_nodes), with unassigned nodes marked as -999.
    - flows (pd.DataFrame): DataFrame with columns ['Node1', 'Node2', 'Flow'], representing 
                            flow values between individual nodes.
    
    Returns:
    - d_dict_flows (dict): Dictionary mapping each level to a 1D array of total incoming flow 
                           for each module at that level.
    - d_flow_matrices (dict): Dictionary mapping each level to a 2D NumPy array representing 
                              the flow matrix between modules (shape = [n_comms, n_comms]).
    """

    # Number of nodes (columns) and levels (rows)
    num_levels, num_nodes = commlabs_by_level.shape
    
    # Initialize a dictionary to store the flow matrix between modules at each level
    d_flow_matrices, d_dict_flows = {}, {}
    
    # For each level, compute the flow matrix between modules
    for level in tqdm(range(num_levels)):
        # Get unique community labels at the current level (excluding -999)
        communities_at_level = commlabs_by_level[level, :]
        unique_communities = np.unique(communities_at_level[communities_at_level != -999])

        # Initialize a flow matrix for the communities (size: number of unique communities x number of unique communities)
        flow_matrix = np.zeros((len(unique_communities), len(unique_communities)))
        
        # Map community label to index in flow matrix
        community_to_index = {community: idx for idx, community in enumerate(unique_communities)}
        
        # Iterate over all rows in the d_flows dataframe to map flows to modules (communities)
        for _, row in flows.iterrows():
            node1, node2, flow_value = int(row['Node1']), int(row['Node2']), row['Flow']
            
            # Get the community labels of the nodes at this level
            community1 = communities_at_level[node1]
            community2 = communities_at_level[node2]
            
            # Skip the link if either community is -999 (unassigned nodes)
            if community1 == -999 or community2 == -999:
                continue
            
            # Map community labels to indices in the flow matrix
            idx1 = community_to_index[community1]
            idx2 = community_to_index[community2]
            
            # Add the flow value to the flow matrix for the corresponding communities
            flow_matrix[idx1, idx2] += flow_value
            #flow_matrix[idx2, idx1] += flow_value  # Since it's undirected (if required)
            
            # Compute the total flow for each module (sum of flows for each community)
            total_flow = flow_matrix.sum(axis=0)  # Sum along rows or columns to get total flow for each module
            
        
        # Store the flow matrix for this level in the dictionary
        d_flow_matrices[level] = flow_matrix
        d_dict_flows[level] = total_flow
    
    return d_dict_flows, d_flow_matrices




def filter_by_flow(lvlcomms, d_total_flows, flow_threshold=0.8):
    """
    Filters communities based on their flow contribution, retaining the top `n` communities
    that together contribute at least `flow_threshold` of the total flow.

    Parameters:
        lvlcomms (np.ndarray): Community labels matrix (levels x nodes).
        d_total_flows (dict): Dictionary containing total flow for each module at each level.
        flow_threshold (float): Fraction of total flow to retain (0.0 to 1.0).

    Returns:
        np.ndarray: Filtered community labels matrix with small-flow communities set to -999.
    """
    # Get the number of levels and nodes
    num_levels, num_nodes = lvlcomms.shape

    # Initialize an array to track if a node is already marked as NaN
    nan_mask = np.zeros(num_nodes, dtype=bool)

    # Copy the community labels to modify them
    a_lvlcomms_filtered = np.copy(lvlcomms)

    # Iterate over levels
    for lvl in range(num_levels):
        # Get the total flow for each community at this level
        total_flows = d_total_flows[lvl]

        # Sort communities by their total flow in descending order
        sorted_indices = np.argsort(total_flows)[::-1]
        sorted_flows = total_flows[sorted_indices]

        # Compute cumulative flow fraction
        cumulative_flow = np.cumsum(sorted_flows) / np.sum(sorted_flows)

        # Determine the cutoff index to retain `flow_threshold` of the total flow
        cutoff_idx = np.searchsorted(cumulative_flow, flow_threshold) + 1

        # Get the communities to retain
        retained_communities = sorted_indices[:cutoff_idx]

        # Map retained communities back to their original labels
        unique_comms = np.unique(lvlcomms[lvl, :])
        retained_labels = unique_comms[retained_communities]

        # Identify nodes in communities not retained
        for comm in unique_comms:
            if comm not in retained_labels:
                small_flow_nodes = np.where(lvlcomms[lvl, :] == comm)[0]

                # Mark these nodes as NaN at this level and higher levels
                nan_mask[small_flow_nodes] = True

        # Apply the mask to the current level
        a_lvlcomms_filtered[lvl, nan_mask] = -999

    return a_lvlcomms_filtered



# %% TRAJECTORY CLASSIFICATION


def classify_trajectories_simple(
    observed_paths, G, 
    ebc_attr='edge_betweenness',
    min_length=4,
    low_stray_quantile=0.33,
    high_stray_quantile=0.66,
    scale='log'):
    """
    Classify trajectories into conformists, intermediates, and deviants based on stray index.

    Parameters:
    -----------
    observed_paths : list of arrays/lists
        Each item is a sequence of H3 hex IDs (trajectory of one AR).
    G : networkx.Graph or DiGraph
        Graph with edge betweenness centrality stored as an edge attribute.
    ebc_attr : str
        Edge attribute name for edge betweenness centrality.
    min_length : int
        Minimum number of hex steps in a trajectory.
    low_stray_quantile : float
        Lower quantile threshold for stray index.
    high_stray_quantile : float
        Upper quantile threshold for stray index.
    scale : str
        'linear' or 'log' scaling of EBC. 'log' emphasizes detours more strongly.

    Returns:
    --------
    result : dict
        Dictionary with keys 'conformists', 'straddlers', 'strays', each containing:
            - 'indices': list of trajectory indices
            - 'stray_scores': list of stray index values
    """
    # Extract all EBC values for scaling
    all_ebc = [edata[ebc_attr] for _, _, edata in G.edges(data=True) if ebc_attr in edata]
    if not all_ebc:
        raise ValueError(f"Graph edges must have '{ebc_attr}' attribute.")

    if scale == 'log':
        log_ebc_vals = np.log(np.array(all_ebc) + 1e-12)
        min_log, max_log = np.min(log_ebc_vals), np.max(log_ebc_vals)
    else:
        max_ebc = max(all_ebc)

    stray_index_all = []

    for path in tqdm(observed_paths):
        path = path.tolist() if isinstance(path, np.ndarray) else path
        filtered_path = [path[0]]
        for u, v in zip(path[:-1], path[1:]):
            if u != v:
                filtered_path.append(v)

        if len(filtered_path) < min_length:
            stray_index_all.append(np.nan)
            continue

        ebc_values = []
        for u, v in zip(filtered_path[:-1], filtered_path[1:]):
            try:
                ebc = G[u][v][ebc_attr]
                if scale == 'log':
                    log_ebc = np.log(ebc + 1e-12)
                    norm_ebc = (log_ebc - min_log) / (max_log - min_log + 1e-12)
                else:
                    norm_ebc = ebc / (max_ebc + 1e-12)
                ebc_values.append(norm_ebc)
            except KeyError:
                ebc_values.append(np.nan)

        stray_index = 1 - np.nanmedian(ebc_values)
        stray_index_all.append(stray_index)

    # Compute thresholds
    high_stray_thresh = np.nanquantile(stray_index_all, high_stray_quantile)
    low_stray_thresh = np.nanquantile(stray_index_all, low_stray_quantile)

    # Classification results
    result = {
        'conformists': {'indices': [], 'stray_scores': []},
        'straddlers': {'indices': [], 'stray_scores': []},
        'strays': {'indices': [], 'stray_scores': []}
    }

    for i, sidx in enumerate(stray_index_all):
        if np.isnan(sidx):
            continue

        if sidx <= low_stray_thresh:
            result['conformists']['indices'].append(i)
            result['conformists']['stray_scores'].append(sidx)
        elif sidx >= high_stray_thresh:
            result['strays']['indices'].append(i)
            result['strays']['stray_scores'].append(sidx)
        else:
            result['straddlers']['indices'].append(i)
            result['straddlers']['stray_scores'].append(sidx)

    return result





def compute_community_flow_ratio(d_flow, communities):
    """
    Computes the ratio of intra-community to inter-community flow for each community 
    in an ARTN, based on flow values from Infomap.

    The intra-community flow sums flows between nodes within the same community, while
    the inter-community flow sums flows from nodes in a community to nodes outside of it.

    Parameters:
    - d_flow (pd.DataFrame): DataFrame with columns ['Node1', 'Node2', 'Flow'], representing
                             directed flow between nodes in the ARTN.
    - communities (dict): Dictionary mapping node IDs to their community label (excluding -999 for unassigned/too small).

    Returns:
    - ratios (dict): Dictionary mapping community IDs to intra/inter flow ratio. NaN if inter-community flow is zero.
    """

    # Remove nodes with invalid community labels
    valid_communities = {k: v for k, v in communities.items() if v != -999}
    
    # Create a reverse mapping from community -> set of nodes
    comm_to_nodes = defaultdict(set)
    for node, comm in valid_communities.items():
        comm_to_nodes[comm].add(node)
    
    # Initialize dictionaries to store intra- and inter-community flows
    intra_flow = defaultdict(float)
    inter_flow = defaultdict(float)

    for _, row in d_flow.iterrows():
        i, j, flow = row['Node1'], row['Node2'], row['Flow']
        if i not in valid_communities or j not in valid_communities:
            continue
        
        ci = valid_communities[i]
        cj = valid_communities[j]

        if ci == cj:
            intra_flow[ci] += flow
        else:
            inter_flow[ci] += flow  # Flow leaving community ci
            # Optionally also: inter_flow[cj] += flow, if flow is undirected/symmetric

    # Compute the ratio for each community
    comms = sorted(comm_to_nodes)
    ratios = {}
    for c in comms:
        if inter_flow[c] > 0:
            ratios[c] = intra_flow[c] / inter_flow[c]
        else:
            ratios[c] = np.nan  # Avoid division by zero or undefined

    return ratios




def compute_jaccard_set_similarity(community_sets):
    """
    Computes the average Jaccard similarity between all pairs of community sets.

    This function evaluates how much the sets overlap across different community
    realisations, telling us how consistent the Infomap output actually is.

    Parameters:
    - community_sets (list of sets): List of node sets, each representing a community 
                                     from a specific realisation or level.

    Returns:
    - mean_similarity (float): Average Jaccard similarity across all set pairs. NaN if no valid pairs exist.
    """

    sims = []
    for s1, s2 in combinations(community_sets, 2):
        if not s1 or not s2:
            continue
        intersection = len(s1 & s2)
        union = len(s1 | s2)
        sims.append(intersection / union if union > 0 else 0)
    return np.mean(sims) if sims else np.nan



def compute_heterogeneity_setwise(d_node_comm, seeds):
    """
    Computes the heterogeneity/label inconsistency of each node's community membership across multiple 
    Infomap realisations of the same ARTN.
    
    A node is considered heterogeneous if it is assigned to structurally different 
    communities across realisations. Jaccard similarity is used to quantify consistency.
    
    Parameters:
    - d_node_comm (pd.DataFrame): DataFrame where each column 'community_i' corresponds to
                                  community labels from seed i. Rows represent nodes.
    - seeds (list): List of seed identifiers (gotta match the order of columns in d_node_comm).
    
    Returns:
    - d_node_comm (pd.DataFrame): Same as input but with an additional column 'heterogeneity' 
                                  (values between 0 and 1, or NaN if unassigned in all runs).
    """

    # Build reverse index: seed -> comm_label -> set of node indices
    seed_to_communities = {seed: defaultdict(set) for seed in seeds}
    for idx, row in d_node_comm.iterrows():
        for i, seed in enumerate(seeds):
            label = row[f'community_{i+1}']
            seed_to_communities[seed][label].add(idx)  # include -999

    heterogeneity = []
    for idx, row in tqdm(d_node_comm.iterrows()):
        # Extract labels across seeds
        labels = [row[f'community_{i+1}'] for i in range(len(seeds))]
        
        # If node is always unassigned, mark as NaN
        if all(label == -999 for label in labels):
            heterogeneity.append(np.nan)
        else:
            # Gather the community sets for each label across seeds
            community_sets = [
                seed_to_communities[seeds[i]][label]
                for i, label in enumerate(labels)
            ]
            # Compute Jaccard similarity across these sets
            avg_sim = compute_jaccard_set_similarity(community_sets)
            heterogeneity.append(1 - avg_sim)  # 1 = fully heterogeneous, 0 = consistent

    d_node_comm["heterogeneity"] = heterogeneity
    return d_node_comm


def compute_jaccard_similarity(s1, s2):
    """
    Computes the Jaccard similarity between two sets.
    
    The Jaccard similarity quantifies set overlap as the ratio of intersection size
    to union size.
    
    Parameters:
    - s1 (set): First set of elements.
    - s2 (set): Second set of elements.
    
    Returns:
    - similarity (float): Jaccard similarity between s1 and s2 (range: 0 to 1).
    """
    intersection = len(s1 & s2)
    union = len(s1 | s2)
    return intersection / union if union > 0 else 0



def most_consistent_community_set(community_sets):
    """
    Identifies the most central/consistent partition
    among multiple realisations based on average Jaccard similarity.
    
    Parameters:
    - community_sets (list of sets): List of node sets representing the community 
                                     a node belongs to in each realisation.
    
    Returns:
    - best_set (set): The community set that is most similar (on average) to the others.
    """

    if len(community_sets) == 1:
        return community_sets[0]
    
    scores = []
    for i, s1 in enumerate(community_sets):
        sim_sum = sum(compute_jaccard_similarity(s1, s2) for j, s2 in enumerate(community_sets) if i != j)
        avg_sim = sim_sum / (len(community_sets) - 1)
        scores.append(avg_sim)
    
    best_index = np.argmax(scores)
    return community_sets[best_index]


def compute_consensus_communities_setwise(d_node_comm, seeds):
    """
   Constructs consensus communities by identifying the most consistent community 
   membership across Infomap realisations of an ARTN.

   Each node is assigned to the community set that overlaps most consistently across 
   realisations, based on Jaccard similarity. Unassigned nodes remain -999.

   Parameters:
   - d_node_comm (pd.DataFrame): DataFrame with columns 'community_i' for each realisation,
                                 where each row corresponds to a node.
   - seeds (list): List of seed identifiers corresponding to community columns.

   Returns:
   - d_node_comm (pd.DataFrame): Same as input with an added 'consensus_community' column 
                                 containing integer community labels.
   """

    # Build reverse index: seed -> comm_label -> set of node indices
    seed_to_communities = {seed: defaultdict(set) for seed in seeds}
    for idx, row in d_node_comm.iterrows():
        for i, seed in enumerate(seeds):
            label = row[f'community_{i+1}']
            if label != -999:
                seed_to_communities[seed][label].add(idx)

    # For each node, collect its community sets across runs
    node_consensus_sets = []
    for idx, row in tqdm(d_node_comm.iterrows()):
        sets_for_node = []
        for i, seed in enumerate(seeds):
            label = row[f'community_{i+1}']
            if label != -999:
                comm_set = seed_to_communities[seed][label]
                sets_for_node.append(comm_set)
        if sets_for_node:
            best_set = most_consistent_community_set(sets_for_node)
            node_consensus_sets.append(best_set)
        else:
            node_consensus_sets.append(None)

    # Assign unique integer labels to consensus sets
    set_to_id = {}
    consensus_ids = []
    current_id = 1
    for s in node_consensus_sets:
        if s is None:
            consensus_ids.append(-999)
        else:
            key = frozenset(s)
            if key not in set_to_id:
                set_to_id[key] = current_id
                current_id += 1
            consensus_ids.append(set_to_id[key])

    d_node_comm["consensus_community"] = consensus_ids
    return d_node_comm


def compute_realisation_centrality(d_node_comm, seeds):
    """
    Evaluates the centrality of each Infomap realisation by computing the average 
    heterogeneity of its communities relative to all other runs on the same ARTN.

    Centrality is defined as how consistently a realisation groups nodes compared 
    to other realisations, using Jaccard similarity.

    Parameters:
    - d_node_comm (pd.DataFrame): DataFrame with one column per realisation ('community_i'),
                                  containing node-level community labels.
    - seeds (list): List of seed identifiers corresponding to community columns.

    Returns:
    - most_central_seed (any): The seed identifier corresponding to the most central realisation.
    - centrality_scores (dict): Mapping from seed to average heterogeneity across nodes.
    """

    # Step 1: Build reverse index for all seed community sets
    seed_to_communities = {seed: defaultdict(set) for seed in seeds}
    for idx, row in d_node_comm.iterrows():
        for i, seed in enumerate(seeds):
            label = row[f'community_{i+1}']
            seed_to_communities[seed][label].add(idx)

    # Step 2: For each realisation, compute average heterogeneity across all nodes
    centrality_scores = {}
    for j, seed_ref in tqdm(enumerate(seeds)):
        seed_name = f'community_{j+1}'
        heterogeneity_vals = []
        for idx, row in d_node_comm.iterrows():
            target_label = row[seed_name]
            target_set = seed_to_communities[seed_ref][target_label]

            # Compare to all other realisations
            sim_vals = []
            for k, seed_cmp in enumerate(seeds):
                if k == j:
                    continue
                cmp_label = row[f'community_{k+1}']
                cmp_set = seed_to_communities[seed_cmp][cmp_label]
                # Jaccard simmi
                intersection = len(target_set & cmp_set)
                union = len(target_set | cmp_set)
                sim = intersection / union if union > 0 else 1.0
                sim_vals.append(sim)
            if sim_vals:
                heterogeneity_vals.append(1 - np.mean(sim_vals))
        centrality_scores[seed_ref] = np.mean(heterogeneity_vals)

    # Step 3: Return the seed with lowest mean heterogeneity
    most_central_seed = min(centrality_scores, key=centrality_scores.get)
    return most_central_seed, centrality_scores


def pick_most_consistent_seed(d_node_comm, d_node_comm_disp, seeds):
    """
    Selects the most consistent Infomap realisation on an ARTN by evaluating 
    how well its communities group nodes with low heterogeneity.
    This is the one displayed in the main figure 5.

    The realisation whose communities contain the most internally consistent 
    (i.e., low label incosistency) nodes is the chosen one!

    Parameters:
    - d_node_comm (pd.DataFrame): DataFrame with community labels for each realisation.
    - d_node_comm_disp (pd.DataFrame): Same DataFrame with a 'heterogeneity' column for each node.
    - seeds (list): List of seed identifiers corresponding to community columns.

    Returns:
    - best_seed (any): The seed identifier of the most consistent realisation.
    - scores (dict): Mapping from seed to its average heterogeneity-weighted community score.
    """

    heterogeneity = d_node_comm_disp['heterogeneity']
    scores = {}

    for i, seed in enumerate(seeds):
        col = f'community_{i+1}'
        comms = d_node_comm[col]
        total_score = 0
        for comm_id in comms.unique():
            nodes_in_comm = d_node_comm.index[comms == comm_id]
            avg_het = heterogeneity.loc[nodes_in_comm].mean()
            total_score += avg_het * len(nodes_in_comm)  # weighted sum
        mean_score = total_score / len(d_node_comm)
        scores[seed] = mean_score

    # Return the seed whose communities group together lowest heterogeneity nodes
    best_seed = min(scores, key=scores.get)
    return best_seed, scores



def assign_flow_ratios_to_nodes(G, d_node_comm, seed_idx):
    """
    Parameters:
    - G: networkx.DiGraph with edge weights (e.g., 'weight')
    - d_node_comm: DataFrame with node-community assignments
    - seed_idx: int, index of the selected community realisation (1-based, e.g., 37 for 'community_37')

    Returns:
    - d_node_comm: same DataFrame with a new column 'flow_ratio_{seed_idx}'
    """
    comm_col = f'community_{seed_idx}'
    comm_to_nodes = d_node_comm.groupby(comm_col).groups  # dict: community_id -> list of node indices

    flow_ratio_per_comm = {}

    for comm_id, nodes in comm_to_nodes.items():
        node_set = set(nodes)
        internal_flow = 0
        total_outflow = 0
        for node in node_set:
            for _, tgt, data in G.out_edges(node, data=True):
                w = data.get("weight", 1)
                total_outflow += w
                if tgt in node_set:
                    internal_flow += w
        if total_outflow > 0:
            flow_ratio = internal_flow / total_outflow
        else:
            flow_ratio = np.nan  # or 0.0
        flow_ratio_per_comm[comm_id] = flow_ratio

    # Assign flow ratio to each node
    d_node_comm[f'flow_ratio_{seed_idx}'] = d_node_comm[comm_col].map(flow_ratio_per_comm)

    return d_node_comm



def relabel_communities(community_dict):
    """
    Relabels community IDs to a consecutive integer range starting from 0.
    
    Useful for normalizing community labels across different realisations or outputs
    from Infomap on an ARTN. The relative grouping of nodes remains unchanged.
    
    Parameters:
    - community_dict (dict): Dictionary mapping node IDs to arbitrary community labels.
    
    Returns:
    - relabeled_dict (dict): Dictionary mapping node IDs to new integer labels (0, 1, 2, ...).
    """

    unique_labels = sorted(set(community_dict.values()))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    return {node: label_map[community_dict[node]] for node in community_dict}


