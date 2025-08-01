# Copyright (C) 2025 by
# Tobias Braun

#------------------ PATHS ---------------------------#

# working directory
import sys
WDPATH = "/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/paper/scripts/ARnetlab"
sys.path.insert(0, WDPATH)
# input and output
INPUT_PATH = '/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/paper/data/'
OUTPUT_PATH = '/Users/tbraun/Desktop/projects/#B_ARTN_LPZ/paper/data/output/ebc/'


# %% IMPORT MODULES

# standard packages
import numpy as np

# specific packages
import networkx as nx


# %% FUNCTION


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


# %% PARAMETERS

Nrealiz = 200
loc = 'centroid'

# %% RANDOM NETWORKS

# RANDOM NETWORKS
l_Gcons_rndm, l_Gcons_genesis, l_Gcons_term, l_Gcons_rewired = [], [], [], []
for n in range(Nrealiz):
    l_Gcons_rndm.append(nx.read_gml(INPUT_PATH + 'random_graphs/l_Gwalk_rndm_' + str(n) + '._cons_' + loc + '.gml'))
    l_Gcons_rewired.append(nx.read_gml(INPUT_PATH + 'random_graphs/l_G_rewired_' + str(n) + '_cons_' + loc + '.gml'))
    l_Gcons_genesis.append(nx.read_gml(INPUT_PATH + 'random_graphs/l_Gwalk_genesis_' + str(n) + '_cons_' + loc + '.gml'))
    l_Gcons_term.append(nx.read_gml(INPUT_PATH + 'random_graphs/l_Gwalk_term_' + str(n) + '_cons_' + loc + '.gml'))


# %% EDGE BETWEENNESS CENTRALITIES


# # FRW
l_Gbetw_rndm = []
for nr in range(Nrealiz):
    Gs = l_Gcons_rndm[nr]
    # Assign edge betweenness centralities
    G = invert_weights(Gs)
    d_ebetw = nx.edge_betweenness_centrality(G, weight='weight')
    nx.set_edge_attributes(G, d_ebetw, "edge_betweenness")
    l_Gbetw_rndm.append(G)


# # GENESIS
l_Gbetw_genesis = []
for nr in range(Nrealiz):
    Gs = l_Gcons_genesis[nr]
    # Assign edge betweenness centralities
    G = invert_weights(Gs)
    d_ebetw = nx.edge_betweenness_centrality(G, weight='weight')
    nx.set_edge_attributes(G, d_ebetw, "edge_betweenness")
    l_Gbetw_genesis.append(G)


# # TERMINATION
l_Gbetw_term = []
for nr in range(Nrealiz):
    Gs = l_Gcons_term[nr]
    # Assign edge betweenness centralities
    G = invert_weights(Gs)
    d_ebetw = nx.edge_betweenness_centrality(G, weight='weight')
    nx.set_edge_attributes(G, d_ebetw, "edge_betweenness")
    # append
    l_Gbetw_term.append(G)


# # REWIRED
l_Gbetw_rwd = []
for nr in range(Nrealiz):
    Gs = l_Gcons_rewired[nr]
    # Assign edge betweenness centralities
    G = invert_weights(Gs)
    d_ebetw = nx.edge_betweenness_centrality(G, weight='weight')
    nx.set_edge_attributes(G, d_ebetw, "edge_betweenness")
    # append
    l_Gbetw_rwd.append(G)


# %% SAVE 

for n in range(Nrealiz):
   nx.write_gml(l_Gbetw_rndm[n], OUTPUT_PATH + 'l_Gcons_rndm_' + str(n) + '.gml')
   nx.write_gml(l_Gbetw_genesis[n], OUTPUT_PATH + 'l_Gcons_genesis_' + str(n) + '.gml')
   nx.write_gml(l_Gbetw_term[n], OUTPUT_PATH +  'l_Gcons_term_' + str(n) + '.gml')
   nx.write_gml(l_Gbetw_rwd[n], OUTPUT_PATH +  'l_Gcons_rewired_' + str(n) + '.gml')

