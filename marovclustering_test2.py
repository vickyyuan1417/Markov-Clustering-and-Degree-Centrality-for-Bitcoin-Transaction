# !pip install markov_clustering
import random
import numpy as np
import markov_clustering as mc
import networkx as nx
import matplotlib.pyplot as plt
import json
import csv
import scipy.sparse as sp

# Read JSON data from file
with open('bitcoin_direct.test.json') as f:
    data = json.load(f)

# Create a directed graph
G = nx.DiGraph()

for entry in data:
    input_id = entry["_id"]["$oid"]
    output_id = entry["txid"]
    # This is for node creation
    G.add_node(input_id)
    G.add_node(output_id)
    # Accessing the nested "value" field
    value = entry["value"]
    if isinstance(value, dict):
        # Extract the numeric value from the dictionary
        value = float(value["$numberLong"])
    else:
        # Use the numeric value directly
        value = float(value)
    # This is for edge creation
    G.add_edge(input_id, output_id, weight=value)

# Define positions using the spring layout algorithm
positions = nx.spring_layout(G)

# Convert NetworkX graph to SciPy sparse matrix
matrix = sp.csr_matrix(nx.adjacency_matrix(G))

# =========================================================================================================
# Markov Clustering Part
result = mc.run_mcl(matrix, inflation = 1.01)
clusters = mc.get_clusters(result)
# print("Clusters:")
# print(clusters)

# =========================================================================================================
# Degree Centrality Part
# Initialize a dictionary to store the central node in each cluster
central_nodes = {}
cluster_colors = {}
for i, cluster in enumerate(clusters):
    cluster_color = (random.random(), random.random(), random.random())
    for node in cluster:
        cluster_colors[node] = cluster_color

for cluster in clusters:
    if len(cluster) > 0:
        cluster_subgraph = G.subgraph(cluster)
        betweenness_centrality = nx.betweenness_centrality(cluster_subgraph)
        if betweenness_centrality:
            central_node = max(cluster_subgraph.nodes(), key=lambda x: betweenness_centrality.get(x, 0))
            central_nodes[central_node] = positions[central_node]

# print("Central Nodes:")
# print(central_nodes)

# this is how I print graph for your reference
# plt.figure(figsize=(10, 8))
# for cluster in clusters:
#     cluster_nodes = G.subgraph(cluster)
#     nx.draw(cluster_nodes, pos=positions, node_size=50, node_color=[cluster_colors[node] for node in cluster], with_labels=False, edge_color="silver")
# nx.draw_networkx_nodes(G, positions, nodelist=central_nodes.keys(), node_size=100, node_color='red')
# plt.show()

# =========================================================================================================
# Define a function to save cluster information to a CSV file
def save_clusters_to_csv(clusters, central_nodes, cluster_colors, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Node', 'Cluster', 'Central'])

        for i, cluster in enumerate(clusters):
            for node in cluster:
                is_central = 1 if node in central_nodes else 0
                writer.writerow([node, i, is_central])

# Save cluster information to a CSV file
save_clusters_to_csv(clusters, central_nodes, cluster_colors, 'cluster_info.csv')

# ===========================================================================================================
# Create a new graph to represent the overview
overview_graph = nx.Graph()

# Add all nodes and edges from the original graph
overview_graph.add_nodes_from(G.nodes())
overview_graph.add_edges_from(G.edges())

# Connect central nodes of each cluster to form the overview
for central_node, cluster in central_nodes.items():
    for node in cluster:
        
        overview_graph.add_edge(central_node, node)

# Save the overview graph to a file
nx.write_graphml(overview_graph, 'overview_graph.graphml')

# ===========================================================================================================
# Draw the overview graph
plt.figure(figsize=(10, 8))
node_colors = [cluster_colors[node] if node in cluster_colors else 'gray' for node in overview_graph.nodes()]
node_colors_central = ['red' if node in central_nodes else color for node, color in zip(overview_graph.nodes(), node_colors)]
nx.draw(overview_graph, pos=positions, node_size=50, node_color=node_colors_central, with_labels=False, edge_color="silver")

# Save the graph as an image file (JPG format)
plt.savefig('overview_graph.jpg')

# # Display the graph
# plt.show()

# Cases notification:
# If in-degree values are 0 for a clusters, which suggests that there are no edges directed towards the nodes in the cluster.
# If crashes due to running time or cache, run steps one by one, for example, run till [# Save cluster information to a CSV file] instead of generating the graph.