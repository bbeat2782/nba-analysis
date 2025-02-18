import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# TODO: need to scrape team for each player

# Load the collected NBA player interaction data
file_path = "data/nba_player_interactions_2001_2025.csv"
df = pd.read_csv(file_path)

season = 2024

df_season = df[df['Season'] == season]
# Filter only assist interactions
df_assists = df_season[df_season["EVENTMSGTYPE"] == 1].dropna(subset=["PLAYER2_ID"])
df_assists = df_assists[df_assists['PLAYER2_ID'] != 0]

# Create a directed graph
G = nx.DiGraph()

# Add edges (Player2_ID = assister, Player1_ID = scorer)
for _, row in df_assists.iterrows():
    assister = row["PLAYER2_ID"]
    scorer = row["PLAYER1_ID"]
    if G.has_edge(assister, scorer):
        G[assister][scorer]["weight"] += 1
    else:
        G.add_edge(assister, scorer, weight=1)

# Calculate node sizes and colors based on scoring frequency
player_shots = df_assists["PLAYER1_ID"].value_counts()
node_sizes = {node: player_shots.get(node, 10) * 0.3 for node in G.nodes()}
node_colors = {node: player_shots.get(node, 0) for node in G.nodes()}  # Shots made

# 🔥 Use Kamada-Kawai Layout Instead of ForceAtlas2
pos = nx.kamada_kawai_layout(G, weight="weight", scale=5)  # Uses edge weights for optimal placement
#pos = nx.spring_layout(G, k=1.2, iterations=100)  # k controls node spacing
#pos = nx.forceatlas2_layout(G)

# Prepare node colors (colormap)
node_color_vals = list(node_colors.values())
cmap_nodes = plt.cm.viridis  # Choose colormap
node_color_map = [cmap_nodes(val / max(node_color_vals) if max(node_color_vals) > 0 else 0) for val in node_color_vals]

# Prepare edge widths (based on assist count)
edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
max_weight = max(edge_weights) if edge_weights else 1
edge_widths = [(w / max_weight) * 5 for w in edge_weights]

# Plot with Kamada-Kawai Layout
fig, ax = plt.subplots(figsize=(20, 20))
ax.set_title(f"NBA Assist Network - {season}", fontsize=12)

# Draw nodes
nx.draw_networkx_nodes(G, pos, 
                        node_size=[node_sizes.get(node, 30) for node in G.nodes()],
                        node_color=node_color_map, edgecolors="black", ax=ax)

# Draw edges with uniform gray color but variable width
nx.draw_networkx_edges(G, pos, 
                        edge_color="gray", 
                        width=edge_widths, 
                        alpha=0.8, 
                        #arrowstyle="-",  
                        connectionstyle="arc3,rad=0.1", ax=ax)

ax.axis("off")
plt.savefig(f'figures/{season}_assist_network.png')
