import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from fa2 import ForceAtlas2

# Load the collected NBA player interaction data
file_path = "nba_player_interactions_2001_2024.csv"  # Update if needed
df = pd.read_csv(file_path)

seasons = range(2001, 2025)  # Seasons from 2001 to 2024
fig, axes = plt.subplots(4, 6, figsize=(20, 15))  # Create subplots for each season
axes = axes.flatten()  # Flatten for easy indexing

for i, season in enumerate(seasons):
    df_season = df[df['Season'] == season]
    # Filter only assist interactions
    df_assists = df_season[df_season["Event_Type"] == "Shot_Made"].dropna(subset=["Player2_ID"])
    df_assists = df_assists[df_assists['Player2_ID'] != 0]

    # Create a directed graph
    G = nx.DiGraph()

    # Add edges (Player2_ID = assister, Player1_ID = scorer)
    for _, row in df_assists.iterrows():
        assister = row["Player2_ID"]
        scorer = row["Player1_ID"]
        if G.has_edge(assister, scorer):
            G[assister][scorer]["weight"] += 1
        else:
            G.add_edge(assister, scorer, weight=1)

    if len(G.nodes) == 0:  # Skip empty seasons
        continue

    # Calculate node sizes and colors based on scoring frequency
    player_shots = df_assists["Player1_ID"].value_counts()
    node_sizes = {node: player_shots.get(node, 10) * 0.3 for node in G.nodes()}
    node_colors = {node: player_shots.get(node, 0) for node in G.nodes()}  # Shots made

    # Compute FA2 Layout
    forceatlas2 = ForceAtlas2(
        outboundAttractionDistribution=True,
        edgeWeightInfluence=2.0,
        jitterTolerance=1.0,
        barnesHutOptimize=True,
        barnesHutTheta=1.2,
        scalingRatio=1.2,
        strongGravityMode=False,
        gravity=0.5,  # Increased to pull nodes together
        verbose=False  # Disable print logs
    )

    positions_array = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=2000)
    pos = nx.rescale_layout_dict(positions_array, scale=4)  # Keeps nodes compact

    # Clip outliers (Move extreme nodes closer to center)
    max_range = 0.5  # Set a reasonable boundary
    for node, (x, y) in pos.items():
        pos[node] = (max(min(x, max_range), -max_range), max(min(y, max_range), -max_range))  

    # Prepare node colors (colormap)
    node_color_vals = list(node_colors.values())
    cmap_nodes = plt.cm.viridis  # Choose colormap
    node_color_map = [cmap_nodes(val / max(node_color_vals) if max(node_color_vals) > 0 else 0) for val in node_color_vals]

    # Prepare edge widths (based on assist count)
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [(w / max_weight) * 5 for w in edge_weights]

    ax = axes[i]
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
                           arrowstyle="-",  
                           connectionstyle="arc3,rad=0.1", ax=ax)

    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.axis("off")

# Adjust layout
plt.tight_layout()
plt.show()
