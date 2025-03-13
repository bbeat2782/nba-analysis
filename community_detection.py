import json
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import community as community_louvain
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# Mapping of NBA Team IDs to Team Names
nbaTeamIdToName = {
    1610612737: "Hawks", 1610612738: "Celtics", 1610612751: "Nets",
    1610612766: "Hornets", 1610612741: "Bulls", 1610612739: "Cavaliers",
    1610612742: "Mavericks", 1610612743: "Nuggets", 1610612765: "Pistons",
    1610612744: "Warriors", 1610612745: "Rockets", 1610612754: "Pacers",
    1610612746: "Clippers", 1610612747: "Lakers", 1610612763: "Grizzlies",
    1610612748: "Heat", 1610612749: "Bucks", 1610612750: "Timberwolves",
    1610612740: "Pelicans", 1610612752: "Knicks", 1610612760: "Thunder",
    1610612753: "Magic", 1610612755: "76ers", 1610612756: "Suns",
    1610612757: "Trail Blazers", 1610612758: "Kings", 1610612759: "Spurs",
    1610612761: "Raptors", 1610612762: "Jazz", 1610612764: "Wizards"
}

def load_data(filename="scripts/cleaned_graph_data.json"):
    """Load and process JSON data."""
    with open(filename, "r", encoding="utf-8") as file:
        return json.load(file)

def filter_data(data, start_season, end_season, selected_team_id="ALL"):
    """Filter players and passes for selected seasons and teams."""
    filtered_nodes = {
        node["id"]: node for node in data["nodes"]
        if start_season <= int(node["season"]) <= end_season and int(node["group"]) in nbaTeamIdToName and
        (selected_team_id == "ALL" or int(node["group"]) == selected_team_id)
    }
    
    for node in filtered_nodes.values():
        node["teamName"] = nbaTeamIdToName[int(node["group"])]

    filtered_links = [
        (link["source"], link["target"], link["value"]) for link in data["links"]
        if link["source"] in filtered_nodes and link["target"] in filtered_nodes
    ]

    return filtered_nodes, filtered_links

def apply_louvain(G):
    """Apply Louvain community detection."""
    undirected_G = G.to_undirected()
    return community_louvain.best_partition(undirected_G)

def plot_network(G, partition, degree_type="out-degree"):
    """Plot the passing network and return a mapping of community labels for the heatmap."""
    num_communities = len(set(partition.values()))
    # Use the "tab20" colormap for distinct colors
    cmap = plt.cm.get_cmap("tab20", num_communities)
    community_colors = {comm: cmap(i) for i, comm in enumerate(sorted(set(partition.values())))}

    # Compute node sizes based on degree type (still using degree for visual sizing)
    if degree_type == "in-degree":
        node_sizes = {node: G.in_degree(node, weight="weight") for node in G.nodes()}
    elif degree_type == "out-degree":
        node_sizes = {node: G.out_degree(node, weight="weight") for node in G.nodes()}
    else:
        raise ValueError("Invalid degree_type. Choose 'in-degree' or 'out-degree'.")

    # Normalize node sizes for plotting
    min_size = 10
    max_size = 150
    scaled_sizes = {node: min_size + (max_size - min_size) * (size / max(node_sizes.values(), default=1))
                    for node, size in node_sizes.items()}

    pos = nx.spring_layout(G, k=0.5, seed=42)
    plt.figure(figsize=(16, 12))
    nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color="gray", arrows=False)
    nx.draw_networkx_nodes(G, pos, 
                           node_size=[scaled_sizes[n] for n in G.nodes()],
                           node_color=[community_colors[partition[n]] for n in G.nodes()], 
                           alpha=0.9, edgecolors="black")

    # Compute centrality scores using weighted PageRank
    centrality = nx.pagerank(G, weight="weight")

    # Get top players per community based on centrality instead of degree
    top_players_per_community = {}
    for community in set(partition.values()):
        community_nodes = [n for n in G.nodes if partition[n] == community]
        top_players = sorted(community_nodes, key=lambda n: centrality[n], reverse=True)[:3]
        top_players_per_community[community] = top_players

    # Sort communities by key for consistent ordering
    sorted_communities = sorted(top_players_per_community.keys())

    # Build the legend with community info using the top central players
    legend_labels = {}
    legend_colors = []
    for community in sorted_communities:
        top_player_names = ", ".join([G.nodes[n]["player"] for n in top_players_per_community[community]])
        legend_labels[community] = top_player_names
        legend_colors.append(community_colors[community])

    legend_patches = [Patch(color=legend_colors[i], label=legend_labels[sorted_communities[i]])
                      for i in range(len(legend_labels))]
    plt.legend(handles=legend_patches, loc="best", fontsize=8, title="Top 3 Central Players per Community", frameon=True)

    plt.title(f"2024-2025 All NBA Communities")
    plt.savefig("figures/2024_allnba_communities.png", dpi=300)
    plt.show()

    return legend_labels  # Return mapping for heatmap function

def plot_team_vs_community_heatmap(team_community_matrix, community_label_mapping):
    """Plot the NBA team vs. Louvain community heatmap with correct labels."""
    # Order the columns (NBA teams) alphabetically
    team_community_matrix = team_community_matrix.reindex(sorted(team_community_matrix.columns), axis=1)
    
    plt.figure(figsize=(20, 10))
    ax = sns.heatmap(team_community_matrix, annot=True, fmt="d", cmap="Blues", linewidths=0.5)

    # Update x-axis with full NBA team names
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)

    # Update y-axis with the community labels from the network legend
    ordered_communities = list(team_community_matrix.index)
    community_labels = [community_label_mapping.get(c, f"Community {c}") for c in ordered_communities]
    ax.set_yticklabels(community_labels, rotation=0, fontsize=10)
    
    plt.xlabel("NBA Teams", fontsize=12)
    plt.ylabel("Top 3 Central Players per Community", fontsize=12)
    plt.title(f"2024-2025 All-NBA Communities vs Actual Rosters", fontsize=14)

    plt.tight_layout()
    plt.savefig("figures/2024_allnba_community_overlap.png", dpi=300)
    plt.show()



def plot_community_table(G, partition):
    """
    Create a table visualization that shows:
      - Community ID
      - Top 3 central players (by weighted PageRank)
      - Size (number of nodes) of each community
      - Mean and standard deviation for both in-degree and out-degree
      - Weighted density of the community (for the directed graph)
    """
    # Compute weighted degree measures for each node
    in_degrees = {node: G.in_degree(node, weight="weight") for node in G.nodes()}
    out_degrees = {node: G.out_degree(node, weight="weight") for node in G.nodes()}

    # Compute weighted PageRank centrality
    centrality = nx.pagerank(G, weight="weight")
    
    # Gather data for each community
    communities = sorted(set(partition.values()))
    rows = []
    for comm in communities:
        # Nodes belonging to this community
        comm_nodes = [n for n in G.nodes if partition[n] == comm]
        size = len(comm_nodes)
        
        # Compute in-degree stats for community
        comm_in_degs = [in_degrees[n] for n in comm_nodes]
        mean_in = np.mean(comm_in_degs) if comm_in_degs else 0
        std_in = np.std(comm_in_degs) if comm_in_degs else 0
        
        # Compute out-degree stats for community
        comm_out_degs = [out_degrees[n] for n in comm_nodes]
        mean_out = np.mean(comm_out_degs) if comm_out_degs else 0
        std_out = np.std(comm_out_degs) if comm_out_degs else 0
        
        # Compute weighted density for directed graph:
        # Weighted density = (sum of weights) / (n*(n-1)) for n > 1, else 0.
        subG_directed = G.subgraph(comm_nodes)
        total_weight = sum(data.get("weight", 1) for u, v, data in subG_directed.edges(data=True))
        weighted_density = total_weight / (size * (size - 1)) if size > 1 else 0
        
        # Get top 3 players by weighted PageRank centrality
        top_players = sorted(comm_nodes, key=lambda n: centrality[n], reverse=True)[:3]
        top_players_names = ", ".join([G.nodes[n]["player"] for n in top_players])
        
        rows.append([
            comm,
            top_players_names,
            size,
            round(mean_out, 2),
            round(std_out, 2),
            round(weighted_density, 3)
        ])

    # Create a DataFrame for the community statistics
    df = pd.DataFrame(rows, columns=[
        "Community", "Top 3 Players by PageRank", "Size",
        "Mean Out-Degree", "Std Dev Out-Degree",
        "Weighted Density"
    ])

    # Create a table figure using matplotlib
    fig, ax = plt.subplots(figsize=(10, len(df)*0.6 + 2))
    ax.axis('off')  # Hide default axes

    # Prepare table data (with headers)
    table_data = [df.columns.to_list()] + df.values.tolist()

    table = ax.table(
        cellText=table_data,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df.columns))))

    plt.title(f"Community Statistics", pad=20)
    plt.savefig("figures/2024_allnba_community_table.png", dpi=300, bbox_inches='tight')
    plt.show()


# Main execution
data = load_data()
start_season, end_season = 2024, 2025
selected_team_id = "ALL"  # ALL for all teams, or use ID from nbaTeamIdToName
degree_type = "in-degree"

filtered_nodes, filtered_links = filter_data(data, start_season, end_season, selected_team_id)

# Create graph
G = nx.DiGraph()
for node_id, node_data in filtered_nodes.items():
    G.add_node(node_id, player=node_data["playerName"], team=node_data["teamName"])

for source, target, value in filtered_links:
    G.add_edge(source, target, weight=value)

# Apply Louvain community detection
partition = apply_louvain(G)

# Plot network and retrieve community label mapping for heatmap
community_label_mapping = plot_network(G, partition, degree_type=degree_type)

if selected_team_id == "ALL":
    # Create Community vs. Team Overlap Matrix
    team_community_matrix = pd.DataFrame(
        0, 
        index=list(set(partition.values())), 
        columns=list(set(nbaTeamIdToName.values()))
    )

    for node, community in partition.items():
        team_name = G.nodes[node]["team"]
        team_community_matrix.loc[community, team_name] += 1

    # Plot the heatmap using the community label mapping
    plot_team_vs_community_heatmap(team_community_matrix, community_label_mapping)


plot_community_table(G, partition)
