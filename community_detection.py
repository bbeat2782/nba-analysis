import json
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import community as community_louvain
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
import statistics
from collections import Counter


# Mapping of NBA Team IDs to Team Names
nbaTeamIdToName = {
    1610612737: "Atlanta Hawks", 1610612738: "Boston Celtics", 1610612751: "Brooklyn Nets",
    1610612766: "Charlotte Hornets", 1610612741: "Chicago Bulls", 1610612739: "Cleveland Cavaliers",
    1610612742: "Dallas Mavericks", 1610612743: "Denver Nuggets", 1610612765: "Detroit Pistons",
    1610612744: "Golden State Warriors", 1610612745: "Houston Rockets", 1610612754: "Indiana Pacers",
    1610612746: "LA Clippers", 1610612747: "Los Angeles Lakers", 1610612763: "Memphis Grizzlies",
    1610612748: "Miami Heat", 1610612749: "Milwaukee Bucks", 1610612750: "Minnesota Timberwolves",
    1610612740: "New Orleans Pelicans", 1610612752: "New York Knicks", 1610612760: "Oklahoma City Thunder",
    1610612753: "Orlando Magic", 1610612755: "Philadelphia 76ers", 1610612756: "Phoenix Suns",
    1610612757: "Portland Trail Blazers", 1610612758: "Sacramento Kings", 1610612759: "San Antonio Spurs",
    1610612761: "Toronto Raptors", 1610612762: "Utah Jazz", 1610612764: "Washington Wizards"
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

def plot_network(G, partition, title="", degree_type="in-degree"):
    """Plot the passing network and return a mapping of community labels for the heatmap."""
    num_communities = len(set(partition.values()))
    colors = plt.cm.rainbow(np.linspace(0, 1, num_communities))
    community_colors = {comm: colors[i] for i, comm in enumerate(set(partition.values()))}

    # Compute node sizes based on degree type
    if degree_type == "in-degree":
        node_sizes = {node: G.in_degree(node, weight="weight") for node in G.nodes()}
    elif degree_type == "out-degree":
        node_sizes = {node: G.out_degree(node, weight="weight") for node in G.nodes()}
    else:
        raise ValueError("Invalid degree_type. Choose 'in-degree' or 'out-degree'.")

    # Normalize node sizes
    min_size = 10
    max_size = 200
    scaled_sizes = {node: min_size + (max_size - min_size) * (size / max(node_sizes.values(), default=1))
                    for node, size in node_sizes.items()}

    pos = nx.spring_layout(G, k=0.5, seed=42)
    plt.figure(figsize=(16, 12))
    nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color="gray")
    nx.draw_networkx_nodes(G, pos, 
                           node_size=[scaled_sizes[n] for n in G.nodes()],
                           node_color=[community_colors[partition[n]] for n in G.nodes()], 
                           alpha=0.9, edgecolors="black")

    # Get top players per community
    top_players_per_community = {}
    density_per_community = {}

    for community in set(partition.values()):
        # Find top players based on degree
        community_nodes = [n for n in G.nodes if partition[n] == community]
        top_players = sorted(community_nodes, key=lambda n: node_sizes[n], reverse=True)[:3]
        top_players_per_community[community] = top_players

        # Compute density of the whole community
        subG = G.subgraph(community_nodes)
        n, m = len(subG.nodes), len(subG.edges)
        density_per_community[community] = round(m / (n * (n - 1) / 2) if n > 1 else 0, 3)

    # Sort communities by density
    sorted_communities = sorted(density_per_community.keys(), key=lambda c: density_per_community[c], reverse=True)

    # Build the legend with community info
    legend_labels = {}
    legend_colors = []

    for community in sorted_communities:
        density = density_per_community[community]
        top_player_names = ", ".join([G.nodes[n]["player"] for n in top_players_per_community[community]])

        legend_label = f"Density: {density}, Top Players: {top_player_names}"
        legend_labels[community] = legend_label  # Store mapping of community number to label
        legend_colors.append(community_colors[community])

    # Add legend
    legend_patches = [Patch(color=legend_colors[i], label=legend_labels[sorted_communities[i]]) for i in range(len(legend_labels))]
    plt.legend(handles=legend_patches, loc="best", fontsize=8, title="Community Density and Top Passers", frameon=True)

    plt.title(f"{title} Assist Network Communities")
    plt.savefig("figures/all_nba_communities.png", dpi=300)
    plt.show()

    return legend_labels  # Return mapping for heatmap function

def plot_team_vs_community_heatmap(team_community_matrix, community_label_mapping, title=""):
    """Plot the NBA team vs. Louvain community heatmap with correct labels."""
    
    plt.figure(figsize=(20, 10))  # Increase figure size
    
    ax = sns.heatmap(team_community_matrix, annot=True, fmt="d", cmap="Blues", linewidths=0.5)

    # Update x-axis with full NBA team names
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)  

    # Update y-axis with the same community labels from the network legend
    ordered_communities = list(team_community_matrix.index)  # Ensure ordering matches matrix
    community_labels = [community_label_mapping.get(c, f"Community {c}") for c in ordered_communities]
    
    ax.set_yticklabels(community_labels, rotation=0, fontsize=10)  
    
    plt.xlabel("NBA Teams", fontsize=12)
    plt.ylabel("Louvain Communities", fontsize=12)
    plt.title(f"{title} Communities vs Teams (Player Overlap)", fontsize=14)

    plt.tight_layout()  # Prevents label cutoff
    plt.savefig("figures/nba_team_vs_community_overlap.png", dpi=300)
    plt.show()


# def plot_jaccard_similarity(jaccard_matrix, title=""):
#     """Plot heatmap of Jaccard similarity between Louvain communities and NBA teams."""
#     plt.figure(figsize=(14, 8))
#     sns.heatmap(jaccard_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
#     plt.xlabel("NBA Teams")
#     plt.ylabel("Louvain Communities")
#     plt.title(f"{title} Communities vs Teams (Jaccard Similarity)")
#     plt.xticks(rotation=90)
#     plt.savefig("figures/nba_team_vs_community_jaccard.png", dpi=300)
#     plt.show()

# Main execution
data = load_data()
start_season, end_season = 2024, 2024
selected_team_id = "ALL" # ALL for all teams, or use ID from nbaTeamIdToName
degree_type = "out-degree"
plot_title = "2024 All NBA" # rename this to reflect years/teams selected

filtered_nodes, filtered_links = filter_data(data, start_season, end_season, selected_team_id)

# Create graph
G = nx.DiGraph()
for node_id, node_data in filtered_nodes.items():
    G.add_node(node_id, player=node_data["playerName"], team=node_data["teamName"])

for source, target, value in filtered_links:
    G.add_edge(source, target, weight=value)

# Apply Louvain
partition = apply_louvain(G)

# Plot network
community_label_mapping = plot_network(G, partition, degree_type=degree_type, title=plot_title)


if selected_team_id == "ALL":
    # Create **Community vs. Team Overlap Matrix**
    team_community_matrix = pd.DataFrame(0, index=list(set(partition.values())), columns=list(set(nbaTeamIdToName.values())))


    for node, community in partition.items():
        team_name = G.nodes[node]["team"]
        team_community_matrix.loc[community, team_name] += 1

    # Pass the mapping to the heatmap function
    plot_team_vs_community_heatmap(team_community_matrix, community_label_mapping, title=plot_title)




    # # Compute Jaccard similarity
    # team_community_jaccard = pd.DataFrame(0.0, index=list(set(partition.values())), columns=list(set(nbaTeamIdToName.values())))

    # for community in team_community_jaccard.index:
    #     community_players = {node for node, c in partition.items() if c == community}
        
    #     for team in team_community_jaccard.columns:
    #         team_players = {node for node in G.nodes if G.nodes[node]["team"] == team}
    #         intersection = len(community_players & team_players)
    #         union = len(community_players | team_players)
    #         jaccard_score = intersection / union if union > 0 else 0
    #         team_community_jaccard.loc[community, team] = jaccard_score

    # plot_jaccard_similarity(team_community_jaccard, title=plot_title)











