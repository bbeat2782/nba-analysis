import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import powerlaw

from structural_analysis_util import plot_degree_pdf, degree_distribution, print_top_players

# Load dataset
file_path = "data/nba_player_interactions_2001_2025.csv"
df = pd.read_csv(file_path)

# Only keep nba teams
nba_team_ids = [1610612737, 1610612738, 1610612751, 1610612766, 1610612741, 1610612739,
        1610612742, 1610612743, 1610612765, 1610612744, 1610612745, 1610612754,
        1610612746, 1610612747, 1610612763, 1610612748, 1610612749, 1610612750,
        1610612740, 1610612752, 1610612760, 1610612753, 1610612755, 1610612756,
        1610612757, 1610612758, 1610612759, 1610612761, 1610612762, 1610612764]
df = df[df['PLAYER1_TEAM_ID'].isin(nba_team_ids)]

# NOTE Need to choose how we will handle this
df = df[(df['Season'] == 2023)]

players_df = pd.concat([
    df[['PLAYER1_ID', 'PLAYER1_NAME']].rename(columns={'PLAYER1_ID': 'PLAYER_ID', 'PLAYER1_NAME': 'PLAYER_NAME'}),
    df[['PLAYER2_ID', 'PLAYER2_NAME']].rename(columns={'PLAYER2_ID': 'PLAYER_ID', 'PLAYER2_NAME': 'PLAYER_NAME'})
]).drop_duplicates()
# Create a mapping from PLAYER_ID to PLAYER_NAME
player_id_to_name = dict(zip(players_df['PLAYER_ID'], players_df['PLAYER_NAME']))

# Create directed graph
G = nx.DiGraph()

# Add weighted edges (Passer -> Scorer)
for _, row in df.iterrows():
    passer = row["PLAYER2_ID"]
    scorer = row["PLAYER1_ID"]
    if passer == 0:
        continue

    if G.has_edge(passer, scorer):
        G[passer][scorer]["weight"] += 1
    else:
        G.add_edge(passer, scorer, weight=1)

# ------------------------ Basic Statistics ------------------------
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
# If we ignore the direction
num_weak_components = nx.number_weakly_connected_components(G)
# Connected components following the direction
num_strong_components = nx.number_strongly_connected_components(G)

print(f"Number of Nodes: {num_nodes}")
print(f"Number of Edges: {num_edges}")
print(f"Number of Weakly Connected Components: {num_weak_components}")
print(f"Number of Strongly Connected Components: {num_strong_components}")


# ------------------------ Degree Distribution ------------------------
in_degrees = np.array(list(dict(G.in_degree()).values()))
out_degrees = np.array(list(dict(G.out_degree()).values()))

# Generate a Random Graph using Erdős-Rényi (ER) Model
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
p = num_edges / (num_nodes * (num_nodes - 1))  # Connection probability
random_G = nx.erdos_renyi_graph(num_nodes, p, directed=True)

# Compute in-degree and out-degree for random graph
random_in_degrees = np.array(list(dict(random_G.in_degree()).values()))
random_out_degrees = np.array(list(dict(random_G.out_degree()).values()))

# Compute degree distributions
k_in, Pk_in = degree_distribution(in_degrees)
k_out, Pk_out = degree_distribution(out_degrees)
k_in_rand, Pk_in_rand = degree_distribution(random_in_degrees)
k_out_rand, Pk_out_rand = degree_distribution(random_out_degrees)

# Log-Log Plot: P(k) vs. k
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot In-Degree Distribution
plot_degree_pdf(k_in, Pk_in, k_in_rand, Pk_in_rand, "In-Degree Distribution (Log-Log)", "blue", axes[0])

# Plot Out-Degree Distribution
plot_degree_pdf(k_out, Pk_out, k_out_rand, Pk_out_rand, "Out-Degree Distribution (Log-Log)", "red", axes[1])

# Adjust layout and save figure
plt.tight_layout()
plt.savefig("figures/Pk_vs_k_LogLog.png")

# ----------- Expected Maximum Degree -----------
k_avg_in = np.mean(in_degrees)
k_avg_out = np.mean(out_degrees)
k_avg_total = (k_avg_in + k_avg_out) / 2  # Should be same as either

# Compute standard deviation
k_std_in = np.std(in_degrees)
k_std_out = np.std(out_degrees)
num_nodes = G.number_of_nodes()

# Compute expected maximum degree separately
expected_k_max_in = k_avg_in + k_std_in * np.sqrt(2 * np.log(num_nodes))
expected_k_max_out = k_avg_out + k_std_out * np.sqrt(2 * np.log(num_nodes))

# ----------- Network Stage -----------
ln_N = np.log(num_nodes)

if k_avg_total < 1:
    network_stage = "Subcritical (Fragmented Network)"
elif k_avg_total > 1 and k_avg_total < ln_N:
    network_stage = "Supercritical (Giant Component Likely)"
else:
    network_stage = "Highly Connected (Almost Fully Connected)"

# Print results
print(f"Average In-Degree ⟨k_in⟩: {k_avg_in:.2f}")
print(f"Average Out-Degree ⟨k_out⟩: {k_avg_out:.2f}")
print(f"Expected Maximum In-Degree k_max_in: {expected_k_max_in:.2f}")
print(f"Expected Maximum Out-Degree k_max_out: {expected_k_max_out:.2f}")
print(f"Natural Log of N (ln(N)): {ln_N:.2f}")
print(f"Network Stage Classification: {network_stage}")

# ----------- Power Law -----------
# Convert degree data into numpy arrays
in_degrees = np.array(list(dict(G.in_degree()).values()))
out_degrees = np.array(list(dict(G.out_degree()).values()))

# Fit power-law distributions
fit_in = powerlaw.Fit(in_degrees, discrete=True)
fit_out = powerlaw.Fit(out_degrees, discrete=True)

# Extract exponents (gamma values)
gamma_in = fit_in.power_law.alpha
gamma_out = fit_out.power_law.alpha

# Compare with alternative distributions
R_in, p_in = fit_in.distribution_compare('power_law', 'exponential')
R_out, p_out = fit_out.distribution_compare('power_law', 'exponential')

# Print results
print(f"Power-law exponent (γ) for In-Degree: {gamma_in:.2f}")
print(f"Power-law exponent (γ) for Out-Degree: {gamma_out:.2f}")

# ------------------------ Degree Correlation ------------------------
# Compute average neighbor degree for each node
knn_dict = nx.average_neighbor_degree(G, source='out', weight='weight')

# Extract degrees and corresponding k_nn values
degrees = np.array(list(dict(G.out_degree()).values()))
knn_values = np.array([knn_dict.get(node, 0) for node in G.nodes()])

# Aggregate by degree
unique_degrees, avg_knn = [], []
for k in np.unique(degrees):
    indices = np.where(degrees == k)[0]
    if len(indices) > 0:
        unique_degrees.append(k)
        avg_knn.append(np.mean(knn_values[indices]))

# Plot k_nn(k) vs. k
plt.figure(figsize=(6, 4))
plt.scatter(unique_degrees, avg_knn, color='blue', alpha=0.7)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Degree k (Passes Made)")
plt.ylabel("Average Neighbor Degree k_nn(k)")
plt.title("Degree Correlation: k_nn(k) vs. k")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig("figures/DegreeCorrelation.png")

# Compute assortativity coefficient (Pearson correlation of degrees at each edge)
assortativity = nx.degree_assortativity_coefficient(G, weight='weight')
print(f"Degree Assortativity Coefficient: {assortativity:.4f}")


# ------------------------ Clustering Coefficient ------------------------

# Generate a random directed graph with same nodes & edges
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
p = num_edges / (num_nodes * (num_nodes - 1))  # Connection probability

random_G = nx.erdos_renyi_graph(num_nodes, p, directed=True)

# Compute global clustering for NBA and random graph
global_C_nba = nx.transitivity(G)
global_C_random = nx.transitivity(random_G)

print(f"Global Clustering (NBA Network): {global_C_nba:.4f}")
print(f"Global Clustering (Random Network): {global_C_random:.4f}")

# ------------------------ Centrality Measures ------------------------

# --- 1. Degree Centrality ---
in_degree_centrality = nx.in_degree_centrality(G)
out_degree_centrality = nx.out_degree_centrality(G)

# --- 2. Closeness & Harmonic Centrality ---
closeness_centrality = nx.closeness_centrality(G)
harmonic_centrality = nx.harmonic_centrality(G)

# --- 3. Eigenvector Centrality & PageRank ---
eigenvector_centrality = nx.eigenvector_centrality_numpy(G, weight="weight")
pagerank = nx.pagerank(G, weight="weight")

# --- 4. Hubs & Authorities (HITS Algorithm) ---
hubs, authorities = nx.hits(G, max_iter=1000, normalized=True)  # Finds important playmakers & scorers

print_top_players(out_degree_centrality, "Out-Degree Centrality (Playmakers)", player_id_to_name)
print_top_players(in_degree_centrality, "In-Degree Centrality (Scorers)", player_id_to_name)
print_top_players(closeness_centrality, "Closeness Centrality", player_id_to_name)
print_top_players(harmonic_centrality, "Harmonic Centrality", player_id_to_name)
print_top_players(eigenvector_centrality, "Eigenvector Centrality", player_id_to_name)
print_top_players(pagerank, "PageRank", player_id_to_name)
print_top_players(hubs, "Hubs (Top Passers)", player_id_to_name)
print_top_players(authorities, "Authorities (Top Receivers)", player_id_to_name)
