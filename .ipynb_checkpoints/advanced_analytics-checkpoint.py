import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv
from torch.utils.data import Dataset, DataLoader

# Load dataset
file_path = "data/nba_player_interactions_2001_2025.csv"
df = pd.read_csv(file_path)
df = df[df['EVENTMSGTYPE'] == 1]

def get_team_info(game_df, id):
    if np.isnan(id):
        return '', ''
    
    team_info = game_df[game_df['PLAYER1_TEAM_ID'] == id][['PLAYER1_TEAM_NICKNAME', 'PLAYER1_TEAM_ABBREVIATION']]
    return team_info.iloc[0]['PLAYER1_TEAM_NICKNAME'], team_info.iloc[0]['PLAYER1_TEAM_ABBREVIATION']

def get_game_winner(game_df):
    game_df = game_df.sort_values(by='EVENTNUM')
    left_team, right_team = -1, -1
    left_team_name, left_team_abbre, right_team_name, right_team_abbre = '', '', '', ''
    team_ids = set(game_df['PLAYER1_TEAM_ID'].values)

    idx = 0
    while True:
        if game_df.iloc[idx]['SCOREMARGIN'] == 'TIE':
            idx += 1
        else:
            break
    first_score = game_df.iloc[idx]['SCORE']
    first_score_parts = first_score.split('-')
    first_score_left = int(first_score_parts[0].strip())
    first_score_right = int(first_score_parts[1].strip())

    if first_score_left > first_score_right:
        left_team = game_df.iloc[idx]['PLAYER1_TEAM_ID']
        left_team_name, left_team_abbre = get_team_info(game_df, left_team)
        right_team = list(team_ids - set([left_team]))[0]
        right_team_name, right_team_abbre = get_team_info(game_df, right_team)
    else:
        right_team = game_df.iloc[idx]['PLAYER1_TEAM_ID']
        right_team_name, right_team_abbre = get_team_info(game_df, right_team)        
        left_team = list(team_ids - set([right_team]))[0]
        left_team_name, left_team_abbre = get_team_info(game_df, left_team)
    
    final_score = game_df.iloc[-1]['SCORE']
    final_score_parts = final_score.split('-')
    final_score_left = int(final_score_parts[0].strip())
    final_score_right = int(final_score_parts[1].strip())
    
    if final_score_left > final_score_right:
        result = {
            'tie': False,
            'winner_team_id': left_team,
            'winner_team_name': left_team_name,
            'winner_team_abbre': left_team_abbre,
            'winner_score': final_score_left,
            'loser_team_id': right_team,
            'loser_team_name': right_team_name,
            'loser_team_abbre': right_team_abbre,
            'loser_score': final_score_right
        }
    elif final_score_left < final_score_right:
        result = {
            'tie': False,
            'winner_team_id': right_team,
            'winner_team_name': right_team_name,
            'winner_team_abbre': right_team_abbre,
            'winner_score': final_score_right,
            'loser_team_id': left_team,
            'loser_team_name': left_team_name,
            'loser_team_abbre': left_team_abbre,
            'loser_score': final_score_left
        }
    else:
        result = {
            'tie': True,
            'winner_team_id': left_team,
            'winner_team_name': left_team_name,
            'winner_team_abbre': left_team_abbre,
            'winner_score': final_score_left,
            'loser_team_id': right_team,
            'loser_team_name': right_team_name,
            'loser_team_abbre': right_team_abbre,
            'loser_score': final_score_right
        }
    return result

def construct_graph_per_game(sub_game_df, normalize=True):
    G = nx.DiGraph()
    for _, row in sub_game_df.iterrows():
        passer = row['PLAYER2_ID']
        scorer = row['PLAYER1_ID']
        if passer == 0:
            continue
        if G.has_edge(passer, scorer):
            G[passer][scorer]['weight'] += 1
        else:
            G.add_edge(passer, scorer, weight=1)

    if normalize:
        # Normalize edge weights so they sum to 1
        total_weight = sum(data['weight'] for _, _, data in G.edges(data=True))
        if total_weight > 0:
            for u, v, data in G.edges(data=True):
                data['weight'] = data['weight'] / total_weight
        assert math.isclose(sum([data['weight'] for _, _, data in G.edges(data=True)]), 1, rel_tol=1e-4)
    return G

# ---- Convert NetworkX graph to PyG Data object with additional features ----
def nx_to_pyg(G):
    # Compute additional centrality measures for node features
    degree_cent = nx.degree_centrality(G)
    betw_cent = nx.betweenness_centrality(G)
    clos_cent = nx.closeness_centrality(G)
    undirected_clust = nx.clustering(G.to_undirected())

    # Update node features with additional centrality measures
    for node in G.nodes():
        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)
        clust = undirected_clust[node]
        d_cent = degree_cent[node]
        b_cent = betw_cent[node]
        c_cent = clos_cent[node]
        # Feature vector: [in_degree, out_degree, clustering, degree centrality, betweenness centrality, closeness centrality]
        G.nodes[node]['x'] = [in_deg, out_deg, clust, d_cent, b_cent, c_cent]
    
    data = from_networkx(G)
    data.x = data.x.clone().detach().float()
    
    # Existing graph-level features
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density = nx.density(G)
    
    # --- Compute additional graph-level features ---
    # 1. Graph Centralization (using undirected degree centrality)
    if num_nodes > 2:
        undirected_deg_cent = nx.degree_centrality(G.to_undirected())
        max_deg = max(undirected_deg_cent.values())
        sum_diff = sum(max_deg - d for d in undirected_deg_cent.values())
        # Maximum possible sum for an undirected graph of n nodes is (n-1)*(n-2)
        max_possible = (num_nodes - 1) * (num_nodes - 2) if num_nodes > 2 else 1
        centralization = sum_diff / max_possible
    else:
        centralization = 0.0
    
    # 2. Modularity
    # Use greedy modularity communities on undirected graph
    try:
        communities = nx.algorithms.community.greedy_modularity_communities(G.to_undirected())
        modularity = nx.algorithms.community.modularity(G.to_undirected(), communities)
    except Exception as e:
        modularity = 0.0  # fallback if modularity cannot be computed
    
    # 3. Assortativity
    try:
        assortativity = nx.degree_assortativity_coefficient(G.to_undirected())
        if np.isnan(assortativity):
            assortativity = 0.0
    except Exception as e:
        assortativity = 0.0

    # Combine all graph-level features into one tensor
    graph_features = torch.tensor([num_nodes, num_edges, density, centralization, modularity, assortativity], dtype=torch.float)
    data.graph_features = graph_features
    
    return data



# ---- Define a GNN Encoder (using a simple 2-layer GCN) ----
class GNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNEncoder, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        # Global mean pooling to get a graph-level embedding
        x = global_mean_pool(x, batch)
        return x

# ---- Siamese Network for Graph-Level Binary Classification (Two-Output Version) ----
class SiameseGNN(nn.Module):
    def __init__(self, encoder, embed_dim, graph_feat_dim, hidden_dim):
        super(SiameseGNN, self).__init__()
        self.encoder = encoder
        # Now, each graph's final representation will be (embed_dim + graph_feat_dim)
        # We combine two such representations (and their absolute difference)
        self.fc1 = nn.Linear((embed_dim + graph_feat_dim) * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)  # Two outputs for two classes
        
    def forward(self, data1, data2):
        x1 = self.encoder(data1.x, data1.edge_index, data1.batch)
        x2 = self.encoder(data2.x, data2.edge_index, data2.batch)
        # Concatenate graph-level features if present
        if hasattr(data1, 'graph_features'):
            x1 = torch.cat([x1, data1.graph_features], dim=1)
        if hasattr(data2, 'graph_features'):
            x2 = torch.cat([x2, data2.graph_features], dim=1)
        diff = torch.abs(x1 - x2)
        out = torch.cat([x1, diff], dim=1)
        out = F.relu(self.fc1(out))
        logits = self.fc2(out)  # Raw scores for each class
        return logits


# ---- Custom Dataset for Paired Team Graphs ----
class TeamGraphPairDataset(Dataset):
    def __init__(self, samples):
        """
        samples: list of tuples (data_team1, data_team2, label)
        """
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

# Custom collate function to batch pairs of graphs, including graph_features
def collate_fn(batch):
    data1_list, data2_list, labels = zip(*batch)
    batch_data1 = Batch.from_data_list(data1_list)
    batch_data2 = Batch.from_data_list(data2_list)
    # Collate graph-level features if present
    if hasattr(data1_list[0], 'graph_features'):
        batch_data1.graph_features = torch.stack([d.graph_features for d in data1_list])
    if hasattr(data2_list[0], 'graph_features'):
        batch_data2.graph_features = torch.stack([d.graph_features for d in data2_list])
    labels = torch.tensor(labels, dtype=torch.float).view(-1, 1)
    return batch_data1, batch_data2, labels

# ================= Create Training Pairs =================
# For now, using season 2001
training_df = df[df['Season'].isin(np.arange(2001,2023))]
valid_df = df[df['Season'] == 2023]
test_df = df[df['Season'] == 2024]

def create_pairs(df, shuffle=False, collate_fn=collate_fn, batch_size=128):
    pairs = []
    unique_games = df['Game_ID'].unique()
    for game_id in unique_games:
        game_df = df[df['Game_ID'] == game_id]
        team_split_result = get_game_winner(game_df)
        # Skip if missing team ids
        if np.isnan(team_split_result['winner_team_id']) or np.isnan(team_split_result['loser_team_id']):
            continue
        team1, team2 = team_split_result['winner_team_id'], team_split_result['loser_team_id']
        
        team1_G = construct_graph_per_game(game_df[game_df['PLAYER1_TEAM_ID'] == team1])
        team2_G = construct_graph_per_game(game_df[game_df['PLAYER1_TEAM_ID'] == team2])
        data1 = nx_to_pyg(team1_G)
        data2 = nx_to_pyg(team2_G)
        
        pairs.append((data1, data2, 1))  # (winner, loser)
        pairs.append((data2, data1, 0))  # reverse sample
    
    dataset = TeamGraphPairDataset(pairs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

    return loader

train_loader = create_pairs(training_df, shuffle=True)
valid_loader = create_pairs(valid_df, shuffle=False)
test_loader = create_pairs(test_df, shuffle=False)

from torch_geometric.nn import SAGEConv, global_mean_pool

# ---- Alternative GNN Encoder using GraphSAGE ----
class GNNEncoderSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNEncoderSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        # Global mean pooling to get a graph-level embedding
        x = global_mean_pool(x, batch)
        return x

# ---- Alternative Siamese Network (SiameseGNN_v2) ----
class SiameseGNN_v2(nn.Module):
    def __init__(self, encoder, embed_dim, graph_feat_dim, hidden_dim):
        super(SiameseGNN_v2, self).__init__()
        self.encoder = encoder
        # Bilinear layer to combine two graph embeddings
        self.bilinear = nn.Bilinear(embed_dim + graph_feat_dim, embed_dim + graph_feat_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 2)  # Two outputs for two classes

    def forward(self, data1, data2):
        # Encode each graph
        x1 = self.encoder(data1.x, data1.edge_index, data1.batch)
        x2 = self.encoder(data2.x, data2.edge_index, data2.batch)
        # Concatenate graph-level features if present
        if hasattr(data1, 'graph_features'):
            x1 = torch.cat([x1, data1.graph_features], dim=1)
        if hasattr(data2, 'graph_features'):
            x2 = torch.cat([x2, data2.graph_features], dim=1)
        # Combine the representations using a bilinear layer
        combined = F.relu(self.bilinear(x1, x2))
        logits = self.fc(combined)
        return logits


# ---- Siamese Network with Attention-based Fusion ----
class SiameseGNN_attention(nn.Module):
    def __init__(self, encoder, embed_dim, graph_feat_dim, hidden_dim):
        """
        encoder: The graph encoder module.
        embed_dim: Dimension output by the encoder.
        graph_feat_dim: Dimension of the graph-level features.
        hidden_dim: Dimension of the hidden layer for classification.
        """
        super(SiameseGNN_attention, self).__init__()
        self.encoder = encoder
        # The total dimension of a single graph embedding after concatenating graph features.
        self.total_dim = embed_dim + graph_feat_dim
        
        # Attention layer to compute a weight from the concatenation of both embeddings.
        # This will output a scalar weight for each sample.
        self.fc_attn = nn.Linear(self.total_dim * 2, 1)
        
        # Final classification layers. Here, we combine a weighted sum of the embeddings
        # and their absolute difference.
        self.fc1 = nn.Linear(self.total_dim + self.total_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)  # Two outputs for two classes

    def forward(self, data1, data2):
        # Encode each graph
        x1 = self.encoder(data1.x, data1.edge_index, data1.batch)  # shape: [batch, embed_dim]
        x2 = self.encoder(data2.x, data2.edge_index, data2.batch)  # shape: [batch, embed_dim]
        
        # Concatenate graph-level features if available
        if hasattr(data1, 'graph_features'):
            x1 = torch.cat([x1, data1.graph_features], dim=1)  # now shape: [batch, total_dim]
        if hasattr(data2, 'graph_features'):
            x2 = torch.cat([x2, data2.graph_features], dim=1)  # now shape: [batch, total_dim]
        
        # Compute attention weight from the concatenated embeddings of both graphs.
        attn_input = torch.cat([x1, x2], dim=1)  # shape: [batch, total_dim*2]
        # Using sigmoid to produce a weight between 0 and 1
        attn_weight = torch.sigmoid(self.fc_attn(attn_input))  # shape: [batch, 1]
        
        # Form a weighted combination of the two embeddings:
        combined = attn_weight * x1 + (1 - attn_weight) * x2  # shape: [batch, total_dim]
        
        # Also compute the absolute difference between x1 and x2
        diff = torch.abs(x1 - x2)  # shape: [batch, total_dim]
        
        # Fusion: concatenate the weighted combination with the difference
        fusion = torch.cat([combined, diff], dim=1)  # shape: [batch, total_dim*2]
        
        out = F.relu(self.fc1(fusion))
        logits = self.fc2(out)
        return logits

# Example Model Setup:
in_channels = 6       # Number of node features
hidden_channels = 128
embed_dim = 64
graph_feat_dim = 6
hidden_dim = 64

# Instantiate the alternative encoder and model
encoder_sage = GNNEncoderSAGE(in_channels, hidden_channels, embed_dim)
model = SiameseGNN_v2(encoder_sage, embed_dim, graph_feat_dim, hidden_dim)

# Move model to device as usual
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

# ---------------- Early Stopping Training Loop ----------------

num_epochs = 50
patience = 5
best_val_loss = float('inf')
epochs_no_improve = 0

for epoch in range(num_epochs):
    # --- Training ---
    model.train()
    total_train_loss = 0
    for batch_data1, batch_data2, labels in train_loader:
        batch_data1 = batch_data1.to(device)
        batch_data2 = batch_data2.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(batch_data1, batch_data2)
        loss = criterion(outputs, labels.squeeze().long())
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_loader)
    
    # --- Validation ---
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch_data1, batch_data2, labels in valid_loader:
            batch_data1 = batch_data1.to(device)
            batch_data2 = batch_data2.to(device)
            labels = labels.to(device)

            outputs = model(batch_data1, batch_data2)
            loss = criterion(outputs, labels.squeeze().long())
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(valid_loader)
    
    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
    
    # Early stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = model.state_dict()  # save best model
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

# Load the best model state after training
model.load_state_dict(best_model_state)

# Make sure the "models" folder exists
os.makedirs("models", exist_ok=True)

# Save the best model state
torch.save(best_model_state, "models/best_model.pth")
print("Best model saved to models/best_model.pth")

def simple_evaluate(model, loader, which_set='Valid'):
    # ---------------- Testing Training Set ----------------
    model.eval()
    all_true_labels = []
    all_predictions = []
    with torch.no_grad():
        for batch_data1, batch_data2, labels in loader:
            batch_data1 = batch_data1.to(device)
            batch_data2 = batch_data2.to(device)
            labels = labels.to(device)
    
            outputs = model(batch_data1, batch_data2)
            preds = outputs.argmax(dim=1)  # choose class with highest logit
            all_true_labels.extend(labels.squeeze().tolist())
            all_predictions.extend(preds.tolist())
    
    correct = sum(1 for true, pred in zip(all_true_labels, all_predictions) if true == pred)
    accuracy = correct / len(all_true_labels)
    print(f"\n===== {which_set} Set Results =====")
    print("Total Samples Evaluated:", len(all_true_labels))
    print("Overall Accuracy: {:.2f}%".format(accuracy * 100))

    return accuracy


train_accuracy = simple_evaluate(model, train_loader, which_set='Train')
valid_accuracy = simple_evaluate(model, valid_loader, which_set='Valid')
test_accuracy = simple_evaluate(model, test_loader, which_set='Test')

def evaluate(model, loader, device=device, which_set='Valid'):
    # ---------------- Testing Test Set (Game-Level Evaluation: Paired Samples) ----------------
    model.eval()
    all_true_labels = []   # will hold pairs like [1, 0] per game
    all_pred_labels = []   # will hold pairs like [1, 0] or [0, 1] per game
    
    with torch.no_grad():
        for batch_data1, batch_data2, labels in loader:
            # Move the batch to the device
            batch_data1 = batch_data1.to(device)
            batch_data2 = batch_data2.to(device)
            labels = labels.to(device)  # labels shape: [batch_size, 1]
            
            outputs = model(batch_data1, batch_data2)  # shape: [batch_size, 2]
            probs = F.softmax(outputs, dim=1)          # shape: [batch_size, 2]
            
            # We expect that for each game, there are 2 samples:
            # The first sample (ordering: (team1, team2)) should have ground truth label 1,
            # and the second sample (ordering: (team2, team1)) should have ground truth label 0.
            batch_size = probs.shape[0]
            # Make sure batch_size is even (each game contributes 2 samples)
            num_games = batch_size // 2
            # Reshape to [num_games, 2, 2]
            probs_reshaped = probs.view(num_games, 2, 2)
            
            # For each game, get the win probability for team1 and team2:
            # For sample 0: probability for class 1 corresponds to team1's win probability.
            # For sample 1: probability for class 1 corresponds to team2's win probability.
            for i in range(num_games):
                p_team1 = probs_reshaped[i, 0, 1].item()
                p_team2 = probs_reshaped[i, 1, 1].item()
                # Determine the predicted pair:
                if p_team1 > p_team2:
                    # Model indicates team1 wins: predicted pair is [1, 0]
                    pred_pair = [1, 0]
                else:
                    # Otherwise, predicted pair is [0, 1]
                    pred_pair = [0, 1]
                all_pred_labels.extend(pred_pair)
                # Since our test pairs were constructed as (winner, loser) and then reversed,
                # we assume that the true ordering for each game is [1, 0].
                all_true_labels.extend([1, 0])
                
    # Now compute sample-level accuracy (the number of correct predictions divided by total predictions)
    correct = sum(1 for true, pred in zip(all_true_labels, all_pred_labels) if true == pred)
    accuracy = correct / len(all_true_labels)
    print(f"\n===== {which_set} Set Results =====")
    print("Total Samples Evaluated (2 per game):", len(all_true_labels))
    print("Overall Sample-Level Accuracy: {:.2f}%".format(accuracy * 100))

    return accuracy


valid_accuracy = evaluate(model, valid_loader, which_set='Valid')
test_accuracy = evaluate(model, test_loader, which_set='Test')