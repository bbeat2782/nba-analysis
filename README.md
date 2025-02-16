# NBA Player Interaction Network Analysis

## Overview

This project constructs an evolving **NBA player interaction network** using event-based data from the **NBA API**. Players are connected based on their in-game interactions (**passes, assists**) over multiple seasons. By modeling this as a graph and analyzing its evolution over time, we aim to **understand long-term player impact, team chemistry, and the dynamics of NBA gameplay across different eras**.

## Things to choose

- year by year or combine the years?
- how to weigh the interactions

## Components of the Project

### 1. Project Goal / Hypothesis

- **Hypothesis:** 
  - The passing network structure can reveal key trends in NBA gameplay, including the evolution of playmaking roles, team chemistry, and long-term player impact.
- **Objectives:**
  - Analyze **how player interactions evolve** during and over multiple seasons.
  - Identify **key playmakers** based on their network influence.
  - Measure **team chemistry** and its relation to team performance.
    - (TODO: how?)
  - Compare **historical playstyles** by analyzing network structures across NBA eras.
    - Can look into centrality or hubs and authorities

---

### 2. Data Acquisition

- **Source:** [NBA API](https://github.com/swar/nba_api)
- **Nodes:** NBA players
- **Edges:** Passes that led to scoring
<!-- - **Temporal aspect:** Multi-season evolving network (layered by year) -->

---

### 3. Network Representation

- **Graph Type:** 
  - **Directed** (captures passing direction)
  - **Weighted** (interaction frequency)
  <!-- - **Multilayer** (each season as a separate layer)
    - (Need to take a look at multilayer graph) -->
  - **Bipartite Graphs** (for hubs & authorities analysis)

- **Network Construction:**
  - Nodes: Individual players
  - Edges: Weighted by interaction frequency

- **Temporal Representation:**
  - Investigate **rolling time windows** vs. **per-season snapshots** to track evolving connections.

---

### 4. Structural Analysis

#### 4.1 Fundamental Analysis
- **Basic Statistics**: 
  - Number of nodes (players) and edges (interactions) (maybe by season?)
  - Number of connected components
  - Visualization of weighted adjaceny matrix
    - TODO need to define how to set weights

- **Degree Distribution**:
  - Separate analysis for **in-degree (received passes)** and **out-degree (passes made)**
  - Identify hubs (high-degree nodes) and compare with a random graph using a **log-log plot**
  - Compute **expected maximum degree**
  - Compute **average degree** and classify network stage (subcritical/supercritical) using **<k> > 1 or >= ln(N)**
  - Fit degree distribution to **power law** and check slope for scale-free properties

- **Degree Correlation**:
  - Compute and plot **k_nn vs. k** to analyze assortativity (team-based vs. star-based passing tendencies)

- **Path Length:**
  - Compute **average shortest path length**, handling disconnected components with harmonic mean

- **Clustering Coefficient:**
  - Check Global clustering coefficient since our graph is directed
  
- **Centrality Measures**:
  - **Degree Centrality**: Identifies players involved in frequent interactions.
  - **Closeness Centrality / Harmonic Centrality:** Captures player accessibility in the network.
  - **Eigenvector Centrality & PageRank:** Measures influence within the passing network.
  - **Hubs and Authorities:**
    - Hubs: Players with **high outgoing edges** (active passers/playmakers).
    - Authorities: Players with **high incoming edges** (finishers/scorers).

---

### 5. Community Detection

- Identify **clusters of players** who frequently interact (e.g., effective lineups).
- Detect **historical "super-teams" and dominant duos**.

**Approaches:**
- Not covered during class yet
<!-- - **Louvain Method** for detecting natural player groupings.
- **Spectral clustering** to analyze team chemistry.
  - (Compare it with true label?) -->

---

### 6. Advanced Analysis - TODO: Need to discuss this

#### 6.1 Temporal Network Dynamics
- Track **network evolution** over time (e.g., player interactions shifting due to trades).
- Compare **short-term vs. long-term impact of trades**.
- Measure how **team chemistry scores** fluctuate across seasons.

#### 6.2 Predictive Modeling (ML Approaches)
- **Graph Neural Networks (GNNs)** for predicting:
  - Player performance (impact on team success)
  - Future passing network structures (link prediction)
  
- **Evaluation Metrics**:
  - **Correlation with traditional stats** (e.g., Plus/Minus, PER)
  - **Predictive accuracy** of future interactions

#### 6.3 Chemistry Measurement
- Define **team chemistry score** based on:
  - Network clustering coefficient
  - Average interaction weights within a team
  - Consistency of high-degree interactions

- Investigate **how chemistry correlates with team success** (e.g., playoff performance).

---

### 7. Visualization

- **Network Graphs**:
  - Static: Season-wise player networks
  - Dynamic: Animation of network evolution over time
    - evaluate growth / preferential attachment

- **Heatmaps**: Team chemistry evolution across seasons.

- **Historical Playstyle Comparisons**:
  - Visualizing **NBA era shifts** (e.g., transition from ISO-heavy to pass-heavy modern basketball).

---

### 8. Outcome & Findings

- **Alignment with NBA Trends:**
  - TODO

- **Unexpected Findings:**
  - TODO

---

## References
- **NBA API** - [https://github.com/swar/nba_api](https://github.com/swar/nba_api)
