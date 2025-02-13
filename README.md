# NBA Player Interaction Network Analysis

## Overview

This project constructs an evolving **NBA player interaction network** using event-based data from the **NBA API**. Players are connected based on their in-game interactions (**passes, assists**) over multiple seasons. By modeling this as a graph and analyzing its evolution over time, we aim to **understand long-term player impact, team chemistry, and the dynamics of NBA gameplay across different eras**.

## Components of the Project

### 1. Project Goal / Hypothesis

- **Hypothesis:** 
- **Objectives:**
  - Analyze **how player interactions evolve** over multiple seasons.
  - Identify **key playmakers** based on their network influence.
  - Measure **team chemistry** and its relation to team performance.
    - (How are we going to measure chemistry? Centrality?)
  - Compare **historical playstyles** by analyzing network structures across NBA eras.

---

### 2. Data Acquisition

- **Source:** [NBA API](https://github.com/swar/nba_api)
- **Nodes:** NBA players (historically ~4,500+ players)
- **Edges:** Passes that led to scoring
- **Temporal aspect:** Multi-season evolving network (layered by year)

---

### 3. Network Representation

- **Graph Type:** 
  - **Directed** (captures passing direction)
  - **Weighted** (interaction frequency)
  - **Multilayer** (each season as a separate layer)
    - (Need to take a look at multilayer graph)

- **Network Construction:**
  - Nodes: Individual players
  - Edges: Weighted by interaction frequency

- **Temporal Representation:**
  - Investigate **rolling time windows** vs. **per-season snapshots** to track evolving connections.

---

### 4. Structural Analysis

#### 4.1 Fundamental Analysis
- **Basic Statistics**: 
  - Number of players, teams, games per season
  - Edge density (interaction frequency)
  - Degree distribution (pass-heavy vs. iso-heavy players)
  
- **Centrality Measures**:
  - **Degree Centrality**: Identifies players involved in frequent interactions.
  - **Betweenness Centrality**: Captures players who act as bridges in the network.
  - **PageRank**: Measures long-term influence within the passing network.

---

### 5. Community Detection

- Identify **clusters of players** who frequently interact (e.g., effective lineups).
- Detect **historical "super-teams" and dominant duos**.

**Approaches:**
- **Louvain Method** for detecting natural player groupings.
- **Spectral clustering** to analyze team chemistry.
  - (Compare it with true label?)

---

### 6. Advanced Analysis

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

- **Heatmaps**: Team chemistry evolution across seasons.

- **Historical Playstyle Comparisons**:
  - Visualizing **NBA era shifts** (e.g., transition from ISO-heavy 90s to pass-heavy modern basketball).

---

### 8. Outcome & Findings

- **Alignment with NBA Trends:**
  - TODO

- **Unexpected Findings:**
  - TODO

---

## References
- **NBA API** - [https://github.com/swar/nba_api](https://github.com/swar/nba_api)
