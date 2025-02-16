import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_degree_pdf(k_values, Pk_values, random_k, random_Pk, title, color, ax):
    ax.scatter(k_values, Pk_values, color=color, label="NBA Network", alpha=0.7)
    ax.scatter(random_k, random_Pk, color="gray", label="Erdős-Rényi Graph", alpha=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Degree k")
    ax.set_ylabel("P(k)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)


def degree_distribution(degree_sequence):
    unique, counts = np.unique(degree_sequence, return_counts=True)
    pk = counts / sum(counts)  # Normalize to probability
    return unique, pk


def print_top_players(metric_dict, title, top_n=10):
    sorted_players = sorted(metric_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    print(f"\n🔥 Top {top_n} Players by {title}:")
    for player, score in sorted_players:
        print(f"{player}: {score:.4f}")