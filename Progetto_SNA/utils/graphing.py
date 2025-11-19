from enum import Enum
from dataclasses import dataclass

from numpy import log
from utils.figsize import FigSize
from typing import Callable
import networkx as nx
import matplotlib.pyplot as plt


class GLAYOUTS(Enum):
    kamada: Callable = nx.kamada_kawai_layout
    spring: Callable = nx.spring_layout
    circular: Callable = nx.circular_layout
    shell: Callable = nx.shell_layout
    spectral: Callable = nx.spectral_layout


class CMAP(str, Enum):
    VIRIDIS = "viridis"
    PLASMA = "plasma"
    INFERNO = "inferno"
    MAGMA = "magma"
    CIVIDIS = "cividis"
    COOLWARM = "coolwarm"
    JET = "jet"
    GREENS = "Greens"
    BLUES = "Blues"


def gen_default(G, pos):
    if not pos:
        pos = GLAYOUTS.kamada(G)
    degrees = dict(G.degree())
    node_sizes = [80 + log(degrees[n] * 10) for n in G.nodes()]
    node_colors = [degrees[n] for n in G.nodes()]
    return {
        "G": G,
        "pos": pos,
        "degrees": degrees,
        "node_sizes": node_sizes,
        "node_colors": node_colors,
        "cmap": CMAP.COOLWARM,
    }


def gen_graph_distance(G):
    pos = GLAYOUTS.spring(G, weight="weight", k=None, iterations=200)
    degrees = dict(G.degree())
    node_sizes = [80 + log(degrees[n] * 10) for n in G.nodes()]
    node_colors = [degrees[n] for n in G.nodes()]
    return {
        "G": G,
        "pos": pos,
        "degrees": degrees,
        "node_sizes": node_sizes,
        "node_colors": node_colors,
        "cmap": CMAP.INFERNO,
    }


def plot_graph(
    data,
    figsize=FigSize.XL16_9,
    dpi=FigSize.DPI.value,
    save_path=None,
    show_labels=True,
    title="Graph",
):
    plt.figure(figsize=figsize.value, dpi=dpi)

    nodes = nx.draw_networkx_nodes(
        data["G"],
        data["pos"],
        node_size=data["node_sizes"],
        node_color=data["node_colors"],
        cmap=data["cmap"],
        alpha=0.85,
        linewidths=0.5,
        edgecolors="black",
    )

    nx.draw_networkx_edges(
        data["G"],
        data["pos"],
        arrowstyle="-|>",
        arrowsize=10,
        edge_color="gray",
        alpha=0.3,
        width=0.8,
    )

    if show_labels:
        nx.draw_networkx_labels(data["G"], data["pos"], font_size=7, font_color="black")

    cbar = plt.colorbar(nodes)
    cbar.set_label("Node degree")

    if title:
        plt.title(title)

    plt.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()
