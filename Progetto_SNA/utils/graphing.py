from enum import Enum
from dataclasses import dataclass

import ipdb
from numpy import log
from utils.figsize import FigSize
from typing import Callable
import networkx as nx
import matplotlib.pyplot as plt


class GLAYOUTS(Enum):
    arf: Callable = nx.arf_layout
    bipartite: Callable = nx.bipartite_layout
    bfs: Callable = nx.bfs_layout
    circular: Callable = nx.circular_layout
    forceatlas2: Callable = nx.forceatlas2_layout
    kamada: Callable = nx.kamada_kawai_layout
    planar: Callable = nx.planar_layout
    random: Callable = nx.random_layout
    rescale: Callable = nx.rescale_layout
    rescale_dict: Callable = nx.rescale_layout_dict
    shell: Callable = nx.shell_layout
    spring: Callable = nx.spring_layout
    spectral: Callable = nx.spectral_layout
    spiral: Callable = nx.spiral_layout
    multipartite: Callable = nx.multipartite_layout


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


def clamp(value, min_value, max_value):
    return max(min(int(value), max_value), min_value)


def gen_graph_data(G, pos):
    if not pos:
        pos = GLAYOUTS.kamada(G)

    degrees = dict(G.degree())
    node_sizes = []
    data_alpha = []
    node_colors = []
    min_degrees = int(min(degrees.values()))
    max_degrees = int(max(degrees.values()))
    for n in G.nodes():
        node_sizes.append(90 + degrees[n])
        data_alpha.append(clamp(degrees[n], min_degrees, max_degrees))
        node_colors.append(degrees[n])

    return {
        "G": G,
        "pos": pos,
        "degrees": degrees,
        "node_sizes": node_sizes,
        "node_colors": node_colors,
        "alpha": data_alpha,
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


def block(func):
    return func()


def plot_graph(
    data,
    figsize=FigSize.AUTO,
    dpi=FigSize.DPI.value,
    save_path=None,
    show_labels=False,
    title="Graph",
    opts={},
):
    try:
        if figsize == FigSize.AUTO:
            n = data["G"].number_of_nodes()
            avg_node_size = sum(data["node_sizes"]) / n
            effective = n * avg_node_size

            # this function is defined here becouse python has no proper lambda function support (no multiline)
            def _size(nodes):
                match nodes:
                    case v if v < 500:
                        return FigSize.XXS16_9
                    case v if v < 1000:
                        return FigSize.XS16_9
                    case v if v < 1500:
                        return FigSize.S16_9
                    case v if v < 2000:
                        return FigSize.M16_9
                    case v if v < 5000:
                        return FigSize.L16_9
                    case v if v < 8000:
                        return FigSize.XL16_9
                    case v if v < 10_000:
                        return FigSize.XXL16_9
                    case v if v < 15_000:
                        return FigSize.XXXL16_9
                    case _:
                        return FigSize.XE16_9

            figsize = _size(effective)

        plt.figure(figsize=figsize.value, dpi=dpi)

        nodes = nx.draw_networkx_nodes(
            data["G"],
            data["pos"],
            node_size=data["node_sizes"],
            node_color=data["node_colors"],
            cmap=data["cmap"],
            alpha=1,
            linewidths=1,
            edgecolors="black",
        )

        nx.draw_networkx_edges(
            data["G"],
            data["pos"],
            arrowstyle="-|>",
            arrowsize=10,
            edge_color="gray",
            alpha=1,
            width=0.1,
        )

        if show_labels:
            nx.draw_networkx_labels(
                data["G"], data["pos"], font_size=7, font_color="black"
            )

        cbar = plt.colorbar(nodes)
        cbar.set_label("Node degree")

        if title:
            plt.title(title)

        plt.axis("off")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

        plt.show()
    except Exception as e:
        print(e)
