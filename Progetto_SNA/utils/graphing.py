from enum import Enum
from dataclasses import dataclass

from numpy import log
from typing import Callable
import networkx as nx
import matplotlib.pyplot as plt
from enum import Enum


def _to_inch(n, dpi=300) -> float:
    return round(n / dpi, 1)


def PixToInch(width, height, dpi):
    return (_to_inch(width, dpi), _to_inch(height, dpi))


class FigSize(Enum):
    _dpi = 300
    DPI = _dpi
    AUTO = None
    XXS1_1 = PixToInch(500, 500, _dpi)
    XS1_1 = PixToInch(1000, 1000, _dpi)
    S1_1 = PixToInch(1500, 1500, _dpi)
    M1_1 = PixToInch(2000, 2000, _dpi)
    L1_1 = PixToInch(2500, 2500, _dpi)
    XL1_1 = PixToInch(3000, 3000, _dpi)
    XXL1_1 = PixToInch(5000, 5000, _dpi)
    XXXL1_1 = PixToInch(10000, 10000, _dpi)
    ENORMOUS1_1 = PixToInch(15000, 15000, _dpi)
    XE1_1 = PixToInch(50000, 50000, _dpi)
    XXS16_9 = (1.7, 0.9)  #  500 × 281
    XS16_9 = (3.3, 1.9)  # 1000 × 562
    S16_9 = (5.0, 2.8)  # 1500 × 844
    M16_9 = (6.7, 3.4)  # 2000 × 1125
    L16_9 = (8.3, 4.2)  # 2500 × 1406
    XL16_9 = (10.0, 5.0)  # 3000 × 1688
    XXL16_9 = (11.7, 5.9)  # 3500 × 1969
    XXXL16_9 = (13.3, 6.8)  # 4000 × 2250
    ENORMOUS16_9 = (16.7, 9.4)  # 5000 × 2812
    XE16_9 = (33.3, 18.8)  # 10000 × 5625
    XXS4_3 = (1.7, 1.3)  #  500 × 375
    XS4_3 = (3.3, 2.5)  # 1000 × 750
    S4_3 = (5.0, 3.8)  # 1500 × 1125
    M4_3 = (6.7, 5.0)  # 2000 × 1500
    L4_3 = (8.3, 6.3)  # 2500 × 1875
    XL4_3 = (10.0, 7.5)  # 3000 × 2250
    XXL4_3 = (11.7, 8.8)  # 3500 × 2625
    XXXL4_3 = (13.3, 10.0)  # 4000 × 3000
    ENORMOUS4_3 = (16.7, 12.5)  # 5000 × 3750
    XE4_3 = (33.3, 25.0)  # 10000 × 7500


class GLAYOUTS(Enum):
    arf: Callable = nx.arf_layout
    bipartite: Callable = nx.bipartite_layout
    # bfs: Callable = nx.bfs_layout
    circular: Callable = nx.circular_layout
    # forceatlas2: Callable = nx.forceatlas2_layout
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


def add_edges(G, df):
    for _, row in df.dropna().iterrows():
        G.add_edge(row["source"], row["target"])


def edge_collapse(G, type: Callable = nx.MultiDiGraph):
    H = type()
    for u, v in G.edges():
        if H.has_edge(u, v):
            H[u][v]["w"] += 1
        else:
            H.add_edge(u, v, w=1)
    return H
