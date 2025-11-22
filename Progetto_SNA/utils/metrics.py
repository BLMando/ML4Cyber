import matplotlib.pyplot as plt
import networkx as nx


def calc_metrics(G):
    closeness = nx.closeness_centrality(G)
    degree = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    eigenvector = nx.eigenvector_centrality(G)
    return (closeness, degree, betweenness, eigenvector)


def plot_metrics(data, title, labelx, labely):
    plt.figure()
    # tipo di plot
    plt.bar(data)
    plt.title(title)
    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.tight_layout()
    plt.show()
