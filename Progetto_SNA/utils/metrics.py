import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np


def calc_metrics(G):
    degree = nx.degree_centrality(G)
    closeness = nx.closeness_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    eigenvector = nx.eigenvector_centrality(G)
    return (degree, closeness, betweenness, eigenvector)

def plot_distribution(values, path, title, scale):

    fig, ax = plt.subplots(figsize=(15, 5))
    
    values = sorted(values)
    n = len(values)
    bin_width = 3.5 * np.std(values) / (n ** (1 / 3))
    numero_bin = int((max(values) - min(values)) / bin_width)*3

    sns.histplot(values, kde=True, bins=numero_bin,  ax=ax)  
    #plt.xticks(rotation=80, fontsize=8, ax=ax)
    ax.set_ylabel("Numero Università", fontsize=11)
    #ax.xlabel("Centralità", fontsize=11)
    print("Mean:", np.mean(values))
    ax.axvline(x=np.mean(values), color='red', ls='--', lw=2, label='Media')
    ax.set_title(title, fontsize=12)
    ax.legend(loc='upper right')#bbox_to_anchor=(1.0, 1), 
    
    # if opzione != "clustering":
    #     ax.set_xlim(0, 80)  
    # else:
    ax.set_xlim(0, scale)  

    plt.savefig(path + "/" + title +".png",format= 'png', bbox_inches='tight') 
    plt.show()
