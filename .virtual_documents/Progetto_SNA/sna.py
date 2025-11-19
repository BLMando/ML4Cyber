


# Importiamo tutte le dipendenze

from utils import FigSize, preproc
from pathlib import Path
from typing import Callable
from enum import Enum
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import re
from tqdm.notebook import tqdm
import warnings
import time

warnings.filterwarnings("ignore")





session_id = str(time.time())


session_id = "1763493272.2638242"  # placeholder





# ed eseguiamo le operazioni preliminari di caricamento dei dati

# leggi il file come edge-list: ignora righe che iniziano con '#' e usa whitespace come separatore
#
cit_hepth = pd.read_csv(
    "./data/cit-HepTh.txt", comment="#", sep="\\s+", header=None, engine="python"
)
# Prendiamo le prime due colonne come source/target
citations = cit_hepth.iloc[:, :2].copy()
citations.columns = ["source", "target"]
citations["source"] = pd.to_numeric(citations["source"])
citations["target"] = pd.to_numeric(citations["target"])

del cit_hepth





records = []
paths = Path("data/cit-HepTh-abstracts").rglob("*")
for abstractsp in tqdm(paths):
    if abstractsp.is_file():
        with open(abstractsp, "r", encoding="utf-8", errors="ignore") as f:
            abstract = f.read()

        data = {"id": abstractsp.stem}
        fields = preproc.extract_fields(abstract)
        if isinstance(fields, dict):
            data.update(fields)
            records.append(data)

papers = pd.DataFrame(records)






papers.to_csv(f"data/papers-{session_id}.csv", index=False)


papers = pd.read_csv(f"data/papers-{session_id}.csv")





papers








ror = pd.read_csv("./data/v1.73-2025-10-28-ror-data.csv")
ror["clean_url"] = (
    ror["links"].str.replace(r"^https?://", "", regex=True).str.split("/").str[0]
)
ror["tld2"] = ror["clean_url"].str.extract(r"([a-zA-Z0-9-]+\.[a-zA-Z0-9-]+)$")
# ror = ror[["name", "tld2"]]

df = papers.copy()


def extract_domain(email):
    if not isinstance(email, str):
        return None
    if "@" not in email:
        return None

    domain = email.split("@", 1)[1].lower()

    # LIKE-style match contro all_universities.csv
    cond_a = universities["domains"].str.contains(domain, case=False, na=False)
    cond_b = universities["domains"].apply(
        lambda d: isinstance(d, str) and d.lower() in domain
    )
    mask = cond_a | cond_b

    # restituisco SOLO name e country
    uni_match = universities.loc[mask, ["name", "alpha_two_code"]]

    if not uni_match.empty:
        row = uni_match.iloc[0]
        return {"name": row["name"], "country": row["alpha_two_code"]}

    # FALLBACK: ROR via TLD2
    m = re.search(r"([a-zA-Z0-9-]+\.[a-zA-Z0-9-]+)$", domain)
    if m:
        tld2 = m.group(1)
        ror_match = ror.loc[ror["tld2"].eq(tld2), ["name", "country.country_code"]]

        if not ror_match.empty:
            row = ror_match.iloc[0]
            return {"name": row["name"], "country": row["country.country_code"]}

    return None


for i in df.itertuples():
    print(i)
    time.sleep(2)


extract_domain("dbernard@spht.saclay.cea.fr")


domain_mapping = {
    str(row.id): extract_domain(row.email) for row in tqdm(df.itertuples())
}


domain_mapping["0208160"]


ror = pd.read_csv("./data/v1.73-2025-10-28-ror-data.csv")
ror["clean_url"] = (
    ror["links"].str.replace(r"^https?://", "", regex=True).str.split("/").str[0]
)
ror["tld2"] = ror["clean_url"].str.extract(r"([a-zA-Z0-9-]+\.[a-zA-Z0-9-]+)$")


citations = pd.read_csv("./data/citations.csv")

citations_uni = citations.copy()
citations_country = citations.copy()


def safe_get_name(x):
    v = domain_mapping.get(x)
    if isinstance(v, dict):
        return v.get("name")
    return None


def safe_get_country(x):
    v = domain_mapping.get(x)
    if isinstance(v, dict):
        return v.get("country")
    return None


citations_uni["source"] = citations["source"].astype(str).map(safe_get_name)
citations_uni["target"] = citations["target"].astype(str).map(safe_get_name)

citations_country["source"] = citations["source"].astype(str).map(safe_get_country)
citations_country["target"] = citations["target"].astype(str).map(safe_get_country)


citations_uni.dropna().sample(n=3)


citations_country.dropna().sample(n=3)


citations_uni.to_csv(f"data/citations-uni-{session_id}.csv", index=False)
citations_country.to_csv(f"data/citations-country-{session_id}.csv", index=False)


citations_uni = pd.read_csv(f"data/citations-uni-{session_id}.csv")
citations_country = pd.read_csv(f"data/citations-country-{session_id}.csv")


citations = pd.read_csv(f"data/citations-{session_id}.csv")
papers = pd.read_csv("data/papers.csv")
universities = pd.read_csv("./data/all_universities.csv")





G_uni = nx.DiGraph()

for _, row in citations_uni.dropna().iterrows():
    src = row["source"]
    tgt = row["target"]
    G_uni.add_edge(src, tgt)


G_country = nx.DiGraph()

for _, row in citations_country.dropna().iterrows():
    src = row["source"]
    tgt = row["target"]
    G_country.add_edge(src, tgt)





class GLAYOUTS(Enum):
    kamada: Callable = nx.kamada_kawai_layout
    spring: Callable = nx.spring_layout
    circular: Callable = nx.circular_layout
    shell: Callable = nx.shell_layout
    spectral: Callable = nx.spectral_layout


def gen_graph(
    G,
    layout=GLAYOUTS.kamada,
    figsize: FigSize = FigSize.M1_1,
    dpi=100,
    cmap="viridis",
    save_path="graph.png",
    show_labels=True,
    title=None,
):
    plt.figure(figsize=figsize.value, dpi=FigSize.DPI.value)
    pos = layout(G)

    degrees = dict(G.degree())
    # ci permette di ottenere un dizionario dei degrees

    node_sizes = [80 + degrees[n] * 10 for n in G.nodes()]
    # definisce la grandezza di un nodo in base al valore del degree

    node_colors = [degrees[n] for n in G.nodes()]
    # evidentemente anche il colore

    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=cmap,
        alpha=0.85,
        linewidths=0.5,
        edgecolors="black",
    )

    # archi
    nx.draw_networkx_edges(
        G,
        pos,
        arrowstyle="-|>",
        arrowsize=10,
        edge_color="gray",
        alpha=0.3,
        width=0.8,
    )

    # etichette opzionali
    if show_labels:
        nx.draw_networkx_labels(G, pos, font_size=7, font_color="black")

    # colorbar e titolo
    cbar = plt.colorbar(nodes)
    cbar.set_label("Node degree")
    if title:
        plt.title(title)

    plt.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()


gen_graph(G_uni, figsize=FigSize.XE16_9)


gen_graph(G_country, figsize=FigSize.XL16_9)
