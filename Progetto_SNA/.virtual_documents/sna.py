


# Importiamo tutte le dipendenze

from importlib import reload
from datetime import datetime
from pathlib import Path
from tqdm.notebook import tqdm
from utils.graphing import FigSize, GLAYOUTS
import utils.graphing as graphing
import utils.metrics as metrics
import networkx as nx
import os
import pandas as pd
import utils.preproc as preproc
import warnings
import ipdb


CITATIONS_DIRECTED_GRAPH = "./data/cit-HepTh.txt"
CITATIONS_ABSTRACTS_DIR = "./data/cit-HepTh-abstracts"

ROR_DATA = "./data/ror-data.csv"
UNIVERSITIES_DATA = "./data/all_universities.csv"

warnings.filterwarnings("ignore")





s = datetime.now().strftime("%y%m%d%H%M")
session_id = f"{s}"  # NUOVA SESSIONE
SESSION_PATH = f"data/sessions/{session_id}"
os.makedirs(SESSION_PATH, exist_ok=True)


session_id = "2511212247"  # RICARICA UNA SESSIONE
SESSION_PATH = f"data/sessions/{session_id}"





records = []

for abp in tqdm(Path(CITATIONS_ABSTRACTS_DIR).rglob("*")):
    if abp.is_file():
        with open(abp, "r", encoding="utf-8", errors="ignore") as f:
            abs = f.read()

        data = {"id": abp.stem}
        fields = preproc.extract_fields(abs)
        # preproc è una classe statica definita in utils.py

        if isinstance(fields, dict):
            data.update(fields)
            records.append(data)

papers = pd.DataFrame(records)
del records





ror = pd.read_csv(ROR_DATA)
ror["clean_url"] = (
    ror["links"].str.replace(r"^https?://", "", regex=True).str.split("/").str[0]
)
ror["tld2"] = ror["clean_url"].str.extract(r"([a-zA-Z0-9-]+\.[a-zA-Z0-9-]+)$")

universities = pd.read_csv(UNIVERSITIES_DATA)

domain_mapping = {
    str(row.id): preproc.extract_domain(row.email, ror, universities)
    for row in tqdm(papers.itertuples())
}


# leggi il file come edge-list: ignora righe che iniziano con '#' e usa whitespace come separatore

cit_hepth = pd.read_csv(
    CITATIONS_DIRECTED_GRAPH, comment="#", sep="\\s+", header=None, engine="python"
)

# Prendiamo le prime due colonne come source/target
citations = cit_hepth.iloc[:, :2].copy()
citations.columns = ["source", "target"]
citations["source"] = pd.to_numeric(citations["source"])
citations["target"] = pd.to_numeric(citations["target"])

del cit_hepth  # non ci serve più

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





citations_uni.to_csv(f"{SESSION_PATH}/citations-uni.csv", index=False)
citations_country.to_csv(f"{SESSION_PATH}/citations-country.csv", index=False)


papers.to_csv(f"{SESSION_PATH}/papers.csv", index=False)





citations_uni = pd.read_csv(f"{SESSION_PATH}/citations-uni.csv")
citations_country = pd.read_csv(f"{SESSION_PATH}/citations-country.csv")


papers = pd.read_csv(f"{SESSION_PATH}/papers.csv")





# per testing e sviluppo delle librerire, rilanciare questo blocco ogni volta che viene
# aggiornata una libreria

# Source - https://stackoverflow.com/a/437591
# Posted by cdleary, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-21, License - CC BY-SA 4.0


graphing = reload(graphing)





name = "circular-wpg"
pg = nx.DiGraph()
add_edges(pg, citations_uni)
wpg = edge_collapse(pg, nx.DiGraph)
lay = graphing.GLAYOUTS.circular
pos = lay(wpg)
data = graphing.gen_graph_data(wpg, pos)
graphing.plot_graph(data, save_path=f"{SESSION_PATH}/{name}")





# Source - https://stackoverflow.com/a/437591
# Posted by cdleary, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-21, License - CC BY-SA 4.0
from importlib import reload  # Python 3.4+
graphing = reload(graphing)
name = "graph-arf-wpg"
pg = nx.DiGraph()
add_edges(pg, citations_uni)
wpg = edge_collapse(pg, nx.DiGraph)
lay = graphing.GLAYOUTS.arf
pos = lay(wpg)
data = graphing.gen_graph_data(wpg, pos)
graphing.plot_graph(data, save_path=f"{SESSION_PATH}/{name}")


# Source - https://stackoverflow.com/a/437591
# Posted by cdleary, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-21, License - CC BY-SA 4.0
from importlib import reload  # Python 3.4+
graphing = reload(graphing)
name = "graph-bfs"

pg = nx.DiGraph()
add_edges(pg, citations_uni)
wpg = edge_collapse(pg, nx.DiGraph)
lay = graphing.GLAYOUTS.bipartite
pos = lay(pg)
data = graphing.gen_default(pg, pos)
graphing.plot_graph(data, save_path=f"{SESSION_PATH}/{name}")


# Source - https://stackoverflow.com/a/437591
# Posted by cdleary, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-21, License - CC BY-SA 4.0
from importlib import reload  # Python 3.4+
graphing = reload(graphing)
name = "graph-bfs"
pg = nx.DiGraph()
add_edges(pg, citations_uni)
wpg = edge_collapse(pg, nx.DiGraph)
lay = graphing.GLAYOUTS.bfs
pos = lay(pg)
data = graphing.gen_default(pg, pos)
graphing.plot_graph(data, save_path=f"{SESSION_PATH}/{name}")


# Source - https://stackoverflow.com/a/437591
# Posted by cdleary, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-21, License - CC BY-SA 4.0
from importlib import reload  # Python 3.4+
graphing = reload(graphing)
name = "kamada-wpg"
pg = nx.DiGraph()
add_edges(pg, citations_uni)
wpg = edge_collapse(pg, nx.DiGraph)
lay = graphing.GLAYOUTS.kamada
pos = lay(wpg, weight="w")
data = graphing.gen_graph_data(wpg, pos)
graphing.plot_graph(data, save_path=f"{SESSION_PATH}/{name}")


# Source - https://stackoverflow.com/a/437591
# Posted by cdleary, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-21, License - CC BY-SA 4.0
from importlib import reload  # Python 3.4+
graphing = reload(graphing)
name = "planar"
pg = nx.DiGraph()
add_edges(pg, citations_uni)
wpg = edge_collapse(pg, nx.DiGraph)
lay = graphing.GLAYOUTS.planar
pos = lay(pg)
data = graphing.gen_default(pg, pos)
graphing.plot_graph(data, save_path=f"{SESSION_PATH}/{name}")


# Source - https://stackoverflow.com/a/437591
# Posted by cdleary, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-21, License - CC BY-SA 4.0
from importlib import reload  # Python 3.4+
graphing = reload(graphing)
name = "spring-base-wpg"
pg = nx.DiGraph()
add_edges(pg, citations_uni)
wpg = edge_collapse(pg, nx.DiGraph)
lay = graphing.GLAYOUTS.spring
pos = lay(wpg, weight="w")
data = graphing.gen_graph_data(wpg, pos)
graphing.plot_graph(data, save_path=f"{SESSION_PATH}/{name}")


# Source - https://stackoverflow.com/a/437591
# Posted by cdleary, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-21, License - CC BY-SA 4.0
from importlib import reload  # Python 3.4+
graphing = reload(graphing)
name = "spring-force-wpg"
pg = nx.DiGraph()
add_edges(pg, citations_uni)
wpg = edge_collapse(pg, nx.DiGraph)
lay = graphing.GLAYOUTS.spring
pos = lay(wpg, weight="w", method="force")
data = graphing.gen_graph_data(wpg, pos)
graphing.plot_graph(data, save_path=f"{SESSION_PATH}/{name}")


# Source - https://stackoverflow.com/a/437591
# Posted by cdleary, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-21, License - CC BY-SA 4.0
from importlib import reload  # Python 3.4+
graphing = reload(graphing)
name = "spring-energy-wpg"
pg = nx.DiGraph()
add_edges(pg, citations_uni)
wpg = edge_collapse(pg, nx.DiGraph)
lay = graphing.GLAYOUTS.spring
pos = lay(wpg, weight="w", method="energy")
data = graphing.gen_graph_data(wpg, pos)
graphing.plot_graph(data, save_path=f"{SESSION_PATH}/{name}", show_labels=False)


# Source - https://stackoverflow.com/a/437591
# Posted by cdleary, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-21, License - CC BY-SA 4.0
from importlib import reload  # Python 3.4+
graphing = reload(graphing)
name = "spiral"
pg = nx.DiGraph()
add_edges(pg, citations_uni)
wpg = edge_collapse(pg, nx.DiGraph)
lay = graphing.GLAYOUTS.spiral
pos = lay(pg, resolution=1)
data = graphing.gen_graph_data(pg, pos)
graphing.plot_graph(data, save_path=f"{SESSION_PATH}/{name}")


# Source - https://stackoverflow.com/a/437591
# Posted by cdleary, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-21, License - CC BY-SA 4.0
from importlib import reload  # Python 3.4+
graphing = reload(graphing)
name = "spiral-equidistant"
pg = nx.DiGraph()
add_edges(pg, citations_uni)
wpg = edge_collapse(pg, nx.DiGraph)
lay = graphing.GLAYOUTS.spiral
pos = lay(pg, resolution=1, equidistant=True)
data = graphing.gen_graph_data(pg, pos)
graphing.plot_graph(data, save_path=f"{SESSION_PATH}/{name}")





unique = len(pd.unique(citations_uni[['source', 'target']].dropna().values.ravel('K')))
self_loops = len(citations_uni[citations_uni['source'] == citations_uni["target"]].dropna())
edges = len(citations_uni.dropna())
print(f"Abbiamo {unique} universita e centri di ricerca")
print(f"        {edges} archi")
print(f"        {self_loops} self loops")


unique = len(pd.unique(citations_country[['source', 'target']].dropna().values.ravel('K')))
self_loops = len(citations_country[citations_country['source'] == citations_country["target"]].dropna())
edges = len(citations_country.dropna())
print(f"Abbiamo {unique} stati")
print(f"        {edges} archi")
print(f"        {self_loops} self loops")


citations_uni.count()














g = nx.DiGraph()
add_edges(g, citations_uni)
wg = g.copy()
wg = edge_collapse(g, nx.DiGraph)


wgm = metrics(wg)
gm  = metrics(g)






