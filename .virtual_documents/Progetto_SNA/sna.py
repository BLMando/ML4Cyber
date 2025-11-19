


# Importiamo tutte le dipendenze

from utils import FigSize, preproc, rand_string
from pathlib import Path
import networkx as nx
import pandas as pd
from tqdm.notebook import tqdm
import warnings
import os
from datetime import datetime


CITATIONS_DIRECTED_GRAPH = "./data/cit-HepTh.txt"
CITATIONS_ABSTRACTS_DIR = "./data/cit-HepTh-abstracts"

ROR_DATA = "./data/ror-data.csv"
UNIVERSITIES_DATA = "./data/all_universities.csv"

warnings.filterwarnings("ignore")






s = datetime.now().strftime("%y%m%d%M%S")
session_id = f"{rand_string(8)}-{(s)}"  # NUOVA SESSIONE
SESSION_PATH = f"data/sessions/{session_id}"
os.makedirs(SESSION_PATH, exist_ok=True)



session_id = ""  # RICARICA UNA SESSIONE
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








gen_graph(G_uni, figsize=FigSize.XE16_9)


gen_graph(G_country, figsize=FigSize.XL16_9)
