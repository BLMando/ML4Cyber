# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: data
#     language: python
#     name: data
# ---

# %% [markdown]
# # Analisi della social network tra istituti di ricerca e università.
# L'obiettivo di questa analisi è quella di individuare se, all'interno della comunità scientifica, esistano
# dei gruppi naturali (comunità) tra i diversi istituti di ricerca nel campo della Energia.
# %%

# Importiamo tutte le dipendenze

from dateutil import parser
from dateutil.parser._parser import UnknownTimezoneWarning
from email_validator import validate_email, EmailNotValidError
from pathlib import Path
from typing import cast
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import re
from tqdm import tqdm
import warnings
import time

warnings.filterwarnings("ignore")

# %% [markdown]
# Questa linea genera la session_id. Se la sovrascrivi si intende che hai fatto cambiamenti al dataset
# perciò il resto del codice non farà più affidamento alla sessione precedente e quindi alcuni file
# vanno rigenerati
#
# Se invece vuoi usare una session precedente, usa il blocco sotto e definisci manualmente il numero di sessione
#
# %% jupyter={"source_hidden": false}
session_id = str(time.time())
# %% jupyter={"source_hidden": false}
session_id = "010101010101"  # placeholder
# %% [markdown]
# ## Preprocessamento

# %%
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

# %% [markdown]
# esegui questo blocco per caricare le funzioni di preprocessamento dei dataset
#
# %%


def normalize_email(email_str):
    if not email_str:
        return None
    match = re.search(r'[\w\.-]+@[\w\.-]+', email_str)
    if match:
        return match.group(0).lower()
    return None

def extract_journal_ref(text: str):
    """
    Extract Journal-ref from Abstract text
    """
    journalref = re.search(r"^\s*Journal[- ]ref\s*:\s*(.+)", text, re.I | re.MULTILINE)
    if journalref:
        return journalref.group(1).strip()
    return None


def extract_pages(comment_str: str):
    """
    Extract page number from an abstract comment string.
    Returns an int or None.
    """
    match = re.search(
        r"(?i)(\b(?:p{1,2}|pages?)\.?\s*(\d+)\b|\b(\d+)\s*(?:p{1,2}|pages?)\.?\b)",
        comment_str,
    )
    if match:
        # groups: match.group(2) OR match.group(3) contains the number
        return int(match.group(2) or match.group(3))
    return None


def extract_date_fields(text):
    """
    Extract pubblication dates and revision
    """
    date_publish = None
    date_revised = None

    # Date originale
    match = re.search(r"^Date:\s*(.+?)(?:\s+\(\d+kb\))?$", text, re.MULTILINE)
    if match:
        date_publish = match.group(1).strip()

    # Date revised (es. 'Date (revised v2): ...')
    match_rev = re.search(
        r"^Date\s*\(revised.*?\):\s*(.+?)(?:\s+\(\d+kb\))?$", text, re.MULTILINE
    )
    if match_rev:
        date_revised = match_rev.group(1).strip()

    return date_publish, date_revised


def extract_fields(text: str):
    """
    Extract fields from abstract
    """

    data: dict[str, object] = {
        "paper": None,
        "from": None,
        "date_published": None,
        "date_revised": None,
        "title": None,
        "authors": None,
        "pages": None,
        "subj_class": None,
        "journal_ref": None,
    }

    keys = data.keys()

    for line in text.splitlines():
        line = line.lower().strip()
        tag, _, content = line.partition(":")
        if tag in keys:
            match tag:
                # case "paper":
                #     data["paper"] = content.strip()
                case "from":
                    data["from"] = normalize_email(content.strip())
                # case "date":
                #     dp, dr = extract_date_fields(content.strip())
                #     if dp:
                #         data["date_published"] = parser.parse(dp)
                #     if dr:
                #         data["date_revised"] = parser.parse(dr)
                # case "title":
                #     data["title"] = content.strip()
                # case "author":
                #     data["authors"] = content.strip()
                # case "comment":
                #     data["pages"] = extract_pages(content.strip())
                # case "journal_ref":
                #     data["journal_ref"] = extract_journal_ref(content.strip())
                case _:
                    continue

    return data


# %% [markdown]
# qui effettuiamo il preprocessamento
# %% jupyter={"source_hidden": false}
#
records = []
paths = Path("data/cit-HepTh-abstracts").rglob("*")
for abstractsp in tqdm(paths):
    if abstractsp.is_file():
        with open(abstractsp, "r", encoding="utf-8", errors="ignore") as f:
            abstract = f.read()

        data = {"id": abstractsp.stem}
        fields = extract_fields(abstract)
        if isinstance(fields, dict):
            data.update(fields)
            records.append(data)

papers = pd.DataFrame(records)

# %%
esegui questo blocco per salvare il preprocessamento

# %%
papers.to_csv(f"data/papers-{session_id}.csv", index=False)

# %% [markdown]
# Non è necessario rieseguire tutto quanto da capo.

# %%
papers

# %%

# %% [markdown]
# Mapping dei paper alle rispettive università

# %%
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
    cond_a = universities["domains"].str.contains(domain, case=False, na=False)
    cond_b = universities["domains"].apply(
        lambda d: isinstance(d, str) and d.lower() in domain
    )
    mask = cond_a | cond_b
    uni_match = universities.loc[mask, "name", "country"]

    if not uni_match.empty:
        return uni_match.iloc[0]

    m = re.search(r"([a-zA-Z0-9-]+\.[a-zA-Z0-9-]+)$", domain)
    if m:
        tld2 = m.group(1)
        if tld2 in ror.tld2.values:
            m = ror.loc[ror["tld2"].eq(tld2), "name", "country.country_code"]
            return m.iat[0] if not m.empty else None

    return None


# %%
domain_mapping = {str(row.id): extract_domain(row.from) for row in df.itertuples()}

# %%
ror = pd.read_csv("./data/v1.73-2025-10-28-ror-data.csv")
ror["clean_url"] = (
    ror["links"].str.replace(r"^https?://", "", regex=True).str.split("/").str[0]
)
ror["tld2"] = ror["clean_url"].str.extract(r"([a-zA-Z0-9-]+\.[a-zA-Z0-9-]+)$")
# ror = ror[["name", "tld2"]]

df = papers.copy()
citations = pd.read_csv("./data/citations.csv")

# Potremmo salvare il risultato invece che rifarlo ogni volta
citations["source"] = (
    citations["source"].astype(str).map(lambda x: domain_mapping.get(x, None))
)
citations["target"] = (
    citations["target"].astype(str).map(lambda x: domain_mapping.get(x, None))
)
citations.to_csv("data/citations-pp.csv", index=False)

# %%
citations = pd.read_csv("data/citations-pp.csv")
papers = pd.read_csv("data/papers.csv")
universities = pd.read_csv("./data/all_universities.csv")

# %% [markdown]
# Preparazione del grafo

# %%
G = nx.DiGraph()

for _, row in citations.iterrows():
    src = row["source"]
    tgt = row["target"]
    if pd.isna(src) or pd.isna(tgt):
        continue
    G.add_edge(src, tgt)

# %% [markdown]
# Visualizzazione del grafo

# %%
plt.figure(figsize=(76.8, 43.2), dpi=100)

pos = nx.kamada_kawai_layout(G)

degrees = dict(G.degree())
node_sizes = [80 + degrees[n] * 10 for n in G.nodes()]
node_colors = [degrees[n] for n in G.nodes()]


nodes = nx.draw_networkx_nodes(
    G,
    pos,
    node_size=node_sizes,
    node_color=node_colors,
    cmap="viridis",
    alpha=0.85,
    linewidths=0.5,
    edgecolors="black",
)


nx.draw_networkx_edges(
    G, pos, arrowstyle="-|>", arrowsize=10, edge_color="gray", alpha=0.3, width=0.8
)


nx.draw_networkx_labels(G, pos, font_size=7, font_color="black")


cbar = plt.colorbar(nodes)
cbar.set_label("Node degree")

plt.axis("off")
plt.tight_layout()
plt.savefig("graph_2.png", dpi=100, bbox_inches="tight")
plt.show()


# %%
