# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
lati = ['left', 'right']

emisfero_sinistro = cervello.drop(cervello[(cervello.side != 'left') | (cervello.side.isnull())].index)

emisfero_destro   = cervello.drop(cervello[(cervello.side != 'right') | (cervello.side.isnull())].index)

print(len(emisfero_sinistro))
print(len(emisfero_destro))
condizione = "emisfero_sinistro[((emisfero_sinistro.flow == nodo['flow'])"
condizione = condizione + " & ( emisfero_sinistro.super_class == nodo['super_class'])"
condizione = condizione + " & (emisfero_sinistro.classe == nodo['classe'])"
condizione = condizione + " & (emisfero_sinistro.sub_class == nodo['sub_class'])"
condizione = condizione + " & (emisfero_sinistro.cell_type == nodo['cell_type'])"
condizione = condizione + " & (emisfero_sinistro.hemibrain_type == nodo['hemibrain_type'])"
condizione = condizione + " & (emisfero_sinistro.hemilineage == nodo['hemilineage'])"
condizione = condizione + " & (emisfero_sinistro.nerve == nodo['nerve']))].index"

for index, nodo in emisfero_destro.iterrows():
    i = eval(condizione)
    #i = emisfero_sinistro[(emisfero_sinistro.flow == nodo['flow'])].index
    if i.size > 0:
        emisfero_destro   = emisfero_destro.drop(emisfero_destro[(emisfero_destro.root_id == nodo['root_id'])].index)
        emisfero_sinistro = emisfero_sinistro.drop(i[0])
print(len(emisfero_sinistro))
print(len(emisfero_destro))

emisfero_sinistro.to_csv('csv/Emisfero sinistro.csv', index=False)
emisfero_destro.to_csv('csv/Emisfero destro.csv', index=False)

# %%
import networkx as nx
import igraph as ig
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pickle
import math
import plotly.graph_objects as go
from plotly.offline import plot

# %%
#per far visualizzare tutto il dataframe
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 3000)

# %%
cervello = pd.read_csv('../data_drosophila_adulta/classification.csv')
cervello.columns.values[3] = 'classe'
cervello = cervello.replace(np.nan, "")

# %%
emisfero_sinistro = pd.read_csv('adulta/csv/Emisfero sinistro.csv')
emisfero_destro   = pd.read_csv('adulta/csv/Emisfero destro.csv')

emisfero_sinistro = emisfero_sinistro.replace(np.nan, "")
emisfero_destro   = emisfero_destro.replace(np.nan, "")


# %%
def stampa(insieme):
    elementi = ['flow', 'super_class', 'classe', 'sub_class', 'cell_type', 'hemibrain_type', 'hemilineage', 'side', 'nerve']
    for el in elementi:
        print(el)
        insieme = insieme.sort_values(by= [el], ascending=[False])
        print(list(dict.fromkeys(insieme[el])))
        print("####################################-")


# %%
#conta le occorrenze per area per ogni emisfero per i nodi aventi corrispettivo
def contaOccorrenze(insieme, titolo):

    aree = list(dict.fromkeys(cervello['super_class']))
    aree.sort()
    df = pd.DataFrame() 

    for area in aree:
        contatore_area = list(insieme['super_class']).count(area)
        new_row = {'super_class': area, 'occorrenze': contatore_area}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    df = df.sort_values(by= ['occorrenze', 'super_class'], ascending=[False, True])
    #print(df)
        
    plt.figure(figsize=(18, 8))
    sns.barplot(x=df["super_class"].tolist(), y=df["occorrenze"].tolist())
    plt.xticks(rotation=80, fontsize=9)
    plt.xlabel("Area", fontsize=13)
    plt.ylabel("Frequenza", fontsize=13)
    plt.title(titolo, fontsize=16)
    plt.tight_layout()
    plt.savefig('adulta/images/' + titolo + '.png', format= 'png', bbox_inches='tight')
    plt.show() 


# %%
#crea grafo 
def creaGrafo():
    df = pd.read_csv("../data_drosophila_adulta/connections.csv")

    G = nx.DiGraph()
    
    archi = df[['pre_root_id', 'post_root_id']]

    G.add_nodes_from(cervello['root_id'].values)

    G.add_edges_from(archi.values)
    
    return G


# %%
def aggiungiColonne(insieme):
    colonne = ['in_degree', 'out_degree', 'clustering', 'in_degree_perc', 'out_degree_perc', 'clustering_perc']
    for colonna in colonne:
        insieme[colonna] = 0

    return insieme

def calcoloInOutClustering(G, insieme):
    #calcolo in e out degree per ogni nodo
    degrees = ["in_degree", "out_degree"]
    for elemento in degrees:
        for index, neurone in insieme.iterrows():
            risultato = eval("G." + str(elemento) + "(neurone['root_id'])")
            if type(risultato) is int:
                insieme.at[index, elemento] = risultato
            else:
                insieme.at[index, elemento] = 0
    
    #calcolo clustering per ogni nodo
    for index, neurone in insieme.iterrows():
        risultato = nx.clustering(G, neurone['root_id'])
        if type(risultato) is float:
            insieme.at[index, 'clustering'] = risultato
        else:
            insieme.at[index, 'clustering'] = 0
            
    return insieme

def calcoloPercentuali(G, insieme1, insieme2):
    #calcolo percentuali di indegree, outdegree e clustering sul totale della rete
    opzioni = ["in_degree", "out_degree", "clustering"]
    for opzione in opzioni:
        stringa = opzione + "_perc"
        totale = insieme1[opzione].sum() + insieme2[opzione].sum()
        for index, neurone in insieme1.iterrows():
            insieme1.at[index, stringa] = (insieme1.at[index, opzione] * 100) / totale
    return insieme1

def calcoloValoriMedi(G, insieme1, insieme2):
    df = pd.DataFrame({'vm_in_degree': [], 'vm_out_degree': [], 'vm_clustering': [], 'vm_in_degree_perc': [], 'vm_out_degree_perc': [], 'vm_clustering_perc': []})
    new_row = {'vm_in_degree': 0, 'vm_out_degree': 0, 'vm_clustering': 0, 'vm_in_degree_perc': 0, 'vm_out_degree_perc': 0, 'vm_clustering_perc': 0}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    opzioni = ["in_degree", "out_degree", "clustering"]
    for opzione in opzioni:
        stringa = "vm_" + opzione + "_perc"
        totale = insieme1[opzione].sum() + insieme2[opzione].sum()
        #print(insieme1[opzione].sum())
        #print(len(insieme1[opzione]))
        #print("******************************")
        df.at[0, "vm_" + opzione] = insieme1[opzione].sum() / len(insieme1[opzione])
        df.at[0, stringa] = (df.at[0, "vm_" + opzione] * 100) / totale
    return df


# %%
def disegnaGraficiAccoppiati(insieme1, valori1, titolo1, insieme2, valori2, titolo2):
    #opzioni = ["in_degree_perc", "out_degree_perc", "clustering_perc"]
    opzioni = ["in_degree", "out_degree", "clustering"]
    for opzione in opzioni:
        
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        
        for i in range(2):
            if  i == 0:
                grafico = insieme1
                valori  = valori1
                titolo  = titolo1
            else:
                grafico = insieme2
                valori  = valori2
                titolo  = titolo2
            grafico = grafico.sort_values(by=[opzione], ascending=[True])
            n = len(list(grafico[opzione]))
            bin_width = 3.5 * np.std(list(grafico[opzione])) / (n ** (1 / 3))
            numero_bin = int((max(list(grafico[opzione])) - min(list(grafico[opzione]))) / bin_width)*3
    
        
            titolo_super = "Distribuzione del "
            match opzione:
                case "in_degree":
                    titolo_super += "grado di input"
                case "out_degree":
                    titolo_super += "grado di output"
                case "clustering":
                    titolo_super += "coefficiente di clustering"

            ax = axs[i]
            sns.histplot(list(grafico[opzione]), kde=True, bins=numero_bin,  ax=ax)  
            #plt.xticks(rotation=80, fontsize=8, ax=ax)
            ax.set_ylabel("Numero neuroni", fontsize=11)
            #ax.xlabel("Centralità", fontsize=11)
    
            
            ax.axvline(x=valori.at[0, "vm_" + opzione], color='red', ls='--', lw=2, label='Media')
            ax.set_title(titolo, fontsize=12)
            ax.legend(loc='upper right')#bbox_to_anchor=(1.0, 1), 
            if opzione != "clustering":
                ax.set_xlim(0, 80)  
            else:
                ax.set_xlim(0, 0.35)  
        fig.suptitle(titolo_super, y=1)
        plt.savefig("adulta/images/" + titolo_super + " " + titolo1 + " - " + titolo2 +".png",format= 'png', bbox_inches='tight') 
        plt.show()

# %%
nodi_senza_corrispettivo = pd.concat([emisfero_sinistro, emisfero_destro])
nodi_con_corrispettivo = cervello.drop(cervello[((cervello.side != 'left') & (cervello.side != 'right'))].index)
for index, nodo in nodi_senza_corrispettivo.iterrows():
    nodi_con_corrispettivo = nodi_con_corrispettivo.drop(nodi_con_corrispettivo[(nodi_con_corrispettivo.root_id == nodo['root_id'])].index)
print(len(nodi_con_corrispettivo))
contaOccorrenze(nodi_senza_corrispettivo, 'Grafico a barre della frequenza per area dei nodi senza corrispettivo')  
contaOccorrenze(nodi_con_corrispettivo,   'Grafico a barre della frequenza per area dei nodi con corrispettivo')
contaOccorrenze(emisfero_sinistro, 'Grafico a barre della frequenza per area dei nodi senza corrispettivo dell\'emisfero sinistro')
contaOccorrenze(emisfero_destro,   'Grafico a barre della frequenza per area dei nodi senza corrispettivo dell\'emisfero destro')

# %%
G = creaGrafo()

#ottengo i nodi senza corrispettivo per l'intero cervello
nodi_senza_corrispettivo = aggiungiColonne(nodi_senza_corrispettivo)
#ottengo i nodi con corrispettivo per l'intero cervello
nodi_con_corrispettivo   = aggiungiColonne(nodi_con_corrispettivo)

#calcolo per ogni nodo senza corrispettivo del cervello l'in_degree, l'out_degree, il clustering
nodi_senza_corrispettivo = calcoloInOutClustering(G, nodi_senza_corrispettivo)
#calcolo per ogni nodo con corrispettivo del cervello l'in_degree, l'out_degree, il clustering
nodi_con_corrispettivo   = calcoloInOutClustering(G, nodi_con_corrispettivo)

#calcolo per ogni nodo senza corrispettivo del cervello la percentuale di in_degree, di out_degree e di clustering del nodo isolato rispetto a tutti i nodi
nodi_senza_corrispettivo = calcoloPercentuali(G, nodi_senza_corrispettivo, nodi_con_corrispettivo)
#calcolo per ogni nodo con corrispettivo del cervello la percentuale di in_degree, di out_degree e di clustering del nodo NON isolato rispetto a tutti i nodi
nodi_con_corrispettivo   = calcoloPercentuali(G, nodi_con_corrispettivo, nodi_senza_corrispettivo)

#print(nodi_senza_corrispettivo)
#print("################################################################################")
#print(nodi_con_corrispettivo)

#calcolo per i nodi senza corrispettivo il valore medio dell'indegree, dell'outdegree e del coefficiente di clustering e nè calcolo per ognuno anche la percentuale
valori_medi_isolati     = calcoloValoriMedi(G, nodi_senza_corrispettivo, nodi_con_corrispettivo)
#calcolo per i nodi con corrispettivo il valore medio dell'indegree, dell'outdegree e del coefficiente di clustering e nè calcolo per ognuno anche la percentuale
valori_medi_non_isolati = calcoloValoriMedi(G, nodi_con_corrispettivo, nodi_senza_corrispettivo)

#print(valori_medi_isolati)
#print("################################################################################")
#print(valori_medi_non_isolati)

# %%
disegnaGraficiAccoppiati(nodi_senza_corrispettivo, valori_medi_isolati, "Nodi senza corrispettivo", nodi_con_corrispettivo, valori_medi_non_isolati, "Nodi con corrispettivo")

# %%
print(nodi_senza_corrispettivo)


# %%
def ottieniMotifPerNodo(insieme):
    opzioni   = ['in', 'out']
    posizione = {'in': 0,'out': 1}
    df_motif = pd.DataFrame({'root_id': [], 'motif': []})
    
    for index, nodo_in_esame in insieme.iterrows():
        motif = ['', '']
        #print(nodo_in_esame)
        for opzione in opzioni:
            nodi_vicini = eval("np.array(list(G." + opzione + "_edges(nodo_in_esame['root_id'])))")
            exec("df_" + opzione + " = pd.DataFrame({'root_id': [], 'super_class': []})")
            for nodo in nodi_vicini:
                nodo_area = cervello.loc[cervello['root_id'] == nodo[posizione[opzione]]]['super_class']
                new_row = {'root_id': nodo[posizione[opzione]], 'super_class': str(nodo_area.item())}
                exec("df_" + opzione + " = pd.concat([df_" + opzione + ", pd.DataFrame([new_row])], ignore_index=True)")
                        
            #print(eval("df_" + opzione))
            
            exec("df2_" + opzione + " = pd.DataFrame({'super_class': [], 'times': []})")
            
            aree = list(dict.fromkeys(eval("df_" + opzione)['super_class']))
            aree.sort()
            for area in aree:
                contatore_area = list(eval("df_" + opzione)['super_class']).count(area)
                new_row = {'super_class': area, 'times': contatore_area}
                exec("df2_" + opzione + " = pd.concat([df2_" + opzione + ", pd.DataFrame([new_row])], ignore_index=True)")
            #print(eval("df2_" + opzione))
            motif[posizione[opzione]] = list(eval("df2_" + opzione)['super_class'])
            motif[posizione[opzione]] = '-'.join(motif[posizione[opzione]])
            #print(motif[posizione[opzione]])
        motif_completo = ''
        
        if motif[0]:
            motif_completo = motif[0]
        
        motif_completo = motif_completo + ' # ' + nodo_in_esame['super_class'] + ' # '
        
        if motif[1]:
            motif_completo = motif_completo + motif[1]
        
        new_row = {'root_id': nodo_in_esame['root_id'], 'motif': motif_completo}
        df_motif = pd.concat([df_motif, pd.DataFrame([new_row])], ignore_index=True)
    return df_motif


# %%
def contaOccorrenzeMotif(insieme):
    df_times_motif = pd.DataFrame({'motif': [], 'times': []})
    
    motifs = list(dict.fromkeys(insieme['motif']))
    motifs.sort()
    for motif in motifs:
        contatore_motif = list(insieme['motif']).count(motif)
        new_row = {'motif': motif, 'times': contatore_motif}
        df_times_motif = pd.concat([df_times_motif, pd.DataFrame([new_row])], ignore_index=True)
    
    df_times_motif = df_times_motif.sort_values(by= ['times', 'motif'], ascending=[False, True])
    return df_times_motif


# %%
def graficoMotif(insieme, titolo, soglia):
    
    insieme = insieme.drop(insieme[insieme.times <= soglia].index)
    insieme = insieme.sort_values(by= ['times', 'motif'], ascending=[False, True])
    
    plt.figure(figsize=(25, 9))
    sns.barplot(x=insieme["motif"].tolist(), y=insieme["times"].tolist())
    plt.xticks(rotation=80, fontsize=7)
    plt.xlabel("Motif", fontsize=13)
    plt.ylabel("Frequenza", fontsize=13)
    plt.title(titolo, fontsize=16)
    #plt.tight_layout()
    plt.savefig('adulta/images/' + titolo + '.png', format= 'png', bbox_inches='tight')
    plt.show() 


# %%
df_motif_nodi_senza_corrispettivo       = ottieniMotifPerNodo(nodi_senza_corrispettivo)
df_times_motif_nodi_senza_corrispettivo = contaOccorrenzeMotif(df_motif_nodi_senza_corrispettivo)

df_motif_nodi_con_corrispettivo       = ottieniMotifPerNodo(nodi_con_corrispettivo)
df_times_motif_nodi_con_corrispettivo = contaOccorrenzeMotif(df_motif_nodi_con_corrispettivo)

df_motif_nodi_senza_corrispettivo.to_csv('adulta/csv/Elenco dei motif dei nodi senza corrispettivo.csv', index=False)
df_times_motif_nodi_senza_corrispettivo.to_csv('adulta/csv/Numero occorrenze dei motif dei nodi senza corrispettivo.csv', index=False)

df_motif_nodi_con_corrispettivo.to_csv('adulta/csv/Elenco dei motif dei nodi con corrispettivo.csv', index=False)
df_times_motif_nodi_con_corrispettivo.to_csv('adulta/csv/Numero occorrenze dei motif dei nodi con corrispettivo.csv', index=False)

# %%
graficoMotif(pd.read_csv('adulta/csv/Numero occorrenze dei motif dei nodi senza corrispettivo.csv'), 'Grafico a barre della frequenza per motif dei nodi senza corrispettivo', 12)
graficoMotif(pd.read_csv('adulta/csv/Numero occorrenze dei motif dei nodi con corrispettivo.csv'), 'Grafico a barre della frequenza per motif dei nodi con corrispettivo', 220)

# %%
aree = list(dict.fromkeys(cervello['super_class']))
aree


# %%

# %%
def ottieniAreaNodo(motif, times):
    index1 = motif.find("#")
    index2 = motif.find("#", index1 + 1)
    nodi_con_archi_entranti = motif[0:index1]
    nodi_con_archi_uscenti  = motif[index2+2:len(motif)]
    area_nodo = motif[index1+2:index2-1]

    aree = list(dict.fromkeys(cervello['super_class']))
    aree.sort()
    
    df = pd.DataFrame()
    for area in aree:
        contatore_area = (nodi_con_archi_entranti).count(area)
        if contatore_area != 0:
            new_row = {'source': ' ' + area, 'target': area_nodo, 'value': times}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    for area in aree:
        contatore_area = (nodi_con_archi_uscenti).count(area)
        if contatore_area != 0:
            new_row = {'source': area_nodo, 'target': area + ' ', 'value': times}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    return df

def disegnaSankey(insieme, titolo):
    sankey = pd.DataFrame()
    for index, nodo in insieme.iterrows():
        elemento = ottieniAreaNodo(nodo['motif'], nodo['times'])
        sankey = pd.concat([sankey, elemento], ignore_index=True, sort=False)
    
    
    aree = list(dict.fromkeys(cervello['super_class']))
    aree_nodi_entranti = [" " + area for area in aree]
    aree_nodi_uscenti  = [area + " " for area in aree]
    aree = aree + aree_nodi_entranti + aree_nodi_uscenti
    

    array_colori = ['#007FFF', '#FF6F61', '#228B22', '#FFDB58', '#967BB6', '#008080', '#FA8072', '#708090', '#DAA520']
    colors = []
    colors += array_colori 
    colors += array_colori 
    colors += array_colori 
    
    df = pd.DataFrame()
    df['aree']   = aree
    df['colors'] = colors

    df.sort_values(by= ['aree'], ascending=[False])

    for index, nodo in sankey.iterrows():
        cont = 0
        for area in df['aree']:
           if sankey.at[index, 'source'] == area:
               sankey.at[index, 'source'] = cont
               
           if sankey.at[index, 'target'] == area:
               sankey.at[index, 'target'] = cont
           cont += 1
    sankey2 = pd.DataFrame()
    for index, nodo in sankey.iterrows():
        valori_duplicati = sankey.drop(sankey[(sankey.source != nodo['source']) & (sankey.target != nodo['target'])].index)
        new_row = {'source': nodo['source'], 'target': nodo['target'], 'value': valori_duplicati['value'].sum()}
        sankey2 = pd.concat([sankey2, pd.DataFrame([new_row])], ignore_index=True)
    sankey2 = sankey2.drop_duplicates()

    fig = go.Figure(go.Sankey(
        arrangement='snap',
        node=dict(
            label=df['aree'],
            align='left',
            color=df['colors'],
        ),
        link=dict(
            arrowlen=15,
            source=sankey2['source'],
            target=sankey2['target'],
            value=sankey2['value'] 
        )
    ))
    fig.update_layout(title_text=titolo,  font_size=10, title_x=0.5)
    #pio.write_image(fig, "larva/images/" + titolo + ".png") 
    plot(fig, filename="adulta/images/" + titolo + ".html", image='png')
    fig.show()

# %%
df_times_motif_nodi_senza_corrispettivo = pd.read_csv('adulta/csv/Numero occorrenze dei motif dei nodi senza corrispettivo.csv')
df_times_motif_nodi_con_corrispettivo   = pd.read_csv('adulta/csv/Numero occorrenze dei motif dei nodi con corrispettivo.csv')

# %%
disegnaSankey(df_times_motif_nodi_senza_corrispettivo ,"Diagramma sankey dei motif dei nodi senza corrispettivo")

# %%
disegnaSankey(df_times_motif_nodi_con_corrispettivo ,"Diagramma sankey dei motif dei nodi con corrispettivo")

# %%
