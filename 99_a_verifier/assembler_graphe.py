import torch
import numpy as np
import pandas as pd
from torch_geometric.data import HeteroData

print("=== Assemblage du Graphe Hétérogène REMED-RESEARCH ===")

# 1. Initialisation de l'objet hétérogène PyG
data = HeteroData()

# 2. Chargement des nœuds et création des dictionnaires de mapping
nodes_df = pd.read_csv("nodes_reel.csv")

patients_df = nodes_df[nodes_df['type'].str.lower() == 'patient'].reset_index(drop=True)
med_df = nodes_df[nodes_df['type'].str.lower().str.contains('med|drug', na=False)].reset_index(drop=True)

data['patient'].num_nodes = len(patients_df)
data['medicament'].num_nodes = len(med_df)

# Dictionnaires de correspondance
patient_to_idx = pd.Series(patients_df.index.values, index=patients_df['node_id']).to_dict()
med_to_idx = pd.Series(med_df.index.values, index=med_df['atc']).to_dict()

print(f"-> Nœuds configurés : {data['patient'].num_nodes} Patients et {data['medicament'].num_nodes} Médicaments.")

# 3. Chargement et conversion des liens de prescription (edges_reel.csv)
edges_df = pd.read_csv("edges_reel.csv")

p_indices = []
m_indices = []

for _, row in edges_df.iterrows():
    # Extraction de l'ID patient
    p_id = row['source']
    
    # Nettoyage du code médicament : on enlève 'DRUG_' s'il est présent
    m_id_raw = str(row['target'])
    m_id = m_id_raw.replace('DRUG_', '') if m_id_raw.startswith('DRUG_') else m_id_raw
    
    # Récupération des index numériques
    p_idx = patient_to_idx.get(p_id)
    m_idx = med_to_idx.get(m_id)
    
    if p_idx is not None and m_idx is not None:
        p_indices.append(p_idx)
        m_indices.append(m_idx)

# Ajout des arêtes de prescription (Patient -> Médicament)
data['patient', 'a_prescrit', 'medicament'].edge_index = torch.tensor([p_indices, m_indices], dtype=torch.long)

print(f"-> Liens ajoutés : {data['patient', 'a_prescrit', 'medicament'].edge_index.shape[1]} arêtes de prescription.")

# 4. Injection des arêtes médicales ANSM (Médicament <-> Médicament)
edge_index_med = np.load("med_interactions_edge_index.npy")
edge_attr_med = np.load("med_interactions_edge_attr.npy")

data['medicament', 'interagit_avec', 'medicament'].edge_index = torch.from_numpy(edge_index_med).long()
data['medicament', 'interagit_avec', 'medicament'].edge_attr = torch.from_numpy(edge_attr_med).float()

print(f"-> Relations ajoutées : {data['medicament', 'interagit_avec', 'medicament'].edge_index.shape[1]} arêtes d'interactions médicamenteuses.")

# 5. Sauvegarde du graphe complet
torch.save(data, "graphe_heterogene_complet.pt")
print("\n[Succès] Graphe hétérogène complet mis à jour et sauvegardé dans 'graphe_heterogene_complet.pt' !")
print(data)