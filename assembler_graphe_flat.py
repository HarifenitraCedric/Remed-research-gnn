import torch
import numpy as np
import pandas as pd
import json
from torch_geometric.data import HeteroData

print("=== [Orchestrateur] Assemblage du graphe ===")
data = HeteroData()

nodes_df = pd.read_csv("nodes_reel.csv")
patients_df = nodes_df[nodes_df['type'].str.lower() == 'patient'].reset_index(drop=True)
med_df = nodes_df[nodes_df['type'].str.lower().str.contains('med|drug', na=False)].reset_index(drop=True)

data['patient'].num_nodes = len(patients_df)
data['medicament'].num_nodes = len(med_df)

patient_to_idx = {str(k): int(v) for k, v in
                   pd.Series(patients_df.index.values, index=patients_df['node_id']).to_dict().items()}
med_to_idx = {str(k): int(v) for k, v in
              pd.Series(med_df.index.values, index=med_df['atc']).to_dict().items()}
idx_to_med = {v: k for k, v in med_to_idx.items()}

data['medicament'].node_id = med_df['atc'].tolist()
data['patient'].node_id = patients_df['node_id'].tolist()
data['medicament'].mapping_cis_to_idx = med_to_idx

print(f"-> Nœuds : {data['patient'].num_nodes} patients, {data['medicament'].num_nodes} médicaments")

edges_df = pd.read_csv("edges_reel.csv")
edges_df['target_clean'] = edges_df['target'].astype(str).str.replace('^DRUG_', '', regex=True)
edges_df['p_idx'] = edges_df['source'].map(patient_to_idx)
edges_df['m_idx'] = edges_df['target_clean'].map(med_to_idx)
valid_edges = edges_df.dropna(subset=['p_idx', 'm_idx'])

p_indices = valid_edges['p_idx'].astype(int).values
m_indices = valid_edges['m_idx'].astype(int).values
data['patient', 'a_prescrit', 'medicament'].edge_index = torch.tensor(
    np.array([p_indices, m_indices]), dtype=torch.long
)
print(f"-> Prescriptions : {data['patient', 'a_prescrit', 'medicament'].edge_index.shape[1]} arêtes")

edge_index_med = np.load("med_interactions_edge_index.npy")
edge_attr_med = np.load("med_interactions_edge_attr.npy")
data['medicament', 'interagit_avec', 'medicament'].edge_index = torch.from_numpy(edge_index_med).long()
data['medicament', 'interagit_avec', 'medicament'].edge_attr = torch.from_numpy(edge_attr_med).float()
print(f"-> Interactions : {data['medicament', 'interagit_avec', 'medicament'].edge_index.shape[1]} arêtes")

torch.save(data, "graphe_heterogene_complet.pt")

with open("mappings_partiels_assemblage.json", "w", encoding="utf-8") as f:
    json.dump({"med_to_idx": med_to_idx, "idx_to_med": idx_to_med, "patient_to_idx": patient_to_idx},
               f, ensure_ascii=False, indent=2)

print("[Succès] graphe_heterogene_complet.pt sauvegardé.")
