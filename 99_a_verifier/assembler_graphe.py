import torch
import numpy as np
import pandas as pd
import json
import os
from torch_geometric.data import HeteroData

print("=== Assemblage du Graphe Hétérogène REMED-RESEARCH ===")

# 1. Initialisation de l'objet hétérogène PyG
data = HeteroData()

# 2. Chargement des nœuds et création des dictionnaires de mapping
if os.path.exists("nodes_reel.csv"):
    nodes_df = pd.read_csv("nodes_reel.csv")
else:
    nodes_df = pd.read_csv("03_pipeline_symbolique/nodes_reel.csv")

patients_df = nodes_df[nodes_df['type'].str.lower() == 'patient'].reset_index(drop=True)
med_df = nodes_df[nodes_df['type'].str.lower().str.contains('med|drug', na=False)].reset_index(drop=True)

data['patient'].num_nodes = len(patients_df)
data['medicament'].num_nodes = len(med_df)

# Dictionnaires de correspondance (clés/valeurs castées en types JSON-safe)
patient_to_idx = {str(k): int(v) for k, v in
                   pd.Series(patients_df.index.values, index=patients_df['node_id']).to_dict().items()}
med_to_idx = {str(k): int(v) for k, v in
              pd.Series(med_df.index.values, index=med_df['atc']).to_dict().items()}
idx_to_med = {v: k for k, v in med_to_idx.items()}

# --- Traçabilité pour le Dashboard/API ---
data['medicament'].node_id = med_df['atc'].tolist()
data['patient'].node_id = patients_df['node_id'].tolist()

# --- CORRECTION : nom d'attribut attendu par train_heterognn_v3.py
# (exporter_artefacts_corrige.py cherche spécifiquement 'mapping_cis_to_idx',
# pas 'node_id' -- sans ce nom exact il retombe sur le mapping générique
# MED_0/MED_1/... qu'on cherche justement à éviter)
data['medicament'].mapping_cis_to_idx = med_to_idx

print(f"-> Nœuds configurés : {data['patient'].num_nodes} Patients et {data['medicament'].num_nodes} Médicaments.")

# 3. Chargement et vectorisation des liens de prescription (edges_reel.csv)
if os.path.exists("edges_reel.csv"):
    edges_df = pd.read_csv("edges_reel.csv")
else:
    edges_df = pd.read_csv("03_pipeline_symbolique/edges_reel.csv")

edges_df['target_clean'] = edges_df['target'].astype(str).str.replace('^DRUG_', '', regex=True)
edges_df['p_idx'] = edges_df['source'].map(patient_to_idx)
edges_df['m_idx'] = edges_df['target_clean'].map(med_to_idx)

valid_edges = edges_df.dropna(subset=['p_idx', 'm_idx'])
n_ignorees = len(edges_df) - len(valid_edges)
if n_ignorees:
    print(f"⚠️  {n_ignorees} arête(s) ignorée(s) (patient ou médicament introuvable)")

p_indices = valid_edges['p_idx'].astype(int).values
m_indices = valid_edges['m_idx'].astype(int).values

data['patient', 'a_prescrit', 'medicament'].edge_index = torch.tensor([p_indices, m_indices], dtype=torch.long)

print(f"-> Liens ajoutés : {data['patient', 'a_prescrit', 'medicament'].edge_index.shape[1]} arêtes de prescription.")

# 4. Injection des arêtes médicales ANSM (Médicament <-> Médicament)
# --- CORRECTION : chemin sans le sous-dossier 'preparer_aretes_gnn/' qui
# n'existe pas dans l'arborescence réorganisée -- les .npy sont directement
# dans 99_a_verifier/
# Fonction helper pour charger les .npy de manière flexible
def charger_npy_safe(chemin_relatif):
    nom_fichier = os.path.basename(chemin_relatif)
    if os.path.exists(nom_fichier):
        return np.load(nom_fichier)
    elif os.path.exists(chemin_relatif):
        return np.load(chemin_relatif)
    else:
        raise FileNotFoundError(
            f"Impossible de trouver {nom_fichier} ou {chemin_relatif}"
        )


edge_index_med = charger_npy_safe(
    "99_a_verifier/preparer_aretes_gnn/med_interactions_edge_index.npy"
)
edge_attr_med = charger_npy_safe(
    "99_a_verifier/preparer_aretes_gnn/med_interactions_edge_attr.npy"
)

data['medicament', 'interagit_avec', 'medicament'].edge_index = torch.from_numpy(edge_index_med).long()
data['medicament', 'interagit_avec', 'medicament'].edge_attr = torch.from_numpy(edge_attr_med).float()

print(f"-> Relations ajoutées : {data['medicament', 'interagit_avec', 'medicament'].edge_index.shape[1]} arêtes d'interactions médicamenteuses.")

# 5. Sauvegarde du graphe complet et des mappings JSON
# --- CORRECTION : dossier 04_gnn/ (cohérent avec l'arborescence), créé s'il
# n'existe pas encore, au lieu de 'data/' qui n'existe pas
# Exemple pour torch.save dans assembler_graphe.py ou exporter_artefacts.py
def sauvegarder_torch_safe(chemin_relatif, obj):
    nom_fichier = os.path.basename(chemin_relatif)
    if os.path.basename(os.getcwd()) == "_run_pipeline" or not os.path.exists(
        os.path.dirname(chemin_relatif)
    ):
        torch.save(obj, nom_fichier)
    else:
        os.makedirs(os.path.dirname(chemin_relatif), exist_ok=True)
        torch.save(obj, chemin_relatif)

sauvegarder_torch_safe("04_gnn/graphe_heterogene_complet.pt", data)

mappings_remed = {
    "med_to_idx": med_to_idx,
    "idx_to_med": idx_to_med,
    "patient_to_idx": patient_to_idx,
}

# --- CORRECTION : le dict était construit mais jamais écrit sur disque ---
def sauvegarder_json_safe(chemin_relatif, obj):
  nom_fichier = os.path.basename(chemin_relatif)
  target_path = (
      nom_fichier
      if os.path.basename(os.getcwd()) == "_run_pipeline"
      or not os.path.exists(os.path.dirname(chemin_relatif))
      else chemin_relatif
  )
  if os.path.dirname(target_path):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
  with open(target_path, "w", encoding="utf-8") as f:
    json.dump(obj, f, ensure_ascii=False, indent=2)


sauvegarder_json_safe("04_gnn/mappings_partiels_assemblage.json", mappings_remed)

print("\n[Succès] Graphe hétérogène et mappings partiels sauvegardés avec succès !")
print("ℹ️  Ce fichier 'mappings_partiels_assemblage.json' est un mapping technique "
      "(index <-> ATC/patient), distinct de 'mappings_remed.json' final qui sera "
      "régénéré par train_heterognn_v3.py avec en plus l'ordre des classes de gravité.")
print(data)