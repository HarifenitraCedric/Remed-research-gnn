import torch
import pandas as pd
import numpy as np

# 1. Charger le graphe hétérogène existant
PATH_GRAPHE_IN = "graphe_heterogene_complet.pt"
PATH_GRAPHE_OUT = "graphe_heterogene_enrichi.pt"

data = torch.load(PATH_GRAPHE_IN, weights_only=False)
print("Chargement du graphe...")

# 2. Charger les données médicaments
df_meds = pd.read_csv("medications.csv")

# Récupérer l'ordre exact des CIS/IDs dans le graphe
mapping_cis_to_idx = data['medicament'].mapping_cis_to_idx
num_nodes = data['medicament'].num_nodes

print(f"Nombre de médicaments dans le graphe : {num_nodes}")

# 3. Extraire les classes ATC Niveau 1 (14 catégories : A, B, C, D, G, H, J, L, M, N, P, R, S, V)
CLASSES_ATC_L1 = ['A', 'B', 'C', 'D', 'G', 'H', 'J', 'L', 'M', 'N', 'P', 'R', 'S', 'V']
atc_l1_to_idx = {cat: i for i, cat in enumerate(CLASSES_ATC_L1)}

# Matrice de features (num_nodes, len(CLASSES_ATC_L1) + 1_flag_ophtalmique)
features_matrix = np.zeros((num_nodes, len(CLASSES_ATC_L1) + 1), dtype=np.float32)

# Remplir les features pour chaque médicament du graphe
for cis_or_id, node_idx in mapping_cis_to_idx.items():
    # Trouver la ligne correspondante dans df_meds
    row = df_meds[df_meds['id'].astype(str) == str(cis_or_id)]
    
    if not row.empty and pd.notna(row.iloc[0]['code_atc']):
        code_atc = str(row.iloc[0]['code_atc']).strip().upper()
        
        # 1. Flag ATC Niveau 1
        l1_letter = code_atc[0] if len(code_atc) > 0 else ''
        if l1_letter in atc_l1_to_idx:
            features_matrix[node_idx, atc_l1_to_idx[l1_letter]] = 1.0
            
        # 2. Flag spécifique Voie / Ophtalmique (Code S)
        if code_atc.startswith('S'):
            features_matrix[node_idx, -1] = 1.0  # Feature dédiée ophtalmique

# Convertir en Tenseur PyTorch et injecter dans 'medicament'
data['medicament'].x = torch.tensor(features_matrix, dtype=torch.float32)

print("\n✅ Features injectées avec succès !")
print(f"Shape de data['medicament'].x : {data['medicament'].x.shape}")

# Save
torch.save(data, PATH_GRAPHE_OUT)
print(f"Nouveau graphe sauvegardé sous : {PATH_GRAPHE_OUT}")