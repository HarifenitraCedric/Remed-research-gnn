import pandas as pd
import numpy as np

# 1. Charger les nœuds pour créer le dictionnaire de mapping (ATC -> Index numérique)
nodes_df = pd.read_csv("nodes_reel.csv")
# On isole les lignes de type médicament pour mapper leur code ATC à leur index de ligne
med_df = nodes_df[nodes_df['type'].str.lower().str.contains('med|drug', na=False)].reset_index(drop=True)
atc_to_idx = pd.Series(med_df.index.values, index=med_df['atc']).to_dict()

print(f"Dictionnaire de mapping créé pour {len(atc_to_idx)} médicaments.")

# 2. Charger les interactions cibles
interactions_df = pd.read_csv("interactions_thesaurus_global.csv")

# 3. Convertir les codes ATC en indices numériques
src_indices = []
dst_indices = []
gravites = []

for _, row in interactions_df.iterrows():
    idx1 = atc_to_idx.get(row['atc_1'])
    idx2 = atc_to_idx.get(row['atc_2'])
    
    # On s'assure que les deux codes existent bien dans notre sous-graphe
    if idx1 is not None and idx2 is not None:
        # Sens A -> B
        src_indices.append(idx1)
        dst_indices.append(idx2)
        gravites.append(row['gravite'])
        
        # Sens B -> A (Bidirectionnalité)
        src_indices.append(idx2)
        dst_indices.append(idx1)
        gravites.append(row['gravite'])

# 4. Encodage One-Hot des gravités
gravites_series = pd.Series(gravites)
one_hot_gravite = pd.get_dummies(gravites_series).astype(float)

print("\nRépartition des niveaux de gravité encodés (arêtes bidirectionnelles) :")
print(one_hot_gravite.sum())

# 5. Création des structures finales
edge_index = np.array([src_indices, dst_indices], dtype=np.int64)
edge_attr = one_hot_gravite.values

print(f"\nShape finale de edge_index : {edge_index.shape}")
print(f"Shape finale de edge_attr (features d'arêtes) : {edge_attr.shape}")

# Sauvegarde des tenseurs au format numpy pour ton script GNN principal
np.save("med_interactions_edge_index.npy", edge_index)
np.save("med_interactions_edge_attr.npy", edge_attr)
print("\nTenseurs sauvegardés avec succès !")