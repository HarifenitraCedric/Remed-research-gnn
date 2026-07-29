import pandas as pd
import numpy as np
import os

# 1. Charger les nœuds pour créer le dictionnaire de mapping (ATC -> Index numérique)
if os.path.exists("nodes_reel.csv"):
    nodes_df = pd.read_csv("nodes_reel.csv")
else:
    nodes_df = pd.read_csv("03_pipeline_symbolique/nodes_reel.csv")
# On isole les lignes de type médicament pour mapper leur code ATC à leur index de ligne
med_df = nodes_df[nodes_df['type'].str.lower().str.contains('med|drug', na=False)].reset_index(drop=True)
atc_to_idx = pd.Series(med_df.index.values, index=med_df['atc']).to_dict()

print(f"Dictionnaire de mapping créé pour {len(atc_to_idx)} médicaments.")
def lire_csv_safe(chemin_relatif):
    """Cherche d'abord à la racine du dossier d'exécution (_run_pipeline),

    sinon tente le chemin relatif d'origine.
    """
    nom_fichier = os.path.basename(chemin_relatif)
    if os.path.exists(nom_fichier):
        return pd.read_csv(nom_fichier)
    elif os.path.exists(chemin_relatif):
        return pd.read_csv(chemin_relatif)
    else:
        raise FileNotFoundError(
            f"Impossible de trouver {nom_fichier} ou {chemin_relatif}"
        )

# 2. Charger les interactions cibles

interactions_df = lire_csv_safe("99_a_verifier/interactions_thesaurus_global.csv")

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

def sauvegarder_npy_safe(chemin_relatif, array):
    """Enregistre le tableau NumPy à la racine de _run_pipeline si présent,

    sinon crée le sous-dossier nécessaire en mode autonome.
    """
    nom_fichier = os.path.basename(chemin_relatif)

    if os.path.basename(os.getcwd()) == "_run_pipeline" or not os.path.exists(
        os.path.dirname(chemin_relatif)
    ):
        np.save(nom_fichier, array)
    else:
        os.makedirs(os.path.dirname(chemin_relatif), exist_ok=True)
        np.save(chemin_relatif, array)

# Sauvegarde des tenseurs au format numpy pour ton script GNN principal
sauvegarder_npy_safe(
    "99_a_verifier/preparer_aretes_gnn/med_interactions_edge_index.npy",
    edge_index,
)
sauvegarder_npy_safe(
    "99_a_verifier/preparer_aretes_gnn/med_interactions_edge_attr.npy",
    edge_attr,
)
print("\nTenseurs sauvegardés avec succès !")