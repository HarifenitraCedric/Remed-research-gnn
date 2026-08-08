"""
Export des artefacts de déploiement (poids + mappings), VERSION FINALE NETTOYÉE.

Inclusions & Corrections :
1. Résolution dynamique des chemins de fichiers (compatibilité avec exécution 
   depuis la racine du projet ou depuis le sous-dossier 04_gnn).
2. Alignment des codes ATC sur la base de données réelle (medications.csv, 179 codes)
   au lieu d'extraire la liste obsolète du fichier .pt (94 codes).
3. Détection automatique du mapping de gravité (ordre des 4 classes d'interactions ANSM).
"""

import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import HeteroConv, SAGEConv


# ---------------------------------------------------------------------------
# Architecture du Réseau de Neurones GNN
# ---------------------------------------------------------------------------
class RemedHeteroGNN(nn.Module):
    def __init__(self, num_patients, num_meds, embed_dim, hidden_dim, out_dim):
        super().__init__()
        self.patient_emb = nn.Embedding(num_patients, embed_dim)
        self.med_emb = nn.Embedding(num_meds, embed_dim)

        self.conv1 = HeteroConv({
            ('patient', 'a_prescrit', 'medicament'): SAGEConv((-1, -1), hidden_dim),
            ('medicament', 'rev_a_prescrit', 'patient'): SAGEConv((-1, -1), hidden_dim),
            ('medicament', 'interagit_avec', 'medicament'): SAGEConv((-1, -1), hidden_dim),
        }, aggr='sum')

        self.conv2 = HeteroConv({
            ('patient', 'a_prescrit', 'medicament'): SAGEConv((-1, -1), out_dim),
            ('medicament', 'rev_a_prescrit', 'patient'): SAGEConv((-1, -1), out_dim),
            ('medicament', 'interagit_avec', 'medicament'): SAGEConv((-1, -1), out_dim),
        }, aggr='sum')

        self.decoder = nn.Sequential(
            nn.Linear(out_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, 4),
        )

    def encode(self, x_dict, edge_index_dict):
        h_dict = self.conv1(x_dict, edge_index_dict)
        h_dict = {key: F.relu(x) for key, x in h_dict.items()}
        return self.conv2(h_dict, edge_index_dict)

    def decode(self, z_med, edge_label_index):
        x_src = z_med[edge_label_index[0]]
        x_dst = z_med[edge_label_index[1]]
        return self.decoder(torch.cat([x_src, x_dst], dim=-1))

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encode(x_dict, edge_index_dict)
        return self.decode(z_dict['medicament'], edge_label_index)


# ---------------------------------------------------------------------------
# Utilitaires de gestion robuste des fichiers et chemins
# ---------------------------------------------------------------------------
def trouver_chemin_fichier(nom_ou_chemin_relatif):
    """
    Tente de localiser un fichier en testant plusieurs chemins plausibles 
    selon le sous-dossier d'exécution courant.
    """
    chemins_a_tester = [
        nom_ou_chemin_relatif,
        os.path.basename(nom_ou_chemin_relatif),
        os.path.join("..", nom_ou_chemin_relatif),
        os.path.join("04_gnn", os.path.basename(nom_ou_chemin_relatif)),
        os.path.join("..", "02_interactions_ansm", "créer_interaction_graphe", os.path.basename(nom_ou_chemin_relatif)),
    ]
    for p in chemins_a_tester:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Impossible de localiser le fichier : {nom_ou_chemin_relatif}")


def charger_torch_safe(chemin_relatif):
    target = trouver_chemin_fichier(chemin_relatif)
    return torch.load(target, weights_only=False)


def lire_csv_safe(chemin_relatif):
    target = trouver_chemin_fichier(chemin_relatif)
    return pd.read_csv(target)


def sauvegarder_torch_safe(chemin_relatif, obj):
    dossier = os.path.dirname(chemin_relatif)
    if dossier and not os.path.exists(dossier):
        os.makedirs(dossier, exist_ok=True)
    torch.save(obj, chemin_relatif)


def sauvegarder_json_safe(chemin_relatif, obj):
    dossier = os.path.dirname(chemin_relatif)
    if dossier and not os.path.exists(dossier):
        os.makedirs(dossier, exist_ok=True)
    with open(chemin_relatif, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 1. Chargement du graphe d'origine
# ---------------------------------------------------------------------------
data = charger_torch_safe("graphe_heterogene_complet.pt")
data_bidirect = T.ToUndirected()(data)
rel = ('medicament', 'interagit_avec', 'medicament')
labels_complet = torch.argmax(data_bidirect[rel].edge_attr, dim=-1)

# ---------------------------------------------------------------------------
# 2. Détection automatique de l'ordre réel des classes ANSM
# ---------------------------------------------------------------------------
proportions_reelles = {
    c: (labels_complet == c).sum().item() / len(labels_complet) for c in range(4)
}

thesaurus = lire_csv_safe("interactions_thesaurus_global.csv")
proportions_attendues = thesaurus["gravite"].value_counts(normalize=True).to_dict()

print("=== Détection automatique de l'ordre des classes ===")
print(f"{'Classe':<8}{'% observé':<12}{'Gravité correspondante':<28}{'% attendu':<10}")

mapping_detecte = {}
gravites_deja_prises = set()
for classe_idx, prop_obs in sorted(proportions_reelles.items(), key=lambda x: x[0]):
    meilleure_gravite, meilleur_ecart = None, float("inf")
    for gravite, prop_att in proportions_attendues.items():
        if gravite in gravites_deja_prises:
            continue
        ecart = abs(prop_obs - prop_att)
        if ecart < meilleur_ecart:
            meilleur_ecart, meilleure_gravite = ecart, gravite
    mapping_detecte[str(classe_idx)] = meilleure_gravite
    gravites_deja_prises.add(meilleure_gravite)
    print(f"{classe_idx:<8}{prop_obs:<12.1%}{meilleure_gravite:<28}{proportions_attendues[meilleure_gravite]:<10.1%}")

print(f"\nMapping détecté : {mapping_detecte}\n")

# ---------------------------------------------------------------------------
# 3. Alignement des médicaments depuis medications.csv (179 codes ATC)
# ---------------------------------------------------------------------------
meds_df = lire_csv_safe("medications.csv")
atc_codes = meds_df['code_atc'].dropna().astype(str).str.strip().str.upper().unique()
atc_codes = sorted([code for code in atc_codes if code and code != 'NAN'])

# Génération du dictionnaire complet med_to_idx
med_mapping = {atc_code: idx for idx, atc_code in enumerate(atc_codes)}
num_meds_total = len(med_mapping)

print(f"✓ Dictionnaire de médicaments aligné sur medications.csv : {num_meds_total} codes ATC retenus.")

# ---------------------------------------------------------------------------
# 4. Entraînement final du modèle PyTorch
# ---------------------------------------------------------------------------
print("🔄 Entraînement final du modèle sur l'ensemble des données...")
torch.manual_seed(42)

model_final = RemedHeteroGNN(
    num_patients=int(data_bidirect['patient'].num_nodes),
    num_meds=num_meds_total,
    embed_dim=64,
    hidden_dim=32,
    out_dim=16,
)

comptes = torch.bincount(labels_complet, minlength=4).float()
poids = (1.0 / comptes.clamp(min=1))
poids = poids / poids.sum() * 4
criterion = nn.CrossEntropyLoss(weight=poids)
optimizer = torch.optim.Adam(model_final.parameters(), lr=0.01, weight_decay=5e-4)

x_init = {
    'patient': model_final.patient_emb.weight,
    'medicament': model_final.med_emb.weight
}

model_final.train()
for epoch in range(1, 101):
    optimizer.zero_grad()
    predictions = model_final(x_init, data_bidirect.edge_index_dict, data_bidirect[rel].edge_index)
    loss = criterion(predictions, labels_complet)
    loss.backward()
    optimizer.step()

# Sauvegarde des poids du modèle
sauvegarder_torch_safe("remed_gnn_weights.pt", model_final.state_dict())
print("✓ Poids du modèle sauvegardés : remed_gnn_weights.pt")

# ---------------------------------------------------------------------------
# 5. Exportation du dictionnaire Mappings au format JSON
# ---------------------------------------------------------------------------
mappings_payload = {
    "med_to_idx": med_mapping,
    "idx_to_class": mapping_detecte,
    "metadata": {
        "num_patients": int(data['patient'].num_nodes),
        "num_medicaments": num_meds_total,
        "embed_dim": 64,
        "hidden_dim": 32,
        "out_dim": 16,
    },
}

sauvegarder_json_safe("mappings_remed.json", mappings_payload)
print("✓ Mappings JSON sauvegardés : mappings_remed.json")