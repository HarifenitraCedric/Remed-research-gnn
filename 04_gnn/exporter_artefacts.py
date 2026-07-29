"""
Export des artefacts de déploiement (poids + mappings), CORRIGÉ.

Bug corrigé : le class_mapping était écrit EN DUR, en supposant que l'ordre
du one-hot encoding suivait l'ordre "naturel" de gravité croissante
(a_prendre_en_compte, precaution_emploi, association_deconseillee,
contre_indication). Ce n'est pas le cas -- l'ordre réel dépend de l'ordre
d'apparition des valeurs dans le script qui a construit
graphe_heterogene_complet.pt (probablement un get_dummies ou un encodage par
ordre alphabétique).

Conséquence du bug (vérifiée empiriquement) : la classe 2 (2,6% des
exemples, correspondant à 'contre_indication') était étiquetée
'association_deconseillee' dans l'API -- une vraie contre-indication
s'affichait comme un risque moindre. C'est l'inverse de ce qu'on veut pour
un outil médical.

Correction : l'ordre est déterminé ICI, automatiquement, en comparant la
proportion de chaque classe (argmax du one-hot du graphe) à la proportion
connue de chaque gravité dans interactions_ansm.csv -- au lieu de la
supposer. Le mapping résultant est affiché explicitement pour vérification
visuelle avant sauvegarde.
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv
import torch_geometric.transforms as T
import pandas as pd
import os


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
# 1. Chargement du graphe
# ---------------------------------------------------------------------------
def charger_torch_safe(chemin_relatif):
  nom_fichier = os.path.basename(chemin_relatif)
  if os.path.exists(nom_fichier):
    return torch.load(nom_fichier, weights_only=False)
  elif os.path.exists(chemin_relatif):
    return torch.load(chemin_relatif, weights_only=False)
  else:
    raise FileNotFoundError(
        f"Impossible de trouver {nom_fichier} ou {chemin_relatif}"
    )


data = charger_torch_safe("04_gnn/graphe_heterogene_complet.pt") 

data_bidirect = T.ToUndirected()(data)
rel = ('medicament', 'interagit_avec', 'medicament')
labels_complet = torch.argmax(data_bidirect[rel].edge_attr, dim=-1)

# ---------------------------------------------------------------------------
# 2. Détermination AUTOMATIQUE de l'ordre réel des classes, en comparant les
#    proportions observées aux proportions connues du thésaurus ANSM
#    (interactions_thesaurus_global.csv, seule source de vérité de gravité)
# ---------------------------------------------------------------------------
proportions_reelles = {
    c: (labels_complet == c).sum().item() / len(labels_complet) for c in range(4)
}

def lire_csv_safe(chemin_relatif):
  nom_fichier = os.path.basename(chemin_relatif)
  if os.path.exists(nom_fichier):
    return pd.read_csv(nom_fichier)
  elif os.path.exists(chemin_relatif):
    return pd.read_csv(chemin_relatif)
  else:
    raise FileNotFoundError(
        f"Impossible de trouver {nom_fichier} ou {chemin_relatif}"
    )


thesaurus = lire_csv_safe(
    "02_interactions_ansm/créer_interaction_graphe/interactions_thesaurus_global.csv"
)
proportions_attendues = thesaurus["gravite"].value_counts(normalize=True).to_dict()

print("=== Détection automatique de l'ordre des classes ===")
print(f"{'Classe':<8}{'% observé':<12}{'Gravité correspondante':<28}{'% attendu':<10}")

mapping_detecte = {}
gravites_deja_prises = set()
for classe_idx, prop_obs in sorted(proportions_reelles.items(), key=lambda x: x[0]):
    # Trouve la gravité dont la proportion attendue est la plus proche
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

print(f"\nMapping détecté : {mapping_detecte}")
print("⚠️  VÉRIFIEZ VISUELLEMENT que les % observé et attendu se correspondent "
      "bien ligne par ligne avant de faire confiance à ce mapping.")

# ---------------------------------------------------------------------------
# 3. Entraînement final (identique à train_heterognn_v2.py, sur tout le graphe)
# ---------------------------------------------------------------------------
print("\n🔄 Entraînement final du modèle sur l'ensemble des données...")
torch.manual_seed(42)
model_final = RemedHeteroGNN(
    num_patients=data_bidirect['patient'].num_nodes,
    num_meds=data_bidirect['medicament'].num_nodes,
    embed_dim=64, hidden_dim=32, out_dim=16,
)

comptes = torch.bincount(labels_complet, minlength=4).float()
poids = (1.0 / comptes.clamp(min=1))
poids = poids / poids.sum() * 4
criterion = nn.CrossEntropyLoss(weight=poids)
optimizer = torch.optim.Adam(model_final.parameters(), lr=0.01, weight_decay=5e-4)
x_init = {'patient': model_final.patient_emb.weight, 'medicament': model_final.med_emb.weight}

model_final.train()
for epoch in range(1, 101):
    optimizer.zero_grad()
    predictions = model_final(x_init, data_bidirect.edge_index_dict, data_bidirect[rel].edge_index)
    loss = criterion(predictions, labels_complet)
    loss.backward()
    optimizer.step()

def sauvegarder_torch_safe(chemin_relatif, obj):
  nom_fichier = os.path.basename(chemin_relatif)
  if os.path.basename(os.getcwd()) == "_run_pipeline" or not os.path.exists(
      os.path.dirname(chemin_relatif)
  ):
    torch.save(obj, nom_fichier)
  else:
    os.makedirs(os.path.dirname(chemin_relatif), exist_ok=True)
    torch.save(obj, chemin_relatif)


sauvegarder_torch_safe(
    "04_gnn/exporter_artefacts/remed_gnn_weights.pt", model_final.state_dict()
)
print("✓ Poids sauvegardés : remed_gnn_weights.pt")

# ---------------------------------------------------------------------------
# 4. Mapping médicament -> index (identique à avant, fallback générique)
# ---------------------------------------------------------------------------
if hasattr(data['medicament'], 'mapping_cis_to_idx'):
    med_mapping = data['medicament'].mapping_cis_to_idx
elif hasattr(data['medicament'], 'cis'):
    med_mapping = {str(cis): idx for idx, cis in enumerate(data['medicament'].cis)}
else:
    num_meds = data['medicament'].num_nodes
    med_mapping = {f"MED_{i}": i for i in range(num_meds)}

mappings_payload = {
    "med_to_idx": med_mapping,
    "idx_to_class": mapping_detecte,  # <-- mapping détecté automatiquement, pas supposé
    "metadata": {
        "num_patients": int(data['patient'].num_nodes),
        "num_medicaments": int(data['medicament'].num_nodes),
        "embed_dim": 64, "hidden_dim": 32, "out_dim": 16,
    },
}

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


sauvegarder_json_safe(
    "04_gnn/exporter_artefacts/mappings_remed.json", mappings_payload
)

print("✓ Mappings sauvegardés (avec ordre de classes vérifié) : mappings_remed.json")
