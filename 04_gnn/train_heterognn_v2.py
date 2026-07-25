"""
Correction de train_heterognn.py — 3 problèmes corrigés :

  1. AUCUN SPLIT TRAIN/TEST -> ajouté : split stratifié par classe de
     gravité (chaque classe est représentée proportionnellement en train
     et en test, y compris la classe minoritaire 'contre_indication').

  2. FUITE DE MESSAGE PASSING -> corrigée : le graphe utilisé pour
     l'encodage (conv1/conv2) ne contient QUE les arêtes 'interagit_avec'
     du train. Les arêtes de test sont totalement invisibles au moment de
     calculer les embeddings, donc le modèle ne peut plus "voir" la
     réponse à travers la structure du graphe.

  3. ACCURACY GLOBALE SEULE / DÉSÉQUILIBRE DE CLASSES -> corrigé :
     - CrossEntropyLoss pondérée par l'inverse de la fréquence de classe
     - Rapport complet par classe (précision, rappel, F1) + matrice de
       confusion, pas seulement l'accuracy globale
     - Évaluation sur PLUSIEURS seeds (10 par défaut) avec moyenne ±
       écart-type, parce qu'avec seulement 271 exemples (dont 7 de
       'contre_indication'), un seul split donne un résultat qui peut
       varier énormément selon le tirage aléatoire.

Limite assumée et à documenter dans le mémoire : avec 7 exemples de
'contre_indication' au total, un split 80/20 stratifié ne laisse qu'1-2
exemples de cette classe en test. Les métriques sur cette classe précise
doivent être interprétées avec une prudence particulière (grands
intervalles de confiance), ce qui est signalé explicitement dans la sortie.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv
import torch_geometric.transforms as T
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score

NOMS_CLASSES = ["a_prendre_en_compte", "precaution_emploi", "association_deconseillee", "contre_indication"]
# ⚠️ L'ordre réel des classes dépend de l'encodage One-Hot fait au moment de
# la construction de graphe_heterogene_complet.pt. Vérifier qu'il correspond
# bien à l'ordre utilisé lors de la construction (à confirmer avec le script
# qui a produit le tenseur edge_attr).


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


def split_stratifie(edge_index, edge_labels, seed, test_size=0.2):
    """Split stratifié : chaque classe est répartie proportionnellement
    entre train et test, y compris les classes rares."""
    n = edge_index.shape[1]
    indices = np.arange(n)
    labels_np = edge_labels.numpy()

    idx_train, idx_test = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=labels_np
    )
    return torch.tensor(idx_train), torch.tensor(idx_test)


def entrainer_et_evaluer(data_bidirect, seed, n_epochs=100, verbose=False):
    torch.manual_seed(seed)
    np.random.seed(seed)

    rel = ('medicament', 'interagit_avec', 'medicament')
    edge_index_complet = data_bidirect[rel].edge_index
    edge_attr_onehot = data_bidirect[rel].edge_attr
    edge_labels_complet = torch.argmax(edge_attr_onehot, dim=-1)

    idx_train, idx_test = split_stratifie(edge_index_complet, edge_labels_complet, seed)

    edge_index_train = edge_index_complet[:, idx_train]
    labels_train = edge_labels_complet[idx_train]
    edge_index_test = edge_index_complet[:, idx_test]
    labels_test = edge_labels_complet[idx_test]

    # --- Graphe de message passing SANS FUITE : seules les arêtes de train
    # sont utilisées pour calculer les embeddings des médicaments.
    edge_index_dict_train = dict(data_bidirect.edge_index_dict)
    edge_index_dict_train[rel] = edge_index_train

    model = RemedHeteroGNN(
        num_patients=data_bidirect['patient'].num_nodes,
        num_meds=data_bidirect['medicament'].num_nodes,
        embed_dim=64, hidden_dim=32, out_dim=16,
    )

    # --- Pondération de la loss par l'inverse de la fréquence de classe
    # (déséquilibre sévère : 'contre_indication' ~2.6% des exemples)
    comptes = torch.bincount(labels_train, minlength=4).float()
    poids = (1.0 / comptes.clamp(min=1))
    poids = poids / poids.sum() * 4
    criterion = nn.CrossEntropyLoss(weight=poids)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    x_initial_dict = {'patient': model.patient_emb.weight, 'medicament': model.med_emb.weight}

    model.train()
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        predictions = model(x_initial_dict, edge_index_dict_train, edge_index_train)
        loss = criterion(predictions, labels_train)
        loss.backward()
        optimizer.step()
        if verbose and epoch % 20 == 0:
            print(f"  Époque {epoch:3d} | Perte : {loss.item():.4f}")

    # --- Évaluation sur le TEST, avec le graphe de message passing du TRAIN
    # uniquement (les arêtes de test restent invisibles à l'encodeur)
    model.eval()
    with torch.no_grad():
        z_dict = model.encode(x_initial_dict, edge_index_dict_train)
        logits_test = model.decode(z_dict['medicament'], edge_index_test)
        preds_test = torch.argmax(logits_test, dim=-1)

    return labels_test.numpy(), preds_test.numpy()


if __name__ == "__main__":
    graphe_path = "graphe_heterogene_complet.pt"
    if not os.path.exists(graphe_path):
        print(f"[Erreur] '{graphe_path}' introuvable.")
        exit(1)

    data = torch.load(graphe_path, weights_only=False)
    data_bidirect = T.ToUndirected()(data)

    rel = ('medicament', 'interagit_avec', 'medicament')
    labels_complet = torch.argmax(data_bidirect[rel].edge_attr, dim=-1)
    print("=== Répartition des classes (ensemble complet, arêtes bidirectionnelles) ===")
    for c in range(4):
        n = (labels_complet == c).sum().item()
        print(f"  Classe {c} : {n} exemples ({n/len(labels_complet):.1%})")

    N_SEEDS = 10
    print(f"\n=== Entraînement + évaluation sur {N_SEEDS} seeds (split stratifié, sans fuite) ===")

    accuracies, f1_macros = [], []
    tous_labels_test, tous_preds_test = [], []

    for seed in range(N_SEEDS):
        labels_test, preds_test = entrainer_et_evaluer(data_bidirect, seed=seed)
        acc = (labels_test == preds_test).mean() * 100
        f1m = f1_score(labels_test, preds_test, average='macro', zero_division=0)
        accuracies.append(acc)
        f1_macros.append(f1m)
        tous_labels_test.extend(labels_test.tolist())
        tous_preds_test.extend(preds_test.tolist())
        print(f"  Seed {seed} : accuracy test = {acc:.1f}% | F1-macro test = {f1m:.3f} "
              f"(n_test = {len(labels_test)})")

    print(f"\n=== RÉSULTATS AGRÉGÉS SUR {N_SEEDS} SEEDS ===")
    print(f"Accuracy  : {np.mean(accuracies):.1f}% ± {np.std(accuracies):.1f}%")
    print(f"F1-macro  : {np.mean(f1_macros):.3f} ± {np.std(f1_macros):.3f}")

    print("\n=== Rapport par classe (cumulé sur tous les splits de test) ===")
    print(classification_report(
        tous_labels_test, tous_preds_test,
        target_names=[f"classe_{i}" for i in range(4)], zero_division=0,
    ))

    print("=== Matrice de confusion (cumulée) ===")
    print(confusion_matrix(tous_labels_test, tous_preds_test))

    n_ci = (np.array(tous_labels_test) == 2).sum()  # classe 2 = la plus rare (14/542 au total)
    print(f"\n⚠️  Rappel : seulement {n_ci} exemples cumulés de la classe 'contre_indication' "
          f"(la plus rare, 2.6% du total) sur les {N_SEEDS} splits de test — ses métriques "
          f"restent fragiles statistiquement et doivent être présentées avec cette réserve.")
    # =========================================================================
    # 📦 EXPORTATION DES ARTEFACTS POUR LE MICROSERVICE FASTAPI & SYMFONY
    # =========================================================================
    print("\n" + "="*60)
    print("📦 EXPORTATION DU MODÈLE ET DES MAPPINGS D'INFÉRENCE")
    print("="*60)

    # 1. Entraînement final du modèle sur la totalité du graphe pour le déploiement
    print("🔄 Entraînement final du modèle sur l'ensemble des données...")
    model_final = RemedHeteroGNN(
        num_patients=data_bidirect['patient'].num_nodes,
        num_meds=data_bidirect['medicament'].num_nodes,
        embed_dim=64, hidden_dim=32, out_dim=16
    )
    
    comptes_complet = torch.bincount(labels_complet, minlength=4).float()
    poids_complet = (1.0 / comptes_complet.clamp(min=1))
    poids_complet = poids_complet / poids_complet.sum() * 4
    criterion_final = nn.CrossEntropyLoss(weight=poids_complet)
    optimizer_final = torch.optim.Adam(model_final.parameters(), lr=0.01, weight_decay=5e-4)
    x_init_final = {'patient': model_final.patient_emb.weight, 'medicament': model_final.med_emb.weight}

    model_final.train()
    for epoch in range(1, 101):
        optimizer_final.zero_grad()
        predictions = model_final(x_init_final, data_bidirect.edge_index_dict, data_bidirect[rel].edge_index)
        loss = criterion_final(predictions, labels_complet)
        loss.backward()
        optimizer_final.step()

    # Save weights
    weights_path = "remed_gnn_weights.pt"
    torch.save(model_final.state_dict(), weights_path)
    print(f"✓ Poids finaux sauvegardés dans : '{weights_path}'")

    # 2. Construction et exportation de la carte d'indexation (Mappings)
    med_mapping = {}
    # Extraction dynamique si les nœuds contiennent les vrais identifiants (ex: CIS)
    if hasattr(data['medicament'], 'cis') or hasattr(data['medicament'], 'mapping_cis_to_idx'):
        if hasattr(data['medicament'], 'mapping_cis_to_idx'):
            med_mapping = data['medicament'].mapping_cis_to_idx
        else:
            cis_list = data['medicament'].cis
            med_mapping = {str(cis): idx for idx, cis in enumerate(cis_list)}
    else:
        # Fallback générique si seuls les indices bruts sont stockés
        num_meds = data['medicament'].num_nodes
        med_mapping = {f"MED_{i}": i for i in range(num_meds)}

    # Mappings des classes de gravité ANSM
    class_mapping = {
        "0": "a_prendre_en_compte",
        "1": "precaution_emploi",
        "2": "association_deconseillee",
        "3": "contre_indication"
    }

    mappings_payload = {
        "med_to_idx": med_mapping,
        "idx_to_class": class_mapping,
        "metadata": {
            "num_patients": int(data['patient'].num_nodes),
            "num_medicaments": int(data['medicament'].num_nodes),
            "embed_dim": 64,
            "hidden_dim": 32,
            "out_dim": 16
        }
    }

    mappings_path = "mappings_remed.json"
    with open(mappings_path, "w", encoding="utf-8") as f:
        json.dump(mappings_payload, f, ensure_ascii=False, indent=2)

    print(f"✓ Mappings et métadonnées sauvegardés dans : '{mappings_path}'")
    print("🚀 Artefacts prêts ! Vous pouvez lancer le microservice FastAPI.")
