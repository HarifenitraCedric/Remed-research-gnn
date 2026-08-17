import torch
import numpy as np
import torch_geometric.transforms as T
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

print("⏳ Chargement du graphe hétérogène (215 codes ATC)...")
data = torch.load("graphe_heterogene_complet.pt", weights_only=False)
data_bidirect = T.ToUndirected()(data)
rel = ("medicament", "interagit_avec", "medicament")
labels_complet = torch.argmax(data_bidirect[rel].edge_attr, dim=-1)

print(f"✅ {len(labels_complet)} arêtes d'interaction (bidirectionnelles)")

N_SEEDS = 10
acc_baseline, f1_baseline = [], []

for seed in range(N_SEEDS):
    idx = np.arange(len(labels_complet))
    idx_train, idx_test = train_test_split(
        idx, test_size=0.2, random_state=seed, stratify=labels_complet.numpy()
    )
    labels_train = labels_complet[idx_train].numpy()
    labels_test = labels_complet[idx_test].numpy()

    # Baseline : prédit toujours la classe majoritaire DU TRAIN (jamais le test)
    classe_majoritaire = np.bincount(labels_train).argmax()
    preds_baseline = np.full(len(labels_test), classe_majoritaire)

    acc_baseline.append(accuracy_score(labels_test, preds_baseline) * 100)
    f1_baseline.append(f1_score(labels_test, preds_baseline, average="macro", zero_division=0))

print("\n" + "=" * 70)
print(" TABLEAU COMPARATIF HONNÊTE (même tâche, même protocole, même graphe)")
print("=" * 70)
print(f"{'Modèle':<35}{'Accuracy':<20}{'F1-macro':<20}")
print(f"{'Baseline (classe majoritaire)':<35}{np.mean(acc_baseline):>6.1f}% ± {np.std(acc_baseline):<10.1f}"
      f"{np.mean(f1_baseline):>6.3f} ± {np.std(f1_baseline):.3f}")
print(f"{'GNN (RemedHeteroGNN, 215 codes)':<35}{'97.9% ± 0.6%':<20}{'0.970 ± 0.013':<20}")
print("=" * 70)
print("\n⚠️  Ligne GNN reprise de train_gnn_prod.py (calculée avec le même protocole")
print("    exact : split stratifié identique, sans fuite de message passing, 10 seeds).")
print("    Ligne baseline calculée ICI, sur les mêmes splits, pour une vraie comparaison.")

print("\nRépartition réelle des classes (train, seed=0, pour référence) :")
idx_train0, _ = train_test_split(np.arange(len(labels_complet)), test_size=0.2, random_state=0,
                                   stratify=labels_complet.numpy())
labels_ref = labels_complet[idx_train0].numpy()
for c in range(4):
    n = (labels_ref == c).sum()
    print(f"  Classe {c} : {n} ({n/len(labels_ref):.1%})")