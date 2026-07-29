import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv
import torch_geometric.transforms as T
import pandas as pd


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
            nn.Linear(out_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(p=0.2), nn.Linear(hidden_dim, 4),
        )

    def encode(self, x_dict, edge_index_dict):
        h_dict = self.conv1(x_dict, edge_index_dict)
        h_dict = {key: F.relu(x) for key, x in h_dict.items()}
        return self.conv2(h_dict, edge_index_dict)

    def decode(self, z_med, edge_label_index):
        return self.decoder(torch.cat([z_med[edge_label_index[0]], z_med[edge_label_index[1]]], dim=-1))

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        return self.decode(self.encode(x_dict, edge_index_dict)['medicament'], edge_label_index)


print("=== [Orchestrateur] Export des artefacts ===")
data = torch.load("graphe_heterogene_complet.pt", weights_only=False)
data_bidirect = T.ToUndirected()(data)
rel = ('medicament', 'interagit_avec', 'medicament')
labels_complet = torch.argmax(data_bidirect[rel].edge_attr, dim=-1)

proportions_reelles = {c: (labels_complet == c).sum().item() / len(labels_complet) for c in range(4)}
thesaurus = pd.read_csv("interactions_thesaurus_global.csv")
proportions_attendues = thesaurus["gravite"].value_counts(normalize=True).to_dict()

print("Classe  % observé   Gravité                      % attendu")
mapping_detecte = {}
gravites_prises = set()
for classe_idx, prop_obs in sorted(proportions_reelles.items(), key=lambda x: x[0]):
    meilleure_gravite, meilleur_ecart = None, float("inf")
    for gravite, prop_att in proportions_attendues.items():
        if gravite in gravites_prises:
            continue
        ecart = abs(prop_obs - prop_att)
        if ecart < meilleur_ecart:
            meilleur_ecart, meilleure_gravite = ecart, gravite
    mapping_detecte[str(classe_idx)] = meilleure_gravite
    gravites_prises.add(meilleure_gravite)
    print(f"{classe_idx:<8}{prop_obs:<12.1%}{meilleure_gravite:<28}{proportions_attendues[meilleure_gravite]:<10.1%}")

print(f"\nMapping détecté : {mapping_detecte}")

torch.manual_seed(42)
model_final = RemedHeteroGNN(
    num_patients=data_bidirect['patient'].num_nodes,
    num_meds=data_bidirect['medicament'].num_nodes,
    embed_dim=64, hidden_dim=32, out_dim=16,
)
comptes = torch.bincount(labels_complet, minlength=4).float()
poids = (1.0 / comptes.clamp(min=1)); poids = poids / poids.sum() * 4
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

torch.save(model_final.state_dict(), "remed_gnn_weights.pt")

if hasattr(data['medicament'], 'mapping_cis_to_idx'):
    med_mapping = data['medicament'].mapping_cis_to_idx
elif hasattr(data['medicament'], 'cis'):
    med_mapping = {str(cis): idx for idx, cis in enumerate(data['medicament'].cis)}
else:
    med_mapping = {f"MED_{i}": i for i in range(data['medicament'].num_nodes)}

mappings_payload = {
    "med_to_idx": med_mapping,
    "idx_to_class": mapping_detecte,
    "metadata": {
        "num_patients": int(data['patient'].num_nodes),
        "num_medicaments": int(data['medicament'].num_nodes),
        "embed_dim": 64, "hidden_dim": 32, "out_dim": 16,
    },
}
with open("mappings_remed.json", "w", encoding="utf-8") as f:
    json.dump(mappings_payload, f, ensure_ascii=False, indent=2)

print("[Succès] remed_gnn_weights.pt + mappings_remed.json sauvegardés.")
