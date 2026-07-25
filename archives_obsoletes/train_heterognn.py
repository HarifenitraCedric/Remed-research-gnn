import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv
import torch_geometric.transforms as T
import os

# =====================================================================
# 1. MODÈLE COMPLET : ENCODEUR GNN + DÉCODEUR DE GRAVITÉ D'INTERACTIONS
# =====================================================================

class RemedHeteroGNN(nn.Module):
    def __init__(self, num_patients, num_meds, embed_dim, hidden_dim, out_dim):
        super().__init__()
        
        # 1.1 Embeddings apprenables
        self.patient_emb = nn.Embedding(num_patients, embed_dim)
        self.med_emb = nn.Embedding(num_meds, embed_dim)
        
        # 1.2 Couches de convolution hétérogène (Encodeur)
        self.conv1 = HeteroConv({
            ('patient', 'a_prescrit', 'medicament'): SAGEConv((-1, -1), hidden_dim),
            ('medicament', 'rev_a_prescrit', 'patient'): SAGEConv((-1, -1), hidden_dim),
            ('medicament', 'interagit_avec', 'medicament'): SAGEConv((-1, -1), hidden_dim)
        }, aggr='sum')
        
        self.conv2 = HeteroConv({
            ('patient', 'a_prescrit', 'medicament'): SAGEConv((-1, -1), out_dim),
            ('medicament', 'rev_a_prescrit', 'patient'): SAGEConv((-1, -1), out_dim),
            ('medicament', 'interagit_avec', 'medicament'): SAGEConv((-1, -1), out_dim)
        }, aggr='sum')
        
        # 1.3 Décodeur d'arêtes (Classification multi-classe : 4 niveaux de gravité)
        # Reçoit en entrée la combinaison de 2 embeddings de médicaments (out_dim * 2)
        self.decoder = nn.Sequential(
            nn.Linear(out_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, 4) # 4 sorties correspondantes aux 4 classes de gravité
        )

    def encode(self, x_dict, edge_index_dict):
        """Passe dans le GNN pour générer les représentations enrichies des nœuds."""
        h_dict = self.conv1(x_dict, edge_index_dict)
        h_dict = {key: F.relu(x) for key, x in h_dict.items()}
        out_dict = self.conv2(h_dict, edge_index_dict)
        return out_dict

    def decode(self, z_med, edge_label_index):
        """
        Prédit la gravité pour les arêtes données.
        
        Args:
            z_med (Tensor): Embeddings des médicaments générés par l'encodeur.
            edge_label_index (Tensor): Indices [2, num_edges_a_predire] des paires de médicaments.
        """
        # Récupération des embeddings des sources (molécule A) et destinations (molécule B)
        nodes_src = edge_label_index[0]
        nodes_dst = edge_label_index[1]
        
        x_src = z_med[nodes_src]
        x_dst = z_med[nodes_dst]
        
        # Concaténation des caractéristiques des deux nœuds pour l'arête
        edge_features = torch.cat([x_src, x_dst], dim=-1)
        
        # Prédiction des scores (logits) pour chaque classe de gravité
        return self.decoder(edge_features)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encode(x_dict, edge_index_dict)
        logits = self.decode(z_dict['medicament'], edge_label_index)
        return logits

# =====================================================================
# 2. CONFIGURATION ET SIMULATION DE LA BOUCLE D'ENTRAÎNEMENT
# =====================================================================

if __name__ == "__main__":
    graphe_path = "graphe_heterogene_complet.pt"
    if not os.path.exists(graphe_path):
        print(f"[Erreur] '{graphe_path}' introuvable.")
        exit(1)
        
    # 2.1 Chargement et préparation des données
    data = torch.load(graphe_path, weights_only=False)
    data_bidirect = T.ToUndirected()(data)
    
    # Récupération des arêtes cibles d'interaction et de leurs étiquettes (gravité)
    # edge_attr est au format One-Hot [542, 4], on utilise argmax pour avoir l'index de classe (0, 1, 2 ou 3)
    edge_label_index = data_bidirect['medicament', 'interagit_avec', 'medicament'].edge_index
    edge_attr_onehot = data_bidirect['medicament', 'interagit_avec', 'medicament'].edge_attr
    edge_labels = torch.argmax(edge_attr_onehot, dim=-1) # Dimensions: [542]
    
    # 2.2 Initialisation du modèle et de l'optimiseur
    model = RemedHeteroGNN(
        num_patients=data_bidirect['patient'].num_nodes,
        num_meds=data_bidirect['medicament'].num_nodes,
        embed_dim=64,
        hidden_dim=32,
        out_dim=16
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    print("=== Lancement d'une simulation d'entraînement (10 époques) ===")
    
    model.train()
    for epoch in range(1, 11):
        optimizer.zero_grad()
        
        # Génération des embeddings d'entrée dynamiques
        x_initial_dict = {
            'patient': model.patient_emb.weight,
            'medicament': model.med_emb.weight
        }
        
        # Forward pass : Prédiction des gravités d'interactions
        predictions = model(x_initial_dict, data_bidirect.edge_index_dict, edge_label_index)
        
        # Calcul de la perte
        loss = criterion(predictions, edge_labels)
        
        # Rétropropagation et optimisation
        loss.backward()
        optimizer.step()
        
        # Calcul d'une précision basique pour le suivi
        preds_classes = torch.argmax(predictions, dim=-1)
        acc = (preds_classes == edge_labels).float().mean().item() * 100
        
        print(f"Époque {epoch:02d} | Perte (Loss): {loss.item():.4f} | Précision d'apprentissage: {acc:.2f}%")
        
    print("\n[Succès] Le décodeur de liens est configuré et la boucle d'apprentissage fonctionne !")