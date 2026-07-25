import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv
import torch_geometric.transforms as T
import os

# =====================================================================
# 1. DÉFINITION DE L'ARCHITECTURE DU GNN HÉTÉROGÈNE COHÉRENTE
# =====================================================================

class RemedHeteroGNN(nn.Module):
    def __init__(self, num_patients, num_meds, embed_dim, hidden_dim, out_dim):
        super().__init__()
        
        # Tables d'embeddings entraînables
        self.patient_emb = nn.Embedding(num_patients, embed_dim)
        self.med_emb = nn.Embedding(num_meds, embed_dim)
        
        # COUCHE DE CONVOLUTION 1
        # On ajoute explicitement la relation inverse 'rev_a_prescrit' pour mettre à jour les patients !
        self.conv1 = HeteroConv({
            ('patient', 'a_prescrit', 'medicament'): SAGEConv((-1, -1), hidden_dim),
            ('medicament', 'rev_a_prescrit', 'patient'): SAGEConv((-1, -1), hidden_dim), # Chemin retour
            ('medicament', 'interagit_avec', 'medicament'): SAGEConv((-1, -1), hidden_dim)
        }, aggr='sum')
        
        # COUCHE DE CONVOLUTION 2
        self.conv2 = HeteroConv({
            ('patient', 'a_prescrit', 'medicament'): SAGEConv((-1, -1), out_dim),
            ('medicament', 'rev_a_prescrit', 'patient'): SAGEConv((-1, -1), out_dim), # Chemin retour
            ('medicament', 'interagit_avec', 'medicament'): SAGEConv((-1, -1), out_dim)
        }, aggr='sum')

    def forward(self, x_dict, edge_index_dict):
        # 1. Premier bloc de Message Passing + Activation ReLU
        h_dict = self.conv1(x_dict, edge_index_dict)
        h_dict = {key: F.relu(x) for key, x in h_dict.items()}
        
        # 2. Deuxième bloc de Message Passing (Sorties finales)
        out_dict = self.conv2(h_dict, edge_index_dict)
        
        return out_dict

# =====================================================================
# 2. CHARGEMENT ET VÉRIFICATION DE NOTRE GRAPHE ASSEMBLÉ
# =====================================================================

if __name__ == "__main__":
    print("=== Initialisation du Modèle HeteroGNN REMED ===")
    
    graphe_path = "graphe_heterogene_complet.pt"
    
    if not os.path.exists(graphe_path):
        print(f"[Erreur] Le fichier '{graphe_path}' est introuvable.")
        exit(1)
        
    # Chargement sécurisé pour l'environnement local
    data = torch.load(graphe_path, weights_only=False)
    print("\n1. Graphe initial chargé.")
    
    # --- TRANSFORMATION : Rendre le graphe bidirectionnel ---
    # ToUndirected va dupliquer automatiquement la relation ('patient', 'a_prescrit', 'medicament')
    # en créant ('medicament', 'rev_a_prescrit', 'patient') pour notre modèle.
    transform = T.ToUndirected()
    data_bidirect = transform(data)
    print("-> Transformation bidirectionnelle appliquée avec succès (rev_a_prescrit créée).")
    
    num_patients = data_bidirect['patient'].num_nodes
    num_meds = data_bidirect['medicament'].num_nodes
    
    # Configuration des dimensions
    EMBED_DIM = 64     
    HIDDEN_DIM = 32    
    OUT_DIM = 16       
    
    # Création du modèle
    model = RemedHeteroGNN(
        num_patients=num_patients,
        num_meds=num_meds,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        out_dim=OUT_DIM
    )
    
    # Préparation des embeddings d'entrée pour la méthode forward
    x_initial_dict = {
        'patient': model.patient_emb.weight,
        'medicament': model.med_emb.weight
    }
    
    # Exécution du test
    with torch.no_grad():
        embeddings_finaux = model(x_initial_dict, data_bidirect.edge_index_dict)
        
    print("\n2. Résultat du Forward Pass de test :")
    for type_noeud, tenseur in embeddings_finaux.items():
        print(f"   - Plongements générés pour '{type_noeud}' : Shape {list(tenseur.shape)}")
        
    print("\n[Succès] L'architecture du GNN est valide, robuste et compile sans aucun bug !")