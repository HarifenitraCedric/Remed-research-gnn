import json
import torch
from modele_heterognn import RemedHeteroGNN # Ton classe d'architecture

class RemedPredictor:
    def __init__(self, weights_path='remed_gnn_weights.pt', mappings_path='mappings_remed.json', graph_path='graphe_heterogene_complet.pt'):
        # 1. Charger les mappings
        with open(mappings_path, 'r', encoding='utf-8') as f:
            self.mappings = json.load(f)
            
        # 2. Charger le graphe de structure
        self.graph = torch.load(graph_path, weights_only=False)
        
        # 3. Initialiser le modèle et charger les poids
        self.model = RemedHeteroGNN(hidden_channels=64, out_channels=16)
        self.model.load_state_dict(torch.load(weights_path))
        self.model.eval()

    def analyser_couple_meds(self, med_a_id, med_b_id):
        """Passe deux médicaments dans le décodeur et renvoie la sévérité prédite."""
        idx_a = self.mappings["med_to_idx"].get(med_a_id)
        idx_b = self.mappings["med_to_idx"].get(med_b_id)

        if idx_a is None or idx_b is None:
            return {"erreur": "Un ou plusieurs médicaments sont inconnus dans le graphe."}

        with torch.no_grad():
            # Génération des embeddings via le passage de messages
            x_dict = self.model(self.graph.x_dict, self.graph.edge_index_dict)
            
            # Reconstruction du lien à évaluer
            edge_eval = torch.tensor([[idx_a], [idx_b]], dtype=torch.long)
            
            # Prédiction via le décodeur MLP
            logits = self.model.decoder(x_dict['médicament'], edge_eval)
            probs = torch.softmax(logits, dim=-1)
            pred_class_idx = torch.argmax(probs, dim=-1).item()

        return {
            "med_a": med_a_id,
            "med_b": med_b_id,
            "gravite_predite": self.mappings["idx_to_class"][str(pred_class_idx)],
            "confiance": round(probs[0][pred_class_idx].item(), 4)
        }