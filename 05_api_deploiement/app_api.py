import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import torch_geometric.transforms as T
from torch_geometric.nn import HeteroConv, SAGEConv


# =====================================================================
# 1. Architecture RemedHeteroGNN
# =====================================================================
class RemedHeteroGNN(nn.Module):
    def __init__(self, num_patients: int, num_meds: int, embed_dim: int = 64, hidden_dim: int = 32, out_dim: int = 16):
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
            nn.Linear(hidden_dim, 4),  # 4 classes de gravité ANSM
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


# =====================================================================
# 2. Gestionnaire d'État Global de l'Application
# =====================================================================
class ModelService:
    def __init__(self):
        self.model: Optional[RemedHeteroGNN] = None
        self.data_bidirect = None
        self.mappings: Dict = {}
        self.med_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[str, str] = {}
        self.cis_to_med: Dict[str, str] = {}
        self.ansm_edges_set: set = set()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_artifacts(
            self,
            weights_path: str = "remed_gnn_weights.pt",
            mappings_path: str = "mappings_remed.json",
            graphe_path: str = "graphe_heterogene_complet.pt"
        ):
            base_dir = Path(__file__).resolve().parent

            full_mappings_path = base_dir / mappings_path
            full_weights_path = base_dir / weights_path
            full_graphe_path = base_dir / graphe_path

            if not full_mappings_path.exists():
                raise FileNotFoundError(f"Fichier de mappings introuvable : {full_mappings_path}")
            if not full_weights_path.exists():
                raise FileNotFoundError(f"Fichier de poids introuvable : {full_weights_path}")
            if not full_graphe_path.exists():
                raise FileNotFoundError(f"Fichier de graphe introuvable : {full_graphe_path}")

            # 1. Chargement du dictionnaire JSON
            with open(full_mappings_path, "r", encoding="utf-8") as f:
                self.mappings = json.load(f)

            self.med_to_idx = self.mappings.get("med_to_idx", {})
            self.idx_to_class = self.mappings.get("idx_to_class", {})
            self.cis_to_med = self.mappings.get("cis_to_med", {})
            metadata = self.mappings.get("metadata", {})

            # 2. Chargement de la structure du graphe
            data = torch.load(full_graphe_path, weights_only=False)

            # Extraire les dimensions EXACTES avant la transformation bidirectionnelle
            raw_num_patients = data['patient'].num_nodes if 'patient' in data.node_types else None
            raw_num_meds = data['medicament'].num_nodes if 'medicament' in data.node_types else None

            self.data_bidirect = T.ToUndirected()(data).to(self.device)

            self.ansm_edges_set = set()
            edge_type = ('medicament', 'interagit_avec', 'medicament')
            if edge_type in data.edge_index_dict:
                edges = data.edge_index_dict[edge_type]
                for u, v in zip(edges[0].tolist(), edges[1].tolist()):
                    self.ansm_edges_set.add((min(u, v), max(u, v)))

            # 3. Résolution robuste du nombre de nœuds pour l'instanciation du modèle
            num_patients = metadata.get("num_patients") or raw_num_patients or 27
            num_meds = metadata.get("num_medicaments") or raw_num_meds or 215

            # Chargement de la structure du state_dict pour validation
            state_dict = torch.load(full_weights_path, map_location=self.device)
            
            # Validation dynamique des dimensions à partir du checkpoint si disponible
            if "patient_emb.weight" in state_dict:
                num_patients = state_dict["patient_emb.weight"].shape[0]
            if "med_emb.weight" in state_dict:
                num_meds = state_dict["med_emb.weight"].shape[0]

            self.model = RemedHeteroGNN(
                num_patients=num_patients,
                num_meds=num_meds,
                embed_dim=64,
                hidden_dim=32,
                out_dim=16
            )

            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

    def resolve_med_index(self, identifier: str) -> int:
        clean_id = str(identifier).strip().upper()

        if clean_id in self.med_to_idx:
            return self.med_to_idx[clean_id]

        for key in self.med_to_idx:
            if key.upper() == clean_id:
                return self.med_to_idx[key]

        if clean_id.startswith("MED_"):
            raw_num = clean_id.replace("MED_", "")
            if raw_num in self.med_to_idx:
                return self.med_to_idx[raw_num]
            try:
                num_idx = int(raw_num)
                if 0 <= num_idx < self.data_bidirect['medicament'].num_nodes:
                    return num_idx
            except ValueError:
                pass

        if self.cis_to_med and clean_id in self.cis_to_med:
            internal_code = self.cis_to_med[clean_id]
            if internal_code in self.med_to_idx:
                return self.med_to_idx[internal_code]

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Médicament / Code ATC '{identifier}' introuvable dans le dictionnaire du modèle GNN."
        )


service = ModelService()


# =====================================================================
# 3. Lifecycle FastAPI
# =====================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        service.load_artifacts()
    except Exception as e:
        print(f"[ERREUR] Échec du chargement des artéfacts : {e}")
    yield


app = FastAPI(
    title="RemedHeteroGNN Inference API",
    description="API de prédiction du niveau de gravité d'interaction médicamenteuse pour REMED.",
    version="2.0.0",
    lifespan=lifespan
)


# =====================================================================
# 4. Schémas Pydantic
# =====================================================================
class PredictionRequest(BaseModel):
    code_cis_1: str = Field(..., description="Identifiant / Code CIS du premier médicament (ex: 'MED_0', '0' ou '60002283')")
    code_cis_2: str = Field(..., description="Identifiant / Code CIS du second médicament (ex: 'MED_1', '1' ou '60002284')")

class PredictionResponse(BaseModel):
    code_cis_1: str
    code_cis_2: str
    class_id: int = Field(..., description="Index de la classe prédite (0, 1, 2, 3)")
    niveau_gravite: str = Field(..., description="Libellé de la gravité (ex: 'contre_indication')")
    probabilities: Dict[str, float] = Field(..., description="Probabilités pour chacune des 4 classes")
    is_ansm: bool = Field(False, description="Vrai si l'interaction est issue du thésaurus ANSM, Faux si prédiction GNN pure")

class BatchPredictionRequest(BaseModel):
    medicaments: List[str] = Field(
        ..., 
        description="Liste des identifiants/codes CIS de l'ordonnance (ex: ['MED_0', 'MED_1', 'MED_2'])"
    )

class BatchPredictionResponse(BaseModel):
    nb_medicaments: int
    nb_paires_analysées: int
    riskScore: float = Field(..., description="Score de risque global maximal calculé pour l'ordonnance (0 à 10)")
    interactions: List[PredictionResponse]
    erreurs_identification: List[Dict[str, str]]


# =====================================================================
# 5. Endpoints
# =====================================================================
@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    is_ready = service.model is not None and bool(service.med_to_idx)
    return {
        "status": "healthy" if is_ready else "degraded",
        "model_loaded": service.model is not None,
        "device": str(service.device)
    }


POIDS_GRAVITE = {
    "contre_indication": 10.0,
    "association_deconseillee": 7.0,
    "precaution_emploi": 4.0,
    "a_prendre_en_compte": 2.0,
}

@app.post("/predict-batch", response_model=BatchPredictionResponse)
def predict_batch(payload: BatchPredictionRequest):
    if not service.model or not service.med_to_idx:
        raise HTTPException(status_code=503, detail="Microservice non initialisé.")

    meds = payload.medicaments
    n = len(meds)
    if n < 2:
        return BatchPredictionResponse(
            nb_medicaments=n,
            nb_paires_analysées=0,
            riskScore=0.0,
            interactions=[],
            erreurs_identification=[]
        )

    indices, erreurs, meds_valides = [], [], []
    for m in meds:
        try:
            indices.append(service.resolve_med_index(m))
            meds_valides.append(m)
        except HTTPException as e:
            erreurs.append({"paire": m, "erreur": e.detail})

    interactions = []
    if len(indices) >= 2:
        with torch.no_grad():
            x_initial_dict = {
                'patient': service.model.patient_emb.weight,
                'medicament': service.model.med_emb.weight,
            }
            z_dict = service.model.encode(x_initial_dict, service.data_bidirect.edge_index_dict)

            paires = [(i, j) for i in range(len(indices)) for j in range(i + 1, len(indices))]
            src = torch.tensor([indices[i] for i, j in paires], device=service.device)
            dst = torch.tensor([indices[j] for i, j in paires], device=service.device)
            edge_label_index = torch.stack([src, dst])

            logits = service.model.decode(z_dict['medicament'], edge_label_index)
            probs = F.softmax(logits, dim=-1)

        for (i, j), p in zip(paires, probs):
            predicted_class_id = int(torch.argmax(p).item())
            class_probs = {
                service.idx_to_class.get(str(k), f"classe_{k}"): round(float(p[k].item()), 4)
                for k in range(len(p))
            }
            idx_1 = indices[i]
            idx_2 = indices[j]
            pair_key = (min(idx_1, idx_2), max(idx_1, idx_2))
            is_ansm_certified = pair_key in service.ansm_edges_set

            interactions.append(PredictionResponse(
                code_cis_1=meds_valides[i],
                code_cis_2=meds_valides[j],
                class_id=predicted_class_id,
                niveau_gravite=service.idx_to_class.get(str(predicted_class_id), f"classe_{predicted_class_id}"),
                probabilities=class_probs,
                is_ansm=is_ansm_certified
            ))

    max_score = 0.0
    for inter in interactions:
        pair_score = sum(POIDS_GRAVITE.get(lbl, 0.0) * prob for lbl, prob in inter.probabilities.items())
        if pair_score > max_score:
            max_score = pair_score

    return BatchPredictionResponse(
        nb_medicaments=n,
        nb_paires_analysées=len(interactions) + len(erreurs),
        riskScore=round(max_score, 2),
        interactions=interactions,
        erreurs_identification=erreurs,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080, reload=True)