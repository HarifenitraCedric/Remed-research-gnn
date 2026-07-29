"""
Simulation en Python de la logique de GnnRiskAnalyzerService.php, appliquée
directement sur les 27 patients réels (nodes_reel.csv / edges_reel.csv +
interactions_ansm.csv pour la gravité). Sert de référence pour vérifier que
l'exécution CLI PHP produit les mêmes résultats.
"""

import os
from pathlib import Path
import pandas as pd

# -----------------------------------------------------------------------------
# Configuration des chemins d'accès (Résolution dynamique et récursive)
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

# Remonte à la racine du projet (C:\RMD\remed-research-gnn)
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name in ["03_pipeline_symbolique", "05_api_deploiement", "02_pipeline_gnn", "01_data_extraction"] else SCRIPT_DIR

def get_data_file_path(filename: str) -> Path:
    """
    Recherche un fichier de données dans les emplacements habituels,
    puis effectue une recherche récursive dans tout le projet si nécessaire.
    """
    candidates = [
        SCRIPT_DIR / filename,
        PROJECT_ROOT / "03_pipeline_symbolique" / filename,
        PROJECT_ROOT / filename,
        PROJECT_ROOT / "01_data_extraction" / filename,
        PROJECT_ROOT / "02_pipeline_gnn" / filename,
        PROJECT_ROOT / "data" / filename,
    ]
    
    # 1. Test des chemins explicites prioritaires
    for candidate in candidates:
        if candidate.exists():
            return candidate
            
    # 2. Recherche récursive globale dans tout le projet
    matches = list(PROJECT_ROOT.glob(f"**/{filename}"))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Impossible de localiser le fichier '{filename}' dans le projet : {PROJECT_ROOT}\n"
        f"Assurez-vous que le fichier existe et qu'il porte le bon nom."
    )

# -----------------------------------------------------------------------------
# Chargement des jeux de données
# -----------------------------------------------------------------------------
WEIGHTS = {
    "contre_indication": 10,
    "association_deconseillee": 5,
    "precaution_emploi": 2,
    "a_prendre_en_compte": 0,
}

print("=== DEBUT DE LA SIMULATION REMED GNN SCORING ===")

nodes_path = get_data_file_path("nodes_reel.csv")
edges_path = get_data_file_path("edges_reel.csv")
ansm_path = get_data_file_path("interactions_ansm.csv")

print(f"\n[INFO] Fichiers localisés avec succès :")
print(f"   • Nodes: {nodes_path}")
print(f"   • Edges: {edges_path}")
print(f"   • ANSM:  {ansm_path}\n")

nodes = pd.read_csv(nodes_path)
edges = pd.read_csv(edges_path)
interactions_ansm = pd.read_csv(ansm_path)

# Mappages d'identifiants
label_par_id = dict(zip(nodes["node_id"], nodes["label"]))
atc_par_id = dict(zip(nodes["node_id"], nodes["atc"]))

# Table de gravité indexée par paire ATC triée
gravite_par_paire = {}
for _, r in interactions_ansm.iterrows():
    cle = tuple(sorted([str(r["atc_1"]), str(r["atc_2"])]))
    gravite_par_paire[cle] = r["gravite"]

# Filtrage des relations dans le graphe
prescribed = edges[edges["type"] == "PRESCRIBED_TO"]
toxic = edges[edges["type"] == "TOXIC_INTERACTION"]

patient_meds = prescribed.groupby("source")["target"].apply(set).to_dict()

# -----------------------------------------------------------------------------
# Calcul du score de risque par patient
# -----------------------------------------------------------------------------
resultats = []
for patient_id, meds in patient_meds.items():
    interactions_patient = toxic[toxic["source"].isin(meds) & toxic["target"].isin(meds)]

    total_score = 0.0
    detail = []
    for _, row in interactions_patient.iterrows():
        atc1 = str(atc_par_id.get(row["source"], ""))
        atc2 = str(atc_par_id.get(row["target"], ""))
        cle = tuple(sorted([atc1, atc2]))
        
        gravite = gravite_par_paire.get(cle, "a_prendre_en_compte")
        poids = WEIGHTS.get(gravite, 0)
        
        contribution = poids * 1.0
        total_score += contribution
        
        med1_label = label_par_id.get(row["source"], row["source"])
        med2_label = label_par_id.get(row["target"], row["target"])
        detail.append((med1_label, med2_label, gravite, poids))

    if total_score >= 10:
        niveau = "CRITICAL"
    elif total_score > 0:
        niveau = "MODERATE"
    else:
        niveau = "LOW"

    resultats.append({
        "patient": patient_id,
        "nb_medicaments": len(meds),
        "nb_interactions": len(interactions_patient),
        "risk_score": total_score,
        "risk_level": niveau,
        "detail": detail,
    })

df = pd.DataFrame(resultats).sort_values("risk_score", ascending=False)

# -----------------------------------------------------------------------------
# Affichage des résultats
# -----------------------------------------------------------------------------
print("=== SCORES DE RISQUE PAR PATIENT (simulation de référence) ===\n")
print(df[["patient", "nb_medicaments", "nb_interactions", "risk_score", "risk_level"]].to_string(index=False))

print(f"\nRépartition : {(df['risk_level']=='CRITICAL').sum()} CRITICAL, "
      f"{(df['risk_level']=='MODERATE').sum()} MODERATE, "
      f"{(df['risk_level']=='LOW').sum()} LOW")

print("\n=== DÉTAIL DES 3 PATIENTS LES PLUS À RISQUE ===")
for _, r in df.head(3).iterrows():
    print(f"\n--- {r['patient']} : score = {r['risk_score']} ({r['risk_level']}) ---")
    for med1, med2, gravite, poids in r["detail"]:
        print(f"   [{gravite:25s} poids={poids:2d}]  {med1}  <->  {med2}")