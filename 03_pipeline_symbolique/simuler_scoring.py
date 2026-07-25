"""
Simulation en Python de la logique de GnnRiskAnalyzerService.php, appliquée
directement sur les 27 patients réels (nodes_reel.csv / edges_reel.csv +
interactions_ansm.csv pour la gravité). Sert de référence pour vérifier que
l'exécution CLI PHP produit les mêmes résultats.
"""

import pandas as pd

WEIGHTS = {
    "contre_indication": 10,
    "association_deconseillee": 5,
    "precaution_emploi": 2,
    "a_prendre_en_compte": 0,
}

nodes = pd.read_csv("nodes_reel.csv")
edges = pd.read_csv("edges_reel.csv")
interactions_ansm = pd.read_csv("interactions_ansm.csv")

label_par_id = dict(zip(nodes["node_id"], nodes["label"]))
atc_par_id = dict(zip(nodes["node_id"], nodes["atc"]))

# Table de gravité indexée par paire ATC triée (même logique que regles_medicales.py)
gravite_par_paire = {}
for _, r in interactions_ansm.iterrows():
    cle = tuple(sorted([r["atc_1"], r["atc_2"]]))
    gravite_par_paire[cle] = r["gravite"]

prescribed = edges[edges["type"] == "PRESCRIBED_TO"]
toxic = edges[edges["type"] == "TOXIC_INTERACTION"]

patient_meds = prescribed.groupby("source")["target"].apply(set).to_dict()

resultats = []
for patient_id, meds in patient_meds.items():
    interactions_patient = toxic[toxic["source"].isin(meds) & toxic["target"].isin(meds)]

    total_score = 0.0
    detail = []
    for _, row in interactions_patient.iterrows():
        atc1, atc2 = atc_par_id[row["source"]], atc_par_id[row["target"]]
        cle = tuple(sorted([atc1, atc2]))
        gravite = gravite_par_paire.get(cle, "a_prendre_en_compte")
        poids = WEIGHTS.get(gravite, 0)
        # Note : ici confidence=1.0 (pas de GNN dans cette simulation, juste
        # la règle symbolique) -- le vrai score PHP utilisera la confiance
        # du GNN, donc les valeurs peuvent légèrement différer. C'est
        # attendu et à documenter : cette simulation valide la LOGIQUE
        # d'agrégation, pas les probabilités exactes du modèle.
        contribution = poids * 1.0
        total_score += contribution
        detail.append((label_par_id[row["source"]], label_par_id[row["target"]], gravite, poids))

    if total_score >= 10:
        niveau = "CRITICAL"
    elif total_score > 0:
        niveau = "MODERATE"
    else:
        niveau = "LOW"

    resultats.append({
        "patient": patient_id, "nb_medicaments": len(meds),
        "nb_interactions": len(interactions_patient),
        "risk_score": total_score, "risk_level": niveau, "detail": detail,
    })

df = pd.DataFrame(resultats).sort_values("risk_score", ascending=False)

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
