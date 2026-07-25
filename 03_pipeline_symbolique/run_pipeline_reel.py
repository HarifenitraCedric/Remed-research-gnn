"""
Exécution du pipeline complet (Marches 1 & 2) sur les données réelles,
avec le vrai mapping ATC (mapping_atc_final.csv) au lieu du dictionnaire de
test à 4 entrées utilisé jusqu'ici.
"""

import numpy as np
import pandas as pd
from regles_medicales import detecter_alertes_patient

# ---------------------------------------------------------------------------
# 1. Calcul de couverture réelle (même logique que calcul_couverture_precis.py)
# ---------------------------------------------------------------------------
df = pd.read_csv("export_remed_avec_atc.csv")
df["date_achat"] = pd.to_datetime(df["date_achat"], format="mixed")

for col in ["dose_morning", "dose_noon", "dose_evening", "dose_night"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
df["dose_journaliere"] = df[["dose_morning", "dose_noon", "dose_evening", "dose_night"]].sum(axis=1)
df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

duree_reelle = df["quantity"] / df["dose_journaliere"].replace(0, np.nan)
duree_fallback = pd.to_numeric(df["duration_days"], errors="coerce")
df["couverture_source"] = np.where(duree_reelle.notna(), "posologie_reelle", "duration_days_fallback")
df["duree_couverture_jours"] = duree_reelle.fillna(duree_fallback).fillna(30)

df["date_debut_reel"] = df["date_achat"]
df["date_fin_reel"] = df["date_achat"] + pd.to_timedelta(df["duree_couverture_jours"], unit="D")

df.to_csv("data_prete_pour_graphe_reel.csv", index=False)
print(f"✅ Couverture calculée : {len(df)} lignes")
print(df["couverture_source"].value_counts().to_string())

# ---------------------------------------------------------------------------
# 2. Construction du graphe
# ---------------------------------------------------------------------------
CODES_EXCLUS = {"ATC_INCONNU", "DISPOSITIF_NON_MEDICAMENTEUX"}
df_pharma = df[~df["code_atc"].isin(CODES_EXCLUS)].copy()

nodes_list, edges_list = [], []
registered_nodes = set()
registered_prescribed_edges = {}


def add_node(node_id, node_type, label, properties=None):
    if node_id not in registered_nodes:
        entry = {"node_id": node_id, "type": node_type, "label": label}
        if properties:
            entry.update(properties)
        nodes_list.append(entry)
        registered_nodes.add(node_id)


def add_edge(source, target, edge_type, properties=None):
    entry = {"source": source, "target": target, "type": edge_type}
    if properties:
        entry.update(properties)
    edges_list.append(entry)


def add_or_increment_prescribed_edge(patient_id, atc):
    key = (patient_id, atc)
    if key in registered_prescribed_edges:
        edges_list[registered_prescribed_edges[key]]["nb_delivrances"] += 1
    else:
        edges_list.append({"source": f"PATIENT_{patient_id}", "target": f"DRUG_{atc}",
                            "type": "PRESCRIBED_TO", "nb_delivrances": 1})
        registered_prescribed_edges[key] = len(edges_list) - 1


n_alertes = 0
for patient_id, group in df_pharma.groupby("patient_id"):
    add_node(f"PATIENT_{patient_id}", "patient", f"Patient {patient_id}")
    achats = group.sort_values("date_debut_reel").to_dict("records")

    for a in achats:
        atc = str(a.get("code_atc", "")).strip()
        add_node(f"DRUG_{atc}", "drug", a["medicament_nom"], properties={"atc": atc})
        add_or_increment_prescribed_edge(patient_id, atc)

    for alerte in detecter_alertes_patient(achats):
        n_alertes += 1
        if alerte["type"] == "OVERDOSE_RISK":
            add_edge(f"PATIENT_{patient_id}", f"DRUG_{alerte['atc']}", "OVERDOSE_RISK",
                      properties={"dose_mg": alerte["dose_mg"], "limit_mg": alerte["limit_mg"]})
        elif alerte["type"] == "SAME_THERAPEUTIC_CLASS":
            add_edge(f"DRUG_{alerte['atc1']}", f"DRUG_{alerte['atc2']}", "SAME_THERAPEUTIC_CLASS")
        elif alerte["type"] == "TOXIC_INTERACTION":
            add_edge(f"DRUG_{alerte['atc1']}", f"DRUG_{alerte['atc2']}", "TOXIC_INTERACTION")

nodes_df = pd.DataFrame(nodes_list)
edges_df = pd.DataFrame(edges_list)
nodes_df.to_csv("nodes_reel.csv", index=False)
edges_df.to_csv("edges_reel.csv", index=False)

print(f"\n✅ Graphe construit sur données réelles : {n_alertes} alertes détectées")
print(f"\n📁 {len(nodes_df)} nœuds :")
print(nodes_df["type"].value_counts().to_string())
print(f"\n📁 {len(edges_df)} arêtes :")
print(edges_df["type"].value_counts().to_string())
