"""
Diagnostic ciblé : pourquoi ATORVASTATINE (C10AA05) et ACIDE ACETYLSALICYLIQUE
(A01AD05) apparaissent-elles en contre_indication dans le graphe, alors
qu'aucune ligne brute du thésaurus ANSM ne les relie directement ?

À lancer depuis la racine du projet (C:\\RMD\\remed-research-gnn) :
    python diagnostic_paire_suspecte.py
"""

import pandas as pd

print("=== 1. Vérifier si la paire existe dans interactions_ansm.csv ===")
interactions = pd.read_csv("02_interactions_ansm/interactions_ansm.csv")
paire = interactions[
    ((interactions["atc_1"] == "C10AA05") & (interactions["atc_2"] == "A01AD05")) |
    ((interactions["atc_1"] == "A01AD05") & (interactions["atc_2"] == "C10AA05"))
]
print(f"Nombre de lignes trouvées : {len(paire)}")
if len(paire):
    print(paire.to_string())
print()

print("=== 2. Vérifier s'il y a UNE SEULE ligne pour C10AA05 et A01AD05 chacun ===")
print("(un doublon indiquerait 2 vraies molécules fusionnées sur le même nœud)")
mapping = pd.read_csv("01_mapping_atc/mapping_atc_final.csv")
c10 = mapping[mapping["code_atc"] == "C10AA05"]
a01 = mapping[mapping["code_atc"] == "A01AD05"]
print(f"\nMédicaments mappés à C10AA05 ({len(c10)}) :")
print(c10[["medicament_nom", "denomination_substance", "methode_matching"]].to_string())
print(f"\nMédicaments mappés à A01AD05 ({len(a01)}) :")
print(a01[["medicament_nom", "denomination_substance", "methode_matching"]].to_string())
print()

print("=== 3. Vérifier la présence de codes ATC dupliqués dans nodes_reel.csv ===")
nodes = pd.read_csv("03_pipeline_symbolique/nodes_reel.csv")
drugs = nodes[nodes["type"] == "drug"]
doublons = drugs[drugs.duplicated(subset="atc", keep=False)]
print(f"Nœuds médicament avec un code ATC dupliqué : {len(doublons)}")
if len(doublons):
    print(doublons.to_string())
print()

print("=== 4. Chercher l'arête TOXIC_INTERACTION exacte dans edges_reel.csv ===")
edges = pd.read_csv("03_pipeline_symbolique/edges_reel.csv")
edge_match = edges[
    (edges["source"].isin(["DRUG_C10AA05", "DRUG_A01AD05"])) &
    (edges["target"].isin(["DRUG_C10AA05", "DRUG_A01AD05"])) &
    (edges["type"] == "TOXIC_INTERACTION")
]
print(f"Arêtes trouvées : {len(edge_match)}")
print(edge_match.to_string())
