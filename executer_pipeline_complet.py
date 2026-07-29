"""
Orchestrateur du pipeline complet REMED-RESEARCH.

À lancer UNE SEULE FOIS depuis la racine du projet :
    python executer_pipeline_complet.py

Ce script :
  1. Crée un dossier de travail temporaire _run_pipeline/
  2. Copie les données sources nécessaires dedans
  3. Exécute chaque script du pipeline, dans l'ordre, avec ce dossier comme
     répertoire de travail (cwd) -- donc chaque script trouve ses fichiers
     d'entrée par un simple nom relatif, sans avoir à modifier son code
  4. Vérifie automatiquement l'absence du bug ACIDE FOLIQUE / A01AD05
  5. Recopie les résultats finaux dans les dossiers organisés
     (01_mapping_atc/, 02_interactions_ansm/, 03_pipeline_symbolique/,
     04_gnn/) ET dans 05_api_deploiement/ (pour qu'app_api.py les trouve
     avec ses chemins par défaut, sans modification).

Rien n'est supprimé dans les dossiers organisés existants -- seulement
écrasé par une version plus fraîche, cohérente de bout en bout.
"""

import os
import shutil
import subprocess
import sys
import pandas as pd

# Force l'UTF-8 pour la sortie de CE script aussi (pas seulement les
# sous-scripts) -- la console Windows par défaut (cp1252) plante sur les
# emojis utilisés dans les messages ci-dessous (✅, 🎉...).
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.getcwd()
WORK = os.path.join(ROOT, "_run_pipeline")


def etape(titre):
    print("\n" + "=" * 70)
    print(titre)
    print("=" * 70)


def lancer(script_path, cwd=WORK):
    """Exécute un script Python avec le dossier de travail donné.
    Force l'UTF-8 pour l'environnement du sous-processus, car la console
    Windows utilise par défaut l'encodage cp1252, qui plante sur les
    emojis (✅, 🎉...) affichés par les scripts du pipeline."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    resultat = subprocess.run(
        [sys.executable, os.path.join(ROOT, script_path)],
        cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=env,
    )
    print(resultat.stdout)
    if resultat.returncode != 0:
        print(resultat.stderr)
        raise RuntimeError(f"Échec de {script_path} (code {resultat.returncode})")


# ---------------------------------------------------------------------------
# 0. Préparation du dossier de travail
# ---------------------------------------------------------------------------
etape("0. Préparation du dossier de travail temporaire")
if os.path.exists(WORK):
    shutil.rmtree(WORK)
os.makedirs(WORK)

sources = {
    "00_donnees_sources/atc_who.csv": "atc_who.csv",
    "00_donnees_sources/export_remed_complet.csv": "export_remed_complet.csv",
}
for src, dst in sources.items():
    shutil.copy(os.path.join(ROOT, src), os.path.join(WORK, dst))

shutil.copytree(os.path.join(ROOT, "00_donnees_sources/csv_export"), os.path.join(WORK, "csv_export"))
shutil.copytree(os.path.join(ROOT, "00_donnees_sources/thesaurus_ansm"), os.path.join(WORK, "thesaurus_ansm"))
print("Fichiers sources copiés dans _run_pipeline/.")

# ---------------------------------------------------------------------------
# 1. Mapping ATC
# ---------------------------------------------------------------------------
etape("1. Mapping ATC (mapping_atc.py -> mapping_atc_v2.py -> finaliser_mapping_atc.py)")
lancer("01_mapping_atc/mapping_atc.py")
lancer("01_mapping_atc/mapping_atc_v2.py")
lancer("01_mapping_atc/finaliser_mapping_atc.py")

# ---------------------------------------------------------------------------
# 2. Fusion export + ATC (remplace fusionner_export_atc.py, manquant du dossier)
# ---------------------------------------------------------------------------
etape("2. Fusion des délivrances avec le mapping ATC")
export = pd.read_csv(os.path.join(WORK, "export_remed_complet.csv"))
mapping = pd.read_csv(os.path.join(WORK, "mapping_atc_final.csv"))[["medicament_id", "code_atc"]]
df = export.merge(mapping, on="medicament_id", how="left")
df["code_atc"] = df["code_atc"].fillna("ATC_INCONNU")
df.to_csv(os.path.join(WORK, "export_remed_avec_atc.csv"), index=False)

af = df[df["medicament_nom"].str.contains("ACIDE FOLIQUE", case=False, na=False)]
if len(af):
    codes = af["code_atc"].unique()
    print(f"Vérification ACIDE FOLIQUE -> {list(codes)}")
    if "A01AD05" in codes:
        raise RuntimeError("❌ BUG TOUJOURS PRÉSENT : ACIDE FOLIQUE mappé à A01AD05 (aspirine). Arrêt.")
    print("✅ ACIDE FOLIQUE correctement mappé.")

# ---------------------------------------------------------------------------
# 3. Interactions ANSM
# ---------------------------------------------------------------------------
etape("3. Construction de la table d'interactions ANSM")
lancer("02_interactions_ansm/construire_interactions_ansm.py")

# ---------------------------------------------------------------------------
# 4. Pipeline symbolique (Marches 1 & 2)
# ---------------------------------------------------------------------------
etape("4. Pipeline symbolique : couverture réelle + graphe")
lancer("03_pipeline_symbolique/run_pipeline_reel.py")

# ---------------------------------------------------------------------------
# 5. Interactions pour le GNN + arêtes vectorisées
# ---------------------------------------------------------------------------
etape("5. Filtrage des interactions pour le graphe GNN")
lancer("99_a_verifier/créer_interactions_graphe.py")

etape("6. Vectorisation des arêtes (.npy)")
lancer("99_a_verifier/preparer_aretes_gnn.py")

# ---------------------------------------------------------------------------
# 7. Assemblage du graphe + export des artefacts GNN
# ---------------------------------------------------------------------------
etape("7. Assemblage du graphe hétérogène")
lancer("99_a_verifier/assembler_graphe.py")

etape("8. Entraînement final + export des artefacts")
lancer("04_gnn/exporter_artefacts.py")

# ---------------------------------------------------------------------------
# 9. Vérification finale de non-régression
# ---------------------------------------------------------------------------
etape("9. Vérification finale")
nodes = pd.read_csv(os.path.join(WORK, "nodes_reel.csv"))
if "B03BB01" not in nodes.get("atc", pd.Series(dtype=str)).values and \
   nodes[nodes["label"].str.contains("ACIDE FOLIQUE", case=False, na=False)].empty is False:
    codes_af = nodes[nodes["label"].str.contains("ACIDE FOLIQUE", case=False, na=False)]["atc"].unique()
    print(f"Code(s) ATC de l'acide folique dans le graphe final : {list(codes_af)}")

# ---------------------------------------------------------------------------
# 10. Redistribution des fichiers finaux vers les dossiers organisés
# ---------------------------------------------------------------------------
etape("10. Redistribution des résultats vers les dossiers organisés")

copies = {
    "mapping_atc_final.csv": "01_mapping_atc/mapping_atc_final.csv",
    "export_remed_avec_atc.csv": "01_mapping_atc/export_remed_avec_atc.csv",
    "interactions_ansm.csv": "02_interactions_ansm/interactions_ansm.csv",
    "data_prete_pour_graphe_reel.csv": "03_pipeline_symbolique/data_prete_pour_graphe_reel.csv",
    "nodes_reel.csv": "03_pipeline_symbolique/nodes_reel.csv",
    "edges_reel.csv": "03_pipeline_symbolique/edges_reel.csv",
    "interactions_thesaurus_global.csv": "02_interactions_ansm/créer_interaction_graphe/interactions_thesaurus_global.csv",
    "med_interactions_edge_index.npy": "99_a_verifier/preparer_aretes_gnn/med_interactions_edge_index.npy",
    "med_interactions_edge_attr.npy": "99_a_verifier/preparer_aretes_gnn/med_interactions_edge_attr.npy",
    "graphe_heterogene_complet.pt": "04_gnn/graphe_heterogene_complet.pt",
    "mappings_partiels_assemblage.json": "04_gnn/mappings_partiels_assemblage.json",
    "remed_gnn_weights.pt": "04_gnn/exporter_artefacts/remed_gnn_weights.pt",
    "mappings_remed.json": "04_gnn/exporter_artefacts/mappings_remed.json",
}
for src, dst in copies.items():
    dst_path = os.path.join(ROOT, dst)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy(os.path.join(WORK, src), dst_path)
    print(f"  {src} -> {dst}")

# Copie supplémentaire pour le microservice, qui attend ces 3 fichiers à la
# racine de son propre dossier (chemins par défaut d'app_api.py)
for f in ["graphe_heterogene_complet.pt", "remed_gnn_weights.pt", "mappings_remed.json"]:
    shutil.copy(os.path.join(WORK, f), os.path.join(ROOT, "05_api_deploiement", f))
print("  + copie des 3 artefacts finaux dans 05_api_deploiement/ (pour app_api.py)")

print("\n" + "=" * 70)
print("✅ PIPELINE COMPLET EXÉCUTÉ AVEC SUCCÈS, DE BOUT EN BOUT")
print("=" * 70)
print("Vous pouvez maintenant lancer app_api.py depuis 05_api_deploiement/.")