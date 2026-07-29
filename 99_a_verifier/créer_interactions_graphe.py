import pandas as pd
import os

# 1. Charger les nœuds réels
if os.path.exists("nodes_reel.csv"):
    nodes_df = pd.read_csv("nodes_reel.csv")
else:
    nodes_df = pd.read_csv("03_pipeline_symbolique/nodes_reel.csv")
colonnes = nodes_df.columns.tolist()
print(f"Colonnes détectées dans nodes_reel.csv : {colonnes}")

# Détection automatique des colonnes pour l'identifiant (ATC) et le type
col_id = 'id' if 'id' in colonnes else (colonnes[0] if colonnes else None)
col_type = 'type' if 'type' in colonnes else (colonnes[1] if len(colonnes) > 1 else None)
def lire_csv_safe(chemin_relatif):
    """Cherche d'abord à la racine du dossier d'exécution (_run_pipeline),

    sinon tente le chemin relatif d'origine.
    """
    nom_fichier = os.path.basename(chemin_relatif)
    if os.path.exists(nom_fichier):
        return pd.read_csv(nom_fichier)
    elif os.path.exists(chemin_relatif):
        return pd.read_csv(chemin_relatif)
    else:
        raise FileNotFoundError(
            f"Impossible de trouver {nom_fichier} ou {chemin_relatif}"
        )

# Si tu as des colonnes nommées explicitement, on les prend en priorité
for col in colonnes:
    if col.lower() in ['node_id', 'code_atc', 'atc']:
        col_id = col
    if col.lower() in ['type', 'category', 'nature']:
        col_type = col

print(f"-> Utilisation de '{col_id}' pour l'identifiant et '{col_type}' pour le type.")
import os


def sauvegarder_csv_safe(df, chemin_relatif):
    """Si on est dans _run_pipeline, sauvegarde à la racine.

    Sinon crée le dossier parent s'il n'existe pas puis sauvegarde.
    """
    nom_fichier = os.path.basename(chemin_relatif)

    # Si on est dans l'environnement orchestré (_run_pipeline)
    if os.path.basename(os.getcwd()) == "_run_pipeline" or not os.path.exists(
        os.path.dirname(chemin_relatif)
    ):
        # On sauvegarde à plat dans le dossier courant
        df.to_csv(nom_fichier, index=False)
    else:
        # En mode autonome, on s'assure que le dossier existe
        os.makedirs(os.path.dirname(chemin_relatif), exist_ok=True)
        df.to_csv(chemin_relatif, index=False)

# 2. Extraire les codes ATC présents
# On gère le cas où la distinction médicament/patient utilise une autre valeur (ex: 'Drug', 'medicament', 1...)
med_mask = nodes_df[col_type].astype(str).str.lower().str.contains('med|drug|medicament', na=False) if col_type else pd.Series([True] * len(nodes_df))

atc_presents = set(nodes_df[med_mask][col_id].dropna().unique())
print(f"Nombre de codes ATC uniques identifiés dans ton graphe : {len(atc_presents)}")

# 3. Charger le master thésaurus et filtrer
thesaurus_df = lire_csv_safe("02_interactions_ansm/interactions_ansm.csv")

interactions_cibles = thesaurus_df[
    thesaurus_df['atc_1'].isin(atc_presents) & 
    thesaurus_df['atc_2'].isin(atc_presents)
]

print(f"Nombre d'interactions médicales ANSM trouvées pour tes couples : {len(interactions_cibles)}")

# 4. Sauvegarde
# ANCIEN CODE :
# interactions_cibles.to_csv("02_interactions_ansm/créer_interaction_graphe/interactions_thesaurus_global.csv", index=False)

# NOUVEAU CODE :
sauvegarder_csv_safe(
    interactions_cibles,
    "02_interactions_ansm/créer_interaction_graphe/interactions_thesaurus_global.csv",
)
print("Fichier 'interactions_thesaurus_global.csv' généré avec succès !")