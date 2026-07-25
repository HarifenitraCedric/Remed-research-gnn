import pandas as pd

# 1. Charger les nœuds réels
nodes_df = pd.read_csv("nodes_reel.csv")
colonnes = nodes_df.columns.tolist()
print(f"Colonnes détectées dans nodes_reel.csv : {colonnes}")

# Détection automatique des colonnes pour l'identifiant (ATC) et le type
col_id = 'id' if 'id' in colonnes else (colonnes[0] if colonnes else None)
col_type = 'type' if 'type' in colonnes else (colonnes[1] if len(colonnes) > 1 else None)

# Si tu as des colonnes nommées explicitement, on les prend en priorité
for col in colonnes:
    if col.lower() in ['node_id', 'code_atc', 'atc']:
        col_id = col
    if col.lower() in ['type', 'category', 'nature']:
        col_type = col

print(f"-> Utilisation de '{col_id}' pour l'identifiant et '{col_type}' pour le type.")

# 2. Extraire les codes ATC présents
# On gère le cas où la distinction médicament/patient utilise une autre valeur (ex: 'Drug', 'medicament', 1...)
med_mask = nodes_df[col_type].astype(str).str.lower().str.contains('med|drug|medicament', na=False) if col_type else pd.Series([True] * len(nodes_df))

atc_presents = set(nodes_df[med_mask][col_id].dropna().unique())
print(f"Nombre de codes ATC uniques identifiés dans ton graphe : {len(atc_presents)}")

# 3. Charger le master thésaurus et filtrer
thesaurus_df = pd.read_csv("interactions_ansm.csv")

interactions_cibles = thesaurus_df[
    thesaurus_df['atc_1'].isin(atc_presents) & 
    thesaurus_df['atc_2'].isin(atc_presents)
]

print(f"Nombre d'interactions médicales ANSM trouvées pour tes couples : {len(interactions_cibles)}")

# 4. Sauvegarde
interactions_cibles.to_csv("interactions_thesaurus_global.csv", index=False)
print("Fichier 'interactions_thesaurus_global.csv' généré avec succès !")