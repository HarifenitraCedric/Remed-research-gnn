import os
import re
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_INPUT = os.path.join(BASE_DIR, "mapping_atc_final.csv")
FILE_OUTPUT = os.path.join(BASE_DIR, "mapping_atc_final.csv")
FILE_READY = os.path.join(BASE_DIR, "medications_atc_ready_for_gnn.csv")

df = pd.read_csv(FILE_INPUT)

# Dictionnaire des classes thérapeutiques globales (Niveau 2/3 ATC)
FALLBACK_CLASSES = {
    # Vitamines & Minéraux
    "VITAMINE": ("A11", "fallback_classe_vitamines"),
    "VIT": ("A11", "fallback_classe_vitamines"),
    "CALCIUM": ("A12AA", "fallback_classe_mineraux"),
    "FER": ("B03AA", "fallback_classe_fer"),
    "MAGNESIUM": ("A12CC", "fallback_classe_mineraux"),
    "ZINC": ("A12CB", "fallback_classe_mineraux"),
    
    # Dermatologie / Antiseptiques
    "CREME": ("D02A", "fallback_classe_dermato"),
    "POMMADE": ("D02A", "fallback_classe_dermato"),
    "GEL DERM": ("D02A", "fallback_classe_dermato"),
    "BETADINE": ("D08AG02", "fallback_classe_antiseptique"),
    "ALCOOL": ("D08AX08", "fallback_classe_antiseptique"),
    "EAU OXYGENEE": ("D08AX01", "fallback_classe_antiseptique"),
    
    # Digestif / Solutés / Réhydratation
    "SRO": ("A07CA", "fallback_classe_rehydratation"),
    "SOLUTE": ("B05BB", "fallback_classe_solute"),
    "SERUM": ("B05BB", "fallback_classe_solute"),
    "HUILE": ("A02A", "fallback_classe_digestif"),
    "CHARBON": ("A07BA01", "fallback_classe_adsorbant"),
    
    # Respiratoire / Toux / ORL
    "SIROP": ("R05X", "fallback_classe_respiratoire"),
    "TOUX": ("R05X", "fallback_classe_respiratoire"),
    "COLLYRE": ("S01X", "fallback_classe_ophtalmo"),
    "NOSE": ("R01A", "fallback_classe_nasal"),
    "SPRAY": ("R01A", "fallback_classe_nasal"),
}

non_resolus = df['methode_matching'] == 'non_resolu_a_verifier'
print(f"Application du fallback par classe sur {non_resolus.sum()} entrées...")

count_fallback = 0

for idx, row in df[non_resolus].iterrows():
    nom = str(row['nom_original']).upper()
    
    for kw, (code, methode) in FALLBACK_CLASSES.items():
        if kw in nom:
            df.at[idx, 'code_atc'] = code
            df.at[idx, 'methode_matching'] = methode
            count_fallback += 1
            break

# Sécurisation des résidus ultimes (Non mappables)
df.loc[df['methode_matching'] == 'non_resolu_a_verifier', 'methode_matching'] = 'non_mappable_ignore'

# 2. Séparation des données exploitables pour le GNN
df_gnn_ready = df[
    (df['code_atc'].notnull()) & 
    (~df['methode_matching'].isin(['exclu_dispositif', 'exclu_bruit_test', 'non_mappable_ignore']))
].copy()

# Enregistrement
df.to_csv(FILE_OUTPUT, index=False)
df_gnn_ready.to_csv(FILE_READY, index=False)

# 3. Bilan Global
print("\n=================== BILAN FINAL DU PIPELINE ATC ===================")
print(df['methode_matching'].value_counts())

exclus = df['methode_matching'].isin(['exclu_dispositif', 'exclu_bruit_test']).sum()
non_mappables = (df['methode_matching'] == 'non_mappable_ignore').sum()
total_init = len(df)
total_pertinent = total_init - exclus
resolus = len(df_gnn_ready)

taux_pertinent = (resolus / total_pertinent) * 100
taux_global = (resolus / total_init) * 100

print("\n-------------------------------------------------------------------")
print(f"Nombre total de lignes initiales : {total_init}")
print(f"Exclus du Graph (Bruit / Dispositifs Médicaux) : {exclus}")
print(f"Médicaments pertinents identifiés : {total_pertinent}")
print(f"--> MAPPÉS AVEC CODE ATC (Prêts pour le GNN) : {resolus} ({taux_pertinent:.2f}%)")
print(f"--> Non mappables / Ignorés : {non_mappables}")
print("-------------------------------------------------------------------")
print(f"Fichier complet mis à jour : {FILE_OUTPUT}")
print(f"Fichier exporté pour le pipeline GNN/FastAPI : {FILE_READY}")