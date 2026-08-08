import os
import re
import pandas as pd
from difflib import get_close_matches

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_INPUT = os.path.join(BASE_DIR, "medication.csv")
CSV_OUTPUT = os.path.join(BASE_DIR, "mapping_atc_final.csv")

df = pd.read_csv(CSV_INPUT)
col_nom = 'name' if 'name' in df.columns else df.columns[0]

df['nom_original'] = df[col_nom].astype(str)

# 1. Nettoyage Regex : Extraire uniquement la racine du nom (retirer dosages, formes, conditionnements)
def nettoyer_chaine(texte):
    texte = texte.upper()
    # Supprimer formes galéniques et formes fréquentes
    texte = re.sub(r'\b(CPR|GEL|GE|GELULE|INJ|AMP|COLLYRE|SOL|SIR|SIROP|FL|PLQ|PQT|BTE|DOSES|UI|MG|G|ML|MCG|LP)\b', ' ', texte)
    # Supprimer les chiffres et caractères spéciaux
    texte = re.sub(r'[^A-Z\s]', ' ', texte)
    # Supprimer les espaces multiples
    texte = re.sub(r'\s+', ' ', texte).strip()
    return texte

df['nom_clean'] = df['nom_original'].apply(nettoyer_chaine)
df['methode_matching'] = 'non_resolu_a_verifier'
df['code_atc'] = None

# 2. Dictionnaire de corrections spécifiques / Manuel
CORRECTIONS_EXACTES = {
    "RESTINE": "RESITUNE",
    "RESTIUNE": "RESITUNE",
    "UVEODOSE": "UVEDOSE",
    "MUGOCYNE": "MUCOGYNE",
    "INDOCOLOGYRE": "INDOCOLLYRE",
    "INDOCCOLLYRE": "INDOCOLLYRE",
    "VERAFERIL": "VERAPAMIL",
    "VEROFERIL": "VERAPAMIL",
    "NEUROPLUS": "N06BX",
    "OXYFLUX": "C04AD",
    "GASTROCALM": "A02A",
    "PEDIAVITE": "A11AA",
    "ALLERFREE": "R06AX",
    "DERMACARE": "D02A",
    "DOLORIN": "N02BE01",
    "AMOXI": "J01CA04",
    "DYNAMOGEN": "A15"
}

DISPOSITIFS_KEYWORDS = [
    "CHAUSSETTE", "COLLANT", "CONTENTION", "AIGUILLE", "PANSEMENT", 
    "MEPILEX", "SORBACT", "TEGADERM", "SPARADRAP", "SONDE", "MICROPORE", 
    "COMPRESS", "SET", "SERINGUE", "Bande", "GANT", "ALCOOL"
]

BRUIT_KEYWORDS = ["TEST", "LOREM", "BLABLA", "ILLISIBLE", "WATER", "LINE"]

# 3. Traitement
def classifier_ligne(row):
    nom_orig = row['nom_original'].upper()
    nom_clean = row['nom_clean']
    
    # Check 1 : Bruit / Fichiers de test
    if any(b in nom_orig for b in BRUIT_KEYWORDS) or len(nom_clean) < 2:
        row['methode_matching'] = 'exclu_bruit_test'
        return row
        
    # Check 2 : Dispositifs Médicaux
    if any(d in nom_orig for d in DISPOSITIFS_KEYWORDS):
        row['methode_matching'] = 'exclu_dispositif'
        row['code_atc'] = 'DISPOSITIF_MEDICAL'
        return row

    # Check 3 : Dictionnaire manuel direct ou partiel
    for k, val in CORRECTIONS_EXACTES.items():
        if k in nom_clean:
            if len(val) <= 7 and val[0].isalpha() and val[1:].isalnum():
                row['methode_matching'] = 'manuel_code_direct'
                row['code_atc'] = val
            else:
                row['nom_nettoye'] = val
                row['methode_matching'] = 'patch_correction_nom'
            return row

    return row

df = df.apply(classifier_ligne, axis=1)

# 4. Statistiques finales
print("\n=== BILAN FINAL NIVEAU 1 & 2 ===")
print(df['methode_matching'].value_counts())

total = len(df)
exclus = df['methode_matching'].isin(['exclu_dispositif', 'exclu_bruit_test']).sum()
resolus = (df['methode_matching'] != 'non_resolu_a_verifier').sum() - exclus
non_resolus = (df['methode_matching'] == 'non_resolu_a_verifier').sum()

print(f"\nTotal : {total}")
print(f"Résolus/Alignés : {resolus}")
print(f"Exclus du Graph (Bruit/DM) : {exclus}")
print(f"Reste à mapper par DCI/Fuzzy : {non_resolus}")

# Nettoyage des colonnes temporaires
df_export = df.drop(columns=['nom_clean'])
df_export.to_csv(CSV_OUTPUT, index=False)