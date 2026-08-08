import os
import re
import pandas as pd
from difflib import get_close_matches

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FILE_MEDICATIONS = os.path.join(BASE_DIR, "mapping_atc_final.csv")
FILE_ATC_REF = os.path.join(BASE_DIR, "atc_who.csv")
FILE_OUTPUT = os.path.join(BASE_DIR, "mapping_atc_final.csv")

# 1. Chargement des données
df_med = pd.read_csv(FILE_MEDICATIONS)
df_atc = pd.read_csv(FILE_ATC_REF)

print(f"Base médicaments : {len(df_med)} lignes")
print(f"Base de référence ATC WHO : {len(df_atc)} lignes")

# 2. Construction d'une table plate de tous les niveaux ATC (Niveau 5 prioritaire)
atc_records = []

for idx, row in df_atc.iterrows():
    # Niveau 5 : Molécule / Substance active spécifique (ex: J01CA04 -> Amoxicillin)
    if pd.notnull(row.get('atc5_code')) and pd.notnull(row.get('atc5_description')):
        atc_records.append({'code': str(row['atc5_code']).strip(), 'label': str(row['atc5_description']).strip()})
    # Niveau 4 : Sous-groupe chimique/thérapeutique
    elif pd.notnull(row.get('atc4_code')) and pd.notnull(row.get('atc4_description')):
        atc_records.append({'code': str(row['atc4_code']).strip(), 'label': str(row['atc4_description']).strip()})
    # Niveau 3 : Sous-groupe pharmacologique
    elif pd.notnull(row.get('atc3_code')) and pd.notnull(row.get('atc3_description')):
        atc_records.append({'code': str(row['atc3_code']).strip(), 'label': str(row['atc3_description']).strip()})

df_flat_atc = pd.DataFrame(atc_records).drop_duplicates(subset=['label'])

# Dictionnaire de référence ATC nettoyé
df_flat_atc['label_clean'] = df_flat_atc['label'].str.upper().str.strip()
atc_dict = dict(zip(df_flat_atc['label_clean'], df_flat_atc['code']))
ref_labels = list(atc_dict.keys())

print(f"Référentiel ATC aplati : {len(ref_labels)} termes de substances/groupes chargés.")

# 3. Mappages des équivalences FR/EN pour les DCI courantes
SYNONYMES_FR_EN = {
    "PARACETAMOL": "PARACETAMOL",
    "AMOXICILLINE": "AMOXICILLIN",
    "IBUPROFENE": "IBUPROFEN",
    "METFORMINE": "METFORMIN",
    "AZITHROMYCINE": "AZITHROMYCIN",
    "CIPROFLOXACINE": "CIPROFLOXACIN",
    "CEFTRIAXONE": "CEFTRIAXONE",
    "ERYTHROMYCINE": "ERYTHROMYCIN",
    "SALBUTAMOL": "SALBUTAMOL",
    "OMEPRAZOLE": "OMEPRAZOLE",
    "AMLODIPINE": "AMLODIPINE",
    "VALSARTAN": "VALSARTAN",
    "LOSARTAN": "LOSARTAN",
    "PREDNISOLONE": "PREDNISOLONE",
    "DICLOFENAC": "DICLOFENAC",
    "ACICLOVIR": "ACICLOVIR",
    "METRONIDAZOLE": "METRONIDAZOLE"
}

# Nettoyage des dosages et suffixes commerciaux
def extraire_racine(nom):
    nom = str(nom).upper()
    # Retirer dosages, unités et formes galéniques
    nom = re.sub(r'\b(\d+([\.,]\d+)?\s*(MG|G|ML|MCG|UI|%))\b', ' ', nom)
    nom = re.sub(r'\b(CPR|GEL|GE|GELULE|INJ|AMP|COLLYRE|SOL|SIR|SIROP|FL|PLQ|PQT|BTE|DOSES|LP|AMPOULE|SACHET|PERFUSION|NOVO|LABO)\b', ' ', nom)
    nom = re.sub(r'[^A-Z\s]', ' ', nom)
    return re.sub(r'\s+', ' ', nom).strip()

# 4. Exécution du Matching
non_resolus = df_med['methode_matching'] == 'non_resolu_a_verifier'
print(f"\nExécution du matching sur {non_resolus.sum()} entrées non résolues...")

count_dci_exact = 0
count_fuzzy = 0

for idx, row in df_med[non_resolus].iterrows():
    racine = extraire_racine(row['nom_original'])
    
    if len(racine) < 3:
        continue
        
    mots = racine.split()
    matched = False
    
    # Étape A : Test Exact DCI FR/EN
    for mot in mots:
        # Conversion du nom générique FR vers EN si présent
        mot_en = SYNONYMES_FR_EN.get(mot, mot)
        
        if len(mot_en) >= 3 and mot_en in atc_dict:
            df_med.at[idx, 'code_atc'] = atc_dict[mot_en]
            df_med.at[idx, 'methode_matching'] = 'match_exact_dci'
            count_dci_exact += 1
            matched = True
            break
            
    if matched:
        continue

    # Étape B : Fuzzy Matching (Levenshtein sur racine entière ou premier terme)
    matches = get_close_matches(racine, ref_labels, n=1, cutoff=0.75)
    if not matches and len(mots) > 0:
        matches = get_close_matches(mots[0], ref_labels, n=1, cutoff=0.75)

    if matches:
        best_match = matches[0]
        df_med.at[idx, 'code_atc'] = atc_dict[best_match]
        df_med.at[idx, 'methode_matching'] = 'fuzzy_matching_atc'
        count_fuzzy += 1

# 5. Bilans & Sauvegarde
print("\n=== BILAN APRES MATCHING COMPLET (LEVEL 5 & DCI) ===")
print(df_med['methode_matching'].value_counts())

exclus = df_med['methode_matching'].isin(['exclu_dispositif', 'exclu_bruit_test']).sum()
total_pertinent = len(df_med) - exclus
resolus = (df_med['code_atc'].notnull() & ~df_med['methode_matching'].isin(['exclu_dispositif', 'exclu_bruit_test'])).sum()
reste = (df_med['methode_matching'] == 'non_resolu_a_verifier').sum()

taux = (resolus / total_pertinent) * 100 if total_pertinent > 0 else 0

print(f"\nTotal Médicaments Pertinents : {total_pertinent}")
print(f"Codes ATC assignés : {resolus} ({taux:.2f}%)")
print(f"Reste non résolu : {reste}")

df_med.to_csv(FILE_OUTPUT, index=False)
print(f"\nFichier mis à jour : {FILE_OUTPUT}")