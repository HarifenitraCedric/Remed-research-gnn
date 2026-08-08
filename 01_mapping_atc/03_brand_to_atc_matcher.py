import os
import re
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_INPUT = os.path.join(BASE_DIR, "mapping_atc_final.csv")
FILE_OUTPUT = os.path.join(BASE_DIR, "mapping_atc_final.csv")

df = pd.read_csv(FILE_INPUT)

# Dictionnaire de correspondance Marque -> Code ATC
BRAND_MAP = {
    # Antalgiques / Anti-inflammatoires
    "DOLIPRANE": "N02BE01", "EFFERALGAN": "N02BE01", "DAFALGAN": "N02BE01", 
    "ADVIL": "M01AE01", "NUROFEN": "M01AE01", "ANTADYS": "M01AG01",
    "VOLTARENE": "M01AB05", "APRANAX": "M01AE02", "BI-PROFENID": "M01AE03",
    "PROFENID": "M01AE03", "CELEBREX": "M01AH01", "CATAFLAM": "M01AB05",
    
    # Antibiotiques / Anti-infectieux
    "AUGMENTIN": "J01CR02", "CLAMOXYL": "J01CA04", "ORELOX": "J01DD13",
    "ZECLAR": "J01FA09", "ZITHROMAX": "J01FA10", "CIFLOX": "J01MA02",
    "PYOSTACINE": "J01FG01", "ROCEPHINE": "J01DD04", "BIORACEN": "J01FA10",
    "BACTRIM": "J01EE01", "FLAGYL": "P01AB01", "FUNGIZONE": "A07AA01",
    
    # Gastro-entérologie
    "SPASFON": "A03AX58", "SMECTA": "A07BC05", "MOPRAL": "A02BC01",
    "INIPOMP": "A02BC02", "EUPANTOL": "A02BC02", "OGASTORO": "A02BC03",
    "INEXIUM": "A02BC05", "IPRAALOX": "A02BC01", "GAVISCON": "A02BX13",
    "MOTILIUM": "A03FA03", "VOGALENE": "A04AD05", "PRIMPERAN": "A03FA01",
    "PANCREALIPASE": "A09AA02", "CREON": "A09AA02",
    
    # Cardiovasculaire / Veinotonique
    "DAFLON": "C05CA53", "EXFORGE": "C09DB01", "TAREG": "C09CA03",
    "APROVEL": "C09CA04", "COAPROVEL": "C09DA04", "CARDENTIEL": "C07AB07",
    "NEBILET": "C07AB12", "TENORMINE": "C07AB03", "KERLONE": "C07AB05",
    "LASILIX": "C03CA01", "ALDACTONE": "C03DA01", "KARDEGIC": "B01AC06",
    "PLAVIX": "B01AC04", "XARELTO": "B01AF01", "ELIQUIS": "B01AF02",
    
    # Système Nerveux / Psychiatrie
    "LEXOMIL": "N05BA08", "XANAX": "N05BA12", "VALIUM": "N05BA01",
    "STILNOX": "N05CF02", "IMOVANE": "N05CF01", "SEROPLEX": "N06AB10",
    "DEROXAT": "N06AB05", "ZOLOFT": "N06AB06", "PROZAC": "N06AB03",
    "LYRICA": "N03AX16", "NEURONTIN": "N03AX12", "LAROXYL": "N06AA09",
    "TEGRETOL": "N03AF01", "DEPAKINE": "N03AG01", "LAMICTAL": "N03AX09",
    
    # Voies Respiratoires / Antihistaminiques
    "VENTOLINE": "R03AC02", "SERETIDE": "R03AK06", "SYMBICORT": "R03AK07",
    "SINGULAIR": "R03DC03", "AERIUS": "R06AX27", "ZYRTEC": "R06AE07",
    "XYZALL": "R06AE09", "CLARITYNE": "R06AX13", "RHINOCORT": "R01AD05",
    
    # Métabolisme / Endocrinologie
    "STAGID": "A10BA02", "GLUCOPHAGE": "A10BA02", "AMAREL": "A10BB12",
    "LANTUS": "A10AE04", "NOVORAPID": "A10AB05", "LEVOTHYROX": "H03AA01",
    "TAZORAC": "D05AX05", "DEXAMETHASONE": "H02AB02", "SOLUPRED": "H02AB07"
}

non_resolus = df['methode_matching'] == 'non_resolu_a_verifier'
print(f"Recherche par marques sur {non_resolus.sum()} entrées...")

count_brand = 0

for idx, row in df[non_resolus].iterrows():
    nom = str(row['nom_original']).upper()
    
    for brand, code in BRAND_MAP.items():
        if brand in nom:
            df.at[idx, 'code_atc'] = code
            df.at[idx, 'methode_matching'] = 'match_marque_connue'
            count_brand += 1
            break

print(f"Marques identifiées et mappées : {count_brand}")

print("\n=== BILAN MIS A JOUR ===")
print(df['methode_matching'].value_counts())

exclus = df['methode_matching'].isin(['exclu_dispositif', 'exclu_bruit_test']).sum()
total_pertinent = len(df) - exclus
resolus = (df['code_atc'].notnull() & ~df['methode_matching'].isin(['exclu_dispositif', 'exclu_bruit_test'])).sum()
reste = (df['methode_matching'] == 'non_resolu_a_verifier').sum()

taux = (resolus / total_pertinent) * 100 if total_pertinent > 0 else 0

print(f"\nTotal Pertinent : {total_pertinent}")
print(f"Codes ATC assignés : {resolus} ({taux:.2f}%)")
print(f"Reste à vérifier : {reste}")

df.to_csv(FILE_OUTPUT, index=False)