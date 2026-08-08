import os
import re
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_INPUT = os.path.join(BASE_DIR, "mapping_atc_final.csv")
FILE_READY = os.path.join(BASE_DIR, "medications_atc_ready_for_gnn.csv")

df = pd.read_csv(FILE_INPUT)

# Exclure uniquement le bruit explicite et les dispositifs (149 lignes)
exclus_mask = df['methode_matching'].isin(['exclu_dispositif', 'exclu_bruit_test'])
df_pertinent = df[~exclus_mask].copy()

# =====================================================================
# 1. DICTIONNAIRE ÉTENDU MARQUES & ASSOCIATIONS (Couverture ciblée)
# =====================================================================
EXTENDED_BRAND_MAP = {
    # Antalgiques, Anti-inflammatoires, Rhumatologie
    "PARACETAMOL": "N02BE01", "ANTALGAN": "N02BE01", "IBUPROFENE": "M01AE01",
    "KETOPROFENE": "M01AE03", "DICLOFENAC": "M01AB05", "TIAPROFENIQUE": "M01AE11",
    "PIROXICAM": "M01AC01", "MELOXICAM": "M01AC06", "NEFOPAM": "N02BG06",
    "ACUPAN": "N02BG06", "TRAMADOL": "N02AX02", "TOPALGIC": "N02AX02",
    "BI-PROFENID": "M01AE03", "CELEBREX": "M01AH01", "ARTHROTEC": "M01AB55",

    # Cardio-vasculaire & Hypertension
    "AMLODIPINE": "C08CA01", "AMOR": "C08CA01", "ATENOLOL": "C07AB03",
    "BISOPROLOL": "C07AB07", "RAMIPRIL": "C09AA05", "PERINDOPRIL": "C09AA04",
    "VALSARTAN": "C09CA03", "IRBESARTAN": "C09CA04", "LOSARTAN": "C09CA01",
    "FUROSEMIDE": "C03CA01", "HYDROCHLOROTHIAZIDE": "C03AA03", "SPIRONOLACTONE": "C03DA01",
    "LOVENOX": "B01AB05", "HEPARINE": "B01AB01", "SINTROM": "B01AA07",

    # Infectiologie & Parasitologie
    "AMOXICILLINE": "J01CA04", "AMPICILLINE": "J01CA01", "CEFTRIAXONE": "J01DD04",
    "CEFIXIME": "J01DD08", "CIPROFLOXACINE": "J01MA02", "LEVOFLOXACINE": "J01MA12",
    "AZITHROMYCINE": "J01FA10", "CLARITHROMYCINE": "J01FA09", "ERYTHROMYCINE": "J01FA01",
    "GENTAMICINE": "J01GB03", "METRONIDAZOLE": "P01AB01", "ALBENDAZOLE": "P02CA03",
    "MEBENDAZOLE": "P02CA01", "ARTEMETHER": "P01BF01", "QUstat": "P01BC01",
    "QUININE": "P01BC01", "COARTEM": "P01BF01", "ARTEQUICK": "P01BF01",

    # Neuro-Psychiatrie & Anxiolytiques
    "DIAZEPAM": "N05BA01", "OXAZEPAM": "N05BA04", "LORAZEPAM": "N05BA06",
    "ALPRAZOLAM": "N05BA12", "BROMAZEPAM": "N05BA08", "CHLORPROMAZINE": "N05AA01",
    "HALOPERIDOL": "N05AD01", "RISPERIDONE": "N05AX08", "OLANZAPINE": "N05AH03",
    "AMITRIPTYLINE": "N06AA09", "FLUOXETINE": "N06AB03", "PAROXETINE": "N06AB05",

    # Gastro-Entérologie & Métabolisme
    "OMEPRAZOLE": "A02BC01", "PANTOPRAZOLE": "A02BC02", "ESOMEPRAZOLE": "A02BC05",
    "METFORMINE": "A10BA02", "GLIBENCLAMIDE": "A10BB01", "GLIMEPIRIDE": "A10BB12",
    "INSULINE": "A10A", "PHOSPHALUGEL": "A02AB10", "DOMPERIDONE": "A03FA03",

    # Pneumologie, Dermatologie & Divers
    "SALBUTAMOL": "R03AC02", "FLUTICASONE": "R03BA05", "PREDNISOLONE": "H02AB06",
    "DEXAMETHASONE": "H02AB02", "HYDROCORTISONE": "H02AB09", "CETIRIZINE": "R06AE07",
    "LORATADINE": "R06AX13", "DEXCHLORPHENIRAMINE": "R06AB02", "POLYSORBATE": "A07",
}

# =====================================================================
# 2. MATCHING PAR RACINES THÉRAPEUTIQUES ET SUFFIXES DCI
# =====================================================================
DCI_SUFFIXES = [
    (r"CAINE$", "N01BB", "fallback_dci_anesthesique"),
    (r"CILLINE$", "J01CA", "fallback_dci_penicilline"),
    (r"MYCINE$", "J01FA", "fallback_dci_macrolide"),
    (r"FLOXACINE$", "J01MA", "fallback_dci_quinolone"),
    (r"ZOLE$", "J02AC", "fallback_dci_antifongique"),
    (r"VIR$", "J05AB", "fallback_dci_antiviral"),
    (r"OLOL$", "C07AB", "fallback_dci_betabloquant"),
    (r"PRIL$", "C09AA", "fallback_dci_ieca"),
    (r"SARTAN$", "C09CA", "fallback_dci_ara2"),
    (r"DIPINE$", "C08CA", "fallback_dci_inhibiteur_calcique"),
    (r"STATINE$", "C10AA", "fallback_dci_statine"),
    (r"PRAZOLE$", "A02BC", "fallback_dci_ipp"),
    (r"TIDINE$", "A02BA", "fallback_dci_anti_h2"),
    (r"ZOLAM$", "N05BA", "fallback_dci_benzodiazepine"),
    (r"ZEPAM$", "N05BA", "fallback_dci_benzodiazepine"),
    (r"SONE$", "H02AB", "fallback_dci_corticoide"),
    (r"LONE$", "H02AB", "fallback_dci_corticoide"),
]

# =====================================================================
# 3. FALLBACK CIBLÉ PAR DOMAINE / ORGANEE (Niveau 1 & 2 ATC)
# =====================================================================
ORGANIC_FALLBACKS = [
    (r"INJECTABLE|PERFUSION|SERUM|SOLUTE|NACL|GLUCOSE", "B05B", "fallback_solutes_solutions"),
    (r"EYE|OPHT|COLLYRE|POMMADE OPHT", "S01X", "fallback_ophtalmologie"),
    (r"EAR|OTIQUE|GOUTTES OTIQUES", "S02X", "fallback_otologie"),
    (r"DERM|CREME|POMMADE|GEL|LAIT|SHAMPOOING", "D11A", "fallback_dermatologie"),
    (r"SIROP|TOUX|EXPECTORANT|BRONCHIQUE", "R05X", "fallback_respiratoire"),
    (r"VITAMINE|MULTIVITAMINES|FORTE|TONIC|COMPLEMENT", "A11AA", "fallback_vitamines_complements"),
    (r"SAVON|DISINFECTANT|ANTISEPTIQUE|ALCOOL|BETADINE", "D08A", "fallback_antiseptiques"),
    (r"PLANTE|TISANE|EXTRAIT|HUILE|HOMEO", "V03AX", "fallback_phytotherapie_divers"),
]

print(f"Lancement de la résolution à 100% sur {len(df_pertinent)} médicaments pertinents...")

count_brand, count_suffix, count_organic, count_generic = 0, 0, 0, 0

for idx in df_pertinent.index:
    code_actuel = df_pertinent.at[idx, 'code_atc']
    methode = str(df_pertinent.at[idx, 'methode_matching'])
    
    # Si non résolu ou ignoré précédemment
    if pd.isna(code_actuel) or methode in ['non_resolu_a_verifier', 'non_mappable_ignore']:
        nom = str(df_pertinent.at[idx, 'nom_original']).upper().strip()
        matched = False

        # Phase A : Dictionnaire étendu
        for brand, code in EXTENDED_BRAND_MAP.items():
            if brand in nom:
                df_pertinent.at[idx, 'code_atc'] = code
                df_pertinent.at[idx, 'methode_matching'] = 'match_marque_etendue'
                count_brand += 1
                matched = True
                break

        # Phase B : Analyse des suffixes DCI
        if not matched:
            for pattern, code, tag in DCI_SUFFIXES:
                if re.search(pattern, nom.split()[0]):
                    df_pertinent.at[idx, 'code_atc'] = code
                    df_pertinent.at[idx, 'methode_matching'] = tag
                    count_suffix += 1
                    matched = True
                    break

        # Phase C : Fallback organique / forme galénique
        if not matched:
            for pattern, code, tag in ORGANIC_FALLBACKS:
                if re.search(pattern, nom):
                    df_pertinent.at[idx, 'code_atc'] = code
                    df_pertinent.at[idx, 'methode_matching'] = tag
                    count_organic += 1
                    matched = True
                    break

        # Phase D : Attribution générique de secours (Système Divers / V03AX)
        if not matched:
            df_pertinent.at[idx, 'code_atc'] = 'V03AX'
            df_pertinent.at[idx, 'methode_matching'] = 'fallback_generique_v03'
            count_generic += 1

# Sauvegarde du dataset 100% prêt
df_pertinent.to_csv(FILE_READY, index=False)

print("\n=================== BILAN FINAL (100% COUVERTURE) ===================")
print(df_pertinent['methode_matching'].value_counts())
print("-------------------------------------------------------------------")
total_mats = len(df_pertinent)
nb_codes = df_pertinent['code_atc'].notnull().sum()
taux = (nb_codes / total_mats) * 100

print(f"Nombre de médicaments pertinents : {total_mats}")
print(f"Codes ATC attribués               : {nb_codes} ({taux:.2f}%)")
print(f"Fichier final généré             : {FILE_READY}")
print("===================================================================")