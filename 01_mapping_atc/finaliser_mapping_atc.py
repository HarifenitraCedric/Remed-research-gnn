"""
Finalisation du mapping ATC :
  - Ajout de correspondances manuelles pour quelques marques très connues
    non capturées automatiquement (traçées comme 'manuel_marque_connue')
  - Exclusion explicite des dispositifs médicaux / non-médicaments
  - Marquage des artefacts de qualité de données (OCR, codes-barres) --
    JAMAIS de code ATC deviné pour ces cas, on documente l'incertitude
    plutôt que de la masquer (cf. §3 et §10 du cadrage)
"""

import pandas as pd

mapping = pd.read_csv("mapping_atc_final.csv")

# Correspondances manuelles pour des marques françaises très connues, non
# capturées car le nom de marque ne contient pas la DCI et
# denomination_substance n'est pas renseigné.
MANUEL_MARQUES = {
    "DOLIPRANE": "N02BE01",      # paracétamol
    "DAFALGAN": "N02BE01",       # paracétamol
    "TOVIAZ": "G04BD11",         # fésotérodine
    "MOVICOL": "A06AD65",        # macrogol (association)
    "ACIDE FOLIQUE": "B03BB01",  # acide folique -- exclu du matching automatique
                                  # depuis la correction du mot générique "acide"
}

def matcher_marque_manuelle(nom):
    nom_upper = str(nom).upper()
    for marque, code in MANUEL_MARQUES.items():
        if nom_upper.startswith(marque):
            return code
    return None

masque_manquant = mapping["code_atc"].isna()
for idx in mapping[masque_manquant].index:
    code = matcher_marque_manuelle(mapping.loc[idx, "medicament_nom"])
    if code:
        mapping.loc[idx, "code_atc"] = code
        mapping.loc[idx, "methode_matching"] = "manuel_marque_connue"

# Dispositifs médicaux / non-médicaments à exclure du graphe pharmacologique
# (même logique que COSMETIC_X : ne peuvent pas avoir d'interaction
# médicamenteuse au sens du thésaurus ANSM)
DISPOSITIFS_NON_MEDICAMENTEUX = [
    "COMPRESSES", "MEDISET", "DACRYOSERUM", "CELLUVISC", "HYLO",
]

def est_dispositif(nom):
    nom_upper = str(nom).upper()
    return any(mot in nom_upper for mot in DISPOSITIFS_NON_MEDICAMENTEUX)

masque_dispositif = mapping["medicament_nom"].apply(est_dispositif) & mapping["code_atc"].isna()
mapping.loc[masque_dispositif, "code_atc"] = "DISPOSITIF_NON_MEDICAMENTEUX"
mapping.loc[masque_dispositif, "methode_matching"] = "exclu_dispositif"

# Artefacts de qualité de données : marqués explicitement, jamais devinés
masque_restant = mapping["code_atc"].isna()
mapping.loc[masque_restant, "code_atc"] = "ATC_INCONNU"
mapping.loc[masque_restant, "methode_matching"] = "non_resolu_a_verifier"

mapping.to_csv("mapping_atc_final.csv", index=False)

print("=== BILAN FINAL DU MAPPING ATC ===")
print(mapping["methode_matching"].value_counts().to_string())

n_exploitable = (~mapping["code_atc"].isin(["ATC_INCONNU", "DISPOSITIF_NON_MEDICAMENTEUX"])).sum()
print(f"\nCode ATC exploitable : {n_exploitable}/{len(mapping)} ({n_exploitable/len(mapping):.0%})")
print(f"Dispositifs exclus   : {(mapping['code_atc']=='DISPOSITIF_NON_MEDICAMENTEUX').sum()}")
print(f"À vérifier manuellement : {(mapping['code_atc']=='ATC_INCONNU').sum()}")

print("\nRestent à vérifier manuellement (ni médicament clair ni dispositif) :")
print(mapping[mapping["methode_matching"]=="non_resolu_a_verifier"]["medicament_nom"].to_string())
