"""
Normalisation ATC des 179 médicaments réels de la base locale via WHO ATC (atc_who.csv).
"""

import re
import unicodedata
import difflib
import pandas as pd

CSV_DIR = "csv_export"  # Ou "." si medication.csv est au même niveau


def normaliser(texte):
    """Minuscule, sans accents, ponctuation simplifiée."""
    if not isinstance(texte, str):
        return ""
    texte = unicodedata.normalize("NFKD", texte).encode("ascii", "ignore").decode()
    texte = texte.lower()
    texte = re.sub(r"[^a-z0-9\s]", " ", texte)
    texte = re.sub(r"\s+", " ", texte).strip()
    return texte


# 1. Référentiel WHO ATC
atc = pd.read_csv("atc_who.csv")
atc = atc.dropna(subset=["atc5_code", "atc5_description"]).copy()
atc["nom_norm"] = atc["atc5_description"].apply(normaliser)
atc_dict = dict(zip(atc["nom_norm"], atc["atc5_code"]))
noms_atc = list(atc_dict.keys())
print(f"✅ Référentiel WHO ATC chargé : {len(atc_dict)} substances distinctes")

# 2. Chargement de l'ensemble de la table medication (179 lignes)
meds = pd.read_csv(f"{CSV_DIR}/medication.csv")
print(f"✅ {len(meds)} médicaments à traiter dans le fichier medication.csv")

# 3. Utilisation directe de code_atc s'il existe déjà dans la base
meds["nom_norm"] = meds["name"].apply(normaliser)
meds["substance_norm"] = meds["denomination_substance"].apply(normaliser)


def trouver_atc(row):
    # Priorité 0: Le code ATC est déjà directement renseigné dans medication.csv
    if pd.notna(row.get("code_atc")) and str(row.get("code_atc")).strip() != "":
        return row["code_atc"], "code_base_direct"

    # 1. Correspondance exacte sur la substance
    if row["substance_norm"] and row["substance_norm"] in atc_dict:
        return atc_dict[row["substance_norm"]], "substance_exacte"

    # 2. Recherche par sous-chaîne dans le nom
    if row["nom_norm"]:
        for nom_substance in noms_atc:
            if len(nom_substance) > 4 and nom_substance in row["nom_norm"]:
                return atc_dict[nom_substance], "sous_chaine_nom"

    # 3. Matching approché (fuzzy)
    if row["substance_norm"]:
        proches = difflib.get_close_matches(row["substance_norm"], noms_atc, n=1, cutoff=0.85)
        if proches:
            return atc_dict[proches[0]], "fuzzy_substance"

    return None, "non_trouve"


resultats = meds.apply(trouver_atc, axis=1)
meds["code_atc"] = resultats.apply(lambda r: r[0])
meds["methode_matching"] = resultats.apply(lambda r: r[1])

# 4. Export final des 179 médicaments
print("\n=== BILAN DU MATCHING ===")
print(meds["methode_matching"].value_counts().to_string())
taux = (meds["code_atc"].notna()).mean()
print(f"\nTaux de couverture ATC : {taux:.0%}")

out = meds[["id", "name", "dosage", "denomination_substance", "code_atc", "methode_matching"]]
out = out.rename(columns={"id": "medicament_id", "name": "medicament_nom"})
out.to_csv("mapping_atc_final.csv", index=False)
print(f"\n💾 mapping_atc_final.csv généré ({len(out)} lignes)")