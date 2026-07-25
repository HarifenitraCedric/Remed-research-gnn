"""
Normalisation ATC des 197 médicaments réels, via le référentiel WHO ATC/DDD
(atc_who.csv), faute de fichier de correspondance CIP->ATC directement
distribué par la BDPM (les fichiers officiels BDPM ne contiennent que le CIS,
le CIP, la composition et les avis HAS -- pas de code ATC).

Stratégie de matching, par ordre de priorité (tracée dans 'methode_matching') :
  1. Correspondance exacte sur denomination_substance (DCI) normalisée
  2. Recherche du nom de substance ATC comme sous-chaîne du nom de
     médicament normalisé (capte les génériques nommés par DCI, ex.
     "PARACETAMOL EG 500 mg")
  3. Correspondance approchée (fuzzy) sur denomination_substance
  4. Non trouvé

Limite assumée et à documenter dans le mémoire (§10) : pour les spécialités
sous nom de marque sans denomination_substance renseignée (ex. TOLEXINE),
le matching peut échouer si le nom de marque ne contient pas la DCI.
"""

import re
import unicodedata
import difflib
import pandas as pd

CSV_DIR = "csv_export"


def normaliser(texte):
    """Minuscule, sans accents, ponctuation simplifiée."""
    if not isinstance(texte, str):
        return ""
    texte = unicodedata.normalize("NFKD", texte).encode("ascii", "ignore").decode()
    texte = texte.lower()
    texte = re.sub(r"[^a-z0-9\s]", " ", texte)
    texte = re.sub(r"\s+", " ", texte).strip()
    return texte


# ---------------------------------------------------------------------------
# 1. Référentiel WHO ATC (niveau 5 = code final à 7 caractères)
# ---------------------------------------------------------------------------
atc = pd.read_csv("atc_who.csv")
atc = atc.dropna(subset=["atc5_code", "atc5_description"]).copy()
atc["nom_norm"] = atc["atc5_description"].apply(normaliser)
# Dictionnaire substance normalisée -> code ATC (garde le premier si doublons)
atc_dict = dict(zip(atc["nom_norm"], atc["atc5_code"]))
noms_atc = list(atc_dict.keys())
print(f"✅ Référentiel WHO ATC chargé : {len(atc_dict)} substances distinctes")

# ---------------------------------------------------------------------------
# 2. Médicaments réellement utilisés dans export_remed_complet.csv
# ---------------------------------------------------------------------------
export = pd.read_csv("export_remed_complet.csv")
medicament_ids = export["medicament_id"].unique()

medication = pd.read_csv(f"{CSV_DIR}/medication.csv")
meds = medication[medication["id"].isin(medicament_ids)].copy()
print(f"✅ {len(meds)} médicaments distincts à traiter")

meds["nom_norm"] = meds["name"].apply(normaliser)
meds["substance_norm"] = meds["denomination_substance"].apply(normaliser)


def trouver_atc(row):
    # 1. Correspondance exacte sur la substance déclarée
    if row["substance_norm"] and row["substance_norm"] in atc_dict:
        return atc_dict[row["substance_norm"]], "substance_exacte"

    # 2. Le nom de la substance ATC apparaît tel quel dans le nom du médicament
    #    (capte les génériques nommés par DCI : "PARACETAMOL EG 500 mg")
    if row["nom_norm"]:
        for nom_substance in noms_atc:
            if len(nom_substance) > 4 and nom_substance in row["nom_norm"]:
                return atc_dict[nom_substance], "sous_chaine_nom"

    # 3. Correspondance approchée sur la substance déclarée (fautes, variantes)
    if row["substance_norm"]:
        proches = difflib.get_close_matches(row["substance_norm"], noms_atc, n=1, cutoff=0.85)
        if proches:
            return atc_dict[proches[0]], "fuzzy_substance"

    return None, "non_trouve"


resultats = meds.apply(trouver_atc, axis=1)
meds["code_atc"] = resultats.apply(lambda r: r[0])
meds["methode_matching"] = resultats.apply(lambda r: r[1])

# ---------------------------------------------------------------------------
# 3. Bilan et export
# ---------------------------------------------------------------------------
print("\n=== BILAN DU MATCHING ===")
print(meds["methode_matching"].value_counts().to_string())
taux = (meds["code_atc"].notna()).mean()
print(f"\nTaux de couverture ATC : {taux:.0%}")

out = meds[["id", "name", "dosage", "denomination_substance", "code_atc", "methode_matching"]]
out = out.rename(columns={"id": "medicament_id", "name": "medicament_nom"})
out.to_csv("mapping_atc_final.csv", index=False)
print(f"\n💾 mapping_atc_final.csv généré ({len(out)} lignes)")

print("\n=== ÉCHANTILLON DE MÉDICAMENTS NON TROUVÉS (à revoir manuellement) ===")
print(out[out["code_atc"].isna()][["medicament_nom", "denomination_substance"]].head(15).to_string())
