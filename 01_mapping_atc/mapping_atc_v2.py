"""
Deuxième passe de matching ATC, sur les médicaments non résolus par
mapping_atc.py. Corrige deux limites identifiées :

  1. denomination_substance contient parfois plusieurs formes séparées par
     '|' (DCI de base + sel), ex. "ACIDE ACÉTYLSALICYLIQUE |
     ACÉTYLSALICYLATE DE DL-LYSINE" -- on découpe et on teste chaque partie.
  2. Les DCI françaises ne suivent pas le même ordre de mots que les noms
     anglais du référentiel OMS ("acide acétylsalicylique" vs
     "acetylsalicylic acid") -- on compare par similarité de tokens
     significatifs (en ignorant les mots de sel/forme : chlorhydrate,
     sodique, dihydrate...) plutôt que par la phrase entière.
"""

import re
import unicodedata
import difflib
import pandas as pd

MOTS_SEL = {
    "chlorhydrate", "sulfate", "phosphate", "besilate", "besylate", "acetate",
    "citrate", "tartrate", "maleate", "fumarate", "mesylate", "succinate",
    "propanediol", "hydrate", "dihydrate", "monohydrate", "trihydrate",
    "hemihydrate", "sodique", "sodium", "potassique", "potassium",
    "calcique", "calcium", "magnesique", "magnesium", "sel", "de", "du",
    "des", "d", "la", "le", "les", "dl", "l", "et",
    "acide", "acid",  # mots trop génériques : provoquaient un faux match
    # (ex. "ACIDE FOLIQUE" apparié à "acetylsalicylic ACID" via ce seul mot)
}


def normaliser(texte):
    if not isinstance(texte, str):
        return ""
    texte = unicodedata.normalize("NFKD", texte).encode("ascii", "ignore").decode()
    texte = texte.lower()
    texte = re.sub(r"[^a-z0-9\s]", " ", texte)
    return re.sub(r"\s+", " ", texte).strip()


def tokens_significatifs(texte_norm, longueur_min=4):
    return [t for t in texte_norm.split() if t not in MOTS_SEL and len(t) >= longueur_min]


# ---------------------------------------------------------------------------
# 1. Référentiel WHO ATC, indexé par token significatif
# ---------------------------------------------------------------------------
atc = pd.read_csv("atc_who.csv").dropna(subset=["atc5_code", "atc5_description"]).copy()
atc["nom_norm"] = atc["atc5_description"].apply(normaliser)
atc["tokens"] = atc["nom_norm"].apply(tokens_significatifs)

# index : token -> liste de (code_atc, tokens_complets_who) pour recherche rapide
index_tokens = {}
for _, r in atc.iterrows():
    for tok in r["tokens"]:
        index_tokens.setdefault(tok, []).append((r["atc5_code"], r["tokens"]))
tous_les_tokens_who = list(index_tokens.keys())
print(f"✅ Index OMS construit : {len(tous_les_tokens_who)} tokens distincts")


def meilleur_match_par_tokens(tokens_requete, seuil=0.82):
    """Cherche le meilleur code ATC en comparant chaque token de la requête
    aux tokens OMS (via une recherche approchée), et retient le candidat
    dont le token OMS a la meilleure similarité globale."""
    meilleur_score, meilleur_code = 0.0, None
    for tok in tokens_requete:
        proches = difflib.get_close_matches(tok, tous_les_tokens_who, n=3, cutoff=seuil)
        for tok_who in proches:
            score = difflib.SequenceMatcher(None, tok, tok_who).ratio()
            if score > meilleur_score:
                meilleur_score = score
                # Prend le premier code associé à ce token OMS
                meilleur_code = index_tokens[tok_who][0][0]
    return meilleur_code


# ---------------------------------------------------------------------------
# 2. Reprise des médicaments non résolus par la première passe
# ---------------------------------------------------------------------------
mapping = pd.read_csv("mapping_atc_final.csv")
a_revoir = mapping[mapping["code_atc"].isna()].copy()
print(f"\n🔍 {len(a_revoir)} médicaments à retraiter avec la méthode par tokens")

resolus = 0
for idx, row in a_revoir.iterrows():
    substance = row["denomination_substance"]
    candidats_texte = []
    if isinstance(substance, str):
        # Découpe sur '|' : chaque partie est une forme (DCI de base ou sel)
        candidats_texte.extend([p.strip() for p in substance.split("|")])
    candidats_texte.append(row["medicament_nom"])  # dernier recours : le nom commercial

    code_trouve = None
    for texte in candidats_texte:
        tokens = tokens_significatifs(normaliser(texte))
        if not tokens:
            continue
        code_trouve = meilleur_match_par_tokens(tokens)
        if code_trouve:
            break

    if code_trouve:
        mapping.loc[idx, "code_atc"] = code_trouve
        mapping.loc[idx, "methode_matching"] = "tokens_fuzzy"
        resolus += 1

print(f"✅ {resolus} médicaments supplémentaires résolus par similarité de tokens")

# ---------------------------------------------------------------------------
# 3. Bilan final et export
# ---------------------------------------------------------------------------
print("\n=== BILAN FINAL ===")
print(mapping["methode_matching"].value_counts().to_string())
taux = mapping["code_atc"].notna().mean()
print(f"\nTaux de couverture ATC final : {taux:.0%} ({mapping['code_atc'].notna().sum()}/{len(mapping)})")

mapping.to_csv("mapping_atc_final.csv", index=False)
print("\n💾 mapping_atc_final.csv mis à jour")

print("\n=== Médicaments encore non résolus (revue manuelle nécessaire) ===")
print(mapping[mapping["code_atc"].isna()]["medicament_nom"].head(20).to_string())
