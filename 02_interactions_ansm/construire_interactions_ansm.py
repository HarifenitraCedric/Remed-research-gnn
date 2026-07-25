"""
Construction de interactions_ansm.csv : table générale d'interactions
médicamenteuses (atc_1, atc_2, gravite, description), à partir du Thésaurus
ANSM structuré (version août 2016, dernière disponible sous forme de paires
déjà extraites -- limite à documenter : une version plus récente (2024)
existe en PDF mais nécessiterait un nouveau parsing).

Étapes :
  1. Charger les 1463 paires brutes (protagoniste1, protagoniste2, gravité)
  2. "protagoniste2" est souvent une CLASSE thérapeutique (ex: "ANTI-TNF
     ALPHA") plutôt qu'une molécule isolée -> déplié via la table
     molécule<->famille du même corpus
  3. Chaque nom de molécule (français) est mis en correspondance avec un
     code ATC via le même moteur de correspondance par tokens que
     mapping_atc_v2.py (réutilisé ici pour cohérence)
  4. Export en (atc_1, atc_2, gravite, description) -- prêt à être chargé
     par regles_medicales.py à la place du dictionnaire à 1 entrée
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
# 1. Référentiel WHO ATC, indexé par token (identique à mapping_atc_v2.py)
# ---------------------------------------------------------------------------
atc = pd.read_csv("atc_who.csv").dropna(subset=["atc5_code", "atc5_description"]).copy()
atc["nom_norm"] = atc["atc5_description"].apply(normaliser)
atc["tokens"] = atc["nom_norm"].apply(tokens_significatifs)

index_tokens = {}
for _, r in atc.iterrows():
    for tok in r["tokens"]:
        index_tokens.setdefault(tok, []).append(r["atc5_code"])
tous_les_tokens_who = list(index_tokens.keys())

# Cache pour éviter de refaire le même matching plusieurs fois (les mêmes
# molécules reviennent dans de nombreuses paires du thésaurus)
cache_matching = {}


def molecule_vers_atc(nom_molecule, seuil=0.82):
    if nom_molecule in cache_matching:
        return cache_matching[nom_molecule]
    tokens = tokens_significatifs(normaliser(nom_molecule))
    meilleur_score, meilleur_code = 0.0, None
    for tok in tokens:
        proches = difflib.get_close_matches(tok, tous_les_tokens_who, n=3, cutoff=seuil)
        for tok_who in proches:
            score = difflib.SequenceMatcher(None, tok, tok_who).ratio()
            if score > meilleur_score:
                meilleur_score, meilleur_code = score, index_tokens[tok_who][0]
    cache_matching[nom_molecule] = meilleur_code
    return meilleur_code


# ---------------------------------------------------------------------------
# 2. Chargement du thésaurus + table molécule<->famille (pour déplier les
#    classes thérapeutiques en molécules individuelles)
# ---------------------------------------------------------------------------
thesaurus = pd.read_csv("thesaurus_ansm/IMthesaurusANSM-master/CSVfiles/thesaurusAout2016.csv", sep=None, engine="python")
mol_famille = pd.read_csv("thesaurus_ansm/IMthesaurusANSM-master/CSVfiles/moleculesfamillesAout2016.csv", sep=None, engine="python")
print(f"✅ {len(thesaurus)} paires brutes, {len(mol_famille)} associations molécule/famille chargées")

famille_vers_molecules = mol_famille.groupby(mol_famille.columns[1])[mol_famille.columns[0]].apply(list).to_dict()
molecules_connues = set(mol_famille[mol_famille.columns[0]].str.lower())


def deplier_protagoniste(nom):
    """Si 'nom' est une famille thérapeutique connue, retourne la liste des
    molécules qui la composent. Sinon, retourne [nom] tel quel (déjà une
    molécule individuelle)."""
    nom_lower = str(nom).strip().lower()
    for famille, molecules in famille_vers_molecules.items():
        if isinstance(famille, str) and famille.lower() == nom_lower:
            return molecules
    return [nom]


def gravite_ligne(row):
    if pd.notna(row.get("CI")):
        return "contre_indication"
    if pd.notna(row.get("AD")):
        return "association_deconseillee"
    if pd.notna(row.get("PE")):
        return "precaution_emploi"
    return "a_prendre_en_compte"


# ---------------------------------------------------------------------------
# 3. Construction des paires molécule-molécule, puis matching vers ATC
# ---------------------------------------------------------------------------
paires_finales = []
for _, row in thesaurus.iterrows():
    mols_1 = deplier_protagoniste(row["protagoniste1"])
    mols_2 = deplier_protagoniste(row["protagoniste2"])
    gravite = gravite_ligne(row)
    desc = str(row.get("description_interaction", ""))[:300]

    for m1 in mols_1:
        for m2 in mols_2:
            paires_finales.append({"molecule_1": m1, "molecule_2": m2,
                                    "gravite": gravite, "description": desc})

paires_df = pd.DataFrame(paires_finales)
print(f"✅ {len(paires_df)} paires molécule-molécule après dépliage des classes")

print("⏳ Correspondance vers les codes ATC (peut prendre 1-2 minutes)...")
paires_df["atc_1"] = paires_df["molecule_1"].apply(molecule_vers_atc)
paires_df["atc_2"] = paires_df["molecule_2"].apply(molecule_vers_atc)

resolues = paires_df.dropna(subset=["atc_1", "atc_2"]).copy()
resolues = resolues[resolues["atc_1"] != resolues["atc_2"]]
print(f"✅ {len(resolues)}/{len(paires_df)} paires résolues en codes ATC des deux côtés "
      f"({len(resolues)/len(paires_df):.0%})")

# Déduplication (une paire non ordonnée = une seule ligne, garde la plus grave)
gravite_ordre = {"contre_indication": 0, "association_deconseillee": 1,
                  "precaution_emploi": 2, "a_prendre_en_compte": 3}
resolues["ordre_gravite"] = resolues["gravite"].map(gravite_ordre)
resolues["paire_triee"] = resolues.apply(lambda r: tuple(sorted([r["atc_1"], r["atc_2"]])), axis=1)
resolues = resolues.sort_values("ordre_gravite").drop_duplicates(subset="paire_triee", keep="first")

out = resolues[["atc_1", "atc_2", "gravite", "description"]].copy()
out[["atc_1", "atc_2"]] = pd.DataFrame(resolues["paire_triee"].tolist(), index=resolues.index)
out.to_csv("interactions_ansm.csv", index=False)

print(f"\n💾 interactions_ansm.csv : {len(out)} paires uniques ATC-ATC")
print(out["gravite"].value_counts().to_string())
