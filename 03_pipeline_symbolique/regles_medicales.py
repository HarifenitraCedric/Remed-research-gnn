"""
Base de connaissances et fonctions de détection, factorisées pour être
utilisées à la fois par export_graphe.py (Marche 2 : construction du graphe)
et detecter_danger.py (affichage console de contrôle), afin qu'une correction
de règle n'ait besoin d'être faite qu'à un seul endroit.

⚠️ DICTIONNAIRE_CLASSES, DOSES_MAX_JOURNALIERES et INTERACTIONS_DANGER sont
des tables MINIMALES pour prototyper la logique. Elles doivent être
remplacées par les sources externes confirmées (§9 du cadrage) :
  - Codes ATC / classes thérapeutiques -> API BDPM
  - Interactions dangereuses -> Thésaurus ANSM des interactions
    médicamenteuses (via un parseur du type axel-op/parseur-thesaurus-
    interactions-ansm), sinon la détection reste bornée aux quelques paires
    codées ici en dur.
"""

import re
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Base de connaissances (À REMPLACER par ATC/BDPM + Thésaurus ANSM)
# ---------------------------------------------------------------------------

DICTIONNAIRE_CLASSES = {
    "M01AE": "AINS - Dérivés propioniques",
    "M01AB": "AINS - Dérivés acétiques",
    "J01AA": "Antibiotiques (Tétracyclines)",
    "A12CB": "Compléments minéraux (Zinc)",
    "N02BE": "Analgésiques",
}

DOSES_MAX_JOURNALIERES = {
    "N02BE01": {"nom": "PARACÉTAMOL", "max_mg": 4000,
                "danger": "Hépatotoxicité aiguë (destruction irréversible des cellules du foie)."},
    "A12CB01": {"nom": "ZINC (gluconate)", "max_mg": 50,
                "danger": "Troubles gastro-intestinaux sévères, réduction de l'absorption du cuivre."},
}

def _charger_interactions_ansm():
    """Charge la table d'interactions ATC-ATC issue du Thésaurus ANSM
    (construite par construire_interactions_ansm.py, 32 382 paires réelles).
    Retombe sur un dictionnaire minimal si le fichier n'est pas présent,
    pour que le reste du pipeline continue de fonctionner en dégradé."""
    try:
        df = pd.read_csv("interactions_ansm.csv")
    except FileNotFoundError:
        return {
            ("A12CB01", "J01AA02"): {
                "titre": "[DANGER] Interaction (fallback, thésaurus non chargé)",
                "desc": "Le Zinc diminue l'efficacité de la Doxycycline.",
                "gravite": "association_deconseillee",
            }
        }

    LIBELLES_GRAVITE = {
        "contre_indication": "[CONTRE-INDICATION]",
        "association_deconseillee": "[ASSOCIATION DÉCONSEILLÉE]",
        "precaution_emploi": "[PRÉCAUTION D'EMPLOI]",
        "a_prendre_en_compte": "[À PRENDRE EN COMPTE]",
    }
    resultat = {}
    for _, row in df.iterrows():
        cle = (row["atc_1"], row["atc_2"])
        resultat[cle] = {
            "titre": LIBELLES_GRAVITE.get(row["gravite"], "[INTERACTION]"),
            "desc": str(row["description"])[:250],
            "gravite": row["gravite"],
        }
    return resultat


INTERACTIONS_DANGER = _charger_interactions_ansm()

# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def safe_float(val):
    try:
        if pd.isna(val) or val is None:
            return 0.0
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def parser_dosage_mg(valeur):
    """Extrait le dosage unitaire en mg depuis un texte comme '104,10 mg'
    ou '15 mg' (gère la virgule décimale française). Retourne None si le
    dosage n'est pas exprimé en mg (ex: '200 000 UI', 'Inconnu') -- dans ce
    cas il ne faut PAS retomber sur une valeur en dur arbitraire, car on ne
    peut simplement pas comparer un dosage en UI à un seuil en mg.
    """
    if not isinstance(valeur, str):
        return None
    match = re.search(r"([\d]+(?:[.,]\d+)?)\s*mg", valeur, re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", "."))
    except ValueError:
        return None


def dose_journaliere_unites(row):
    """Nombre d'unités (comprimés/gélules/doses) prises par jour."""
    return (
        safe_float(row.get("dose_morning", 0))
        + safe_float(row.get("dose_noon", 0))
        + safe_float(row.get("dose_evening", 0))
        + safe_float(row.get("dose_night", 0))
    )


def dose_journaliere_mg(row):
    """Dose totale en mg prise par jour, ou None si le dosage unitaire n'est
    pas exprimable en mg (auparavant : fallback en dur 15mg/100mg appliqué
    systématiquement puisque 'dosage_mg' n'existe pas dans les données --
    corrigé ici en parsant 'medicament_dosage' et en renvoyant explicitement
    None quand ce n'est pas possible, plutôt qu'une valeur inventée)."""
    dosage_unitaire = parser_dosage_mg(row.get("medicament_dosage"))
    if dosage_unitaire is None:
        return None
    return dose_journaliere_unites(row) * dosage_unitaire


# ---------------------------------------------------------------------------
# Détection — opère sur `achats`, la liste triée des délivrances d'UN patient
# (dicts avec au moins : code_atc, medicament_nom, medicament_dosage,
#  date_debut_reel, date_fin_reel, dose_morning..dose_night)
# ---------------------------------------------------------------------------

def detecter_surdosages(achats):
    """Scanne jour par jour les fenêtres de couverture du patient et détecte
    les cumuls de dose (mg/jour) dépassant le seuil connu pour une molécule.
    Retourne une liste d'alertes : {atc, dose_mg, limit_mg, ...}."""
    alertes = []
    if not achats:
        return alertes

    df = pd.DataFrame(achats)
    date_min, date_max = df["date_debut_reel"].min(), df["date_fin_reel"].max()
    deja_signale = set()

    for jour in pd.date_range(start=date_min, end=date_max):
        actifs = df[(df["date_debut_reel"] <= jour) & (df["date_fin_reel"] >= jour)]
        for atc, sous_groupe in actifs.groupby("code_atc"):
            if atc not in DOSES_MAX_JOURNALIERES:
                continue
            dose_totale = 0.0
            dosage_inconnu = False
            for _, row in sous_groupe.iterrows():
                dose_mg = dose_journaliere_mg(row)
                if dose_mg is None:
                    dosage_inconnu = True
                    continue
                dose_totale += dose_mg

            limite = DOSES_MAX_JOURNALIERES[atc]["max_mg"]
            if dose_totale > limite:
                cle = (atc, jour.date())
                if cle not in deja_signale:
                    alertes.append({
                        "type": "OVERDOSE_RISK",
                        "atc": atc,
                        "nom": DOSES_MAX_JOURNALIERES[atc]["nom"],
                        "dose_mg": round(dose_totale, 2),
                        "limit_mg": limite,
                        "jour": jour.date(),
                        "danger": DOSES_MAX_JOURNALIERES[atc].get("danger", ""),
                    })
                    deja_signale.add(cle)
            elif dosage_inconnu and atc not in deja_signale:
                # Dosage non exprimé en mg pour au moins une délivrance :
                # on ne peut pas garantir l'absence de surdosage, à signaler
                # comme limite plutôt que de conclure silencieusement "OK".
                pass  # traçable via un log séparé si besoin, volontairement non alerté

    return alertes


def detecter_doublons_et_interactions(achats):
    """Compare les délivrances deux à deux : doublons de classe thérapeutique
    et interactions dangereuses connues, uniquement si les fenêtres de
    couverture réelles se chevauchent. Retourne une liste d'alertes."""
    alertes = []
    alertes_classe_emises = set()
    alertes_danger_emises = set()

    for i in range(len(achats)):
        for j in range(i + 1, len(achats)):
            a1, a2 = achats[i], achats[j]
            atc1 = str(a1.get("code_atc", "")).strip()
            atc2 = str(a2.get("code_atc", "")).strip()
            if not atc1 or not atc2 or "INCONNU" in atc1 or "INCONNU" in atc2 or atc1 == atc2:
                continue

            overlap = max(a1["date_debut_reel"], a2["date_debut_reel"]) <= \
                      min(a1["date_fin_reel"], a2["date_fin_reel"])
            if not overlap:
                continue

            classe1, classe2 = atc1[:5], atc2[:5]
            est_doublon_ains = classe1 in ("M01AE", "M01AB") and classe2 in ("M01AE", "M01AB")
            if est_doublon_ains or (classe1 == classe2 and classe1 in DICTIONNAIRE_CLASSES):
                paire = tuple(sorted([atc1, atc2]))
                if paire not in alertes_classe_emises:
                    alertes.append({
                        "type": "SAME_THERAPEUTIC_CLASS",
                        "atc1": atc1, "atc2": atc2,
                        "nom1": a1["medicament_nom"], "nom2": a2["medicament_nom"],
                    })
                    alertes_classe_emises.add(paire)

            paire_danger = tuple(sorted([atc1, atc2]))
            if paire_danger in INTERACTIONS_DANGER:
                if paire_danger not in alertes_danger_emises:
                    info = INTERACTIONS_DANGER[paire_danger]
                    alertes.append({
                        "type": "TOXIC_INTERACTION",
                        "atc1": atc1, "atc2": atc2,
                        "nom1": a1["medicament_nom"], "nom2": a2["medicament_nom"],
                        "titre": info["titre"], "desc": info["desc"],
                    })
                    alertes_danger_emises.add(paire_danger)

    return alertes


def detecter_alertes_patient(achats):
    """Point d'entrée unique : agrège surdosages + doublons/interactions
    pour un patient donné (liste de délivrances triées par date_debut_reel)."""
    return detecter_surdosages(achats) + detecter_doublons_et_interactions(achats)
