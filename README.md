# REMED-RESEARCH — Organisation du projet

## Ordre d'exécution du pipeline complet

À relancer dans cet ordre exact si une donnée source change (voir dépendances) :

```
01_mapping_atc/mapping_atc.py
01_mapping_atc/mapping_atc_v2.py
01_mapping_atc/finaliser_mapping_atc.py
        ↓
01_mapping_atc/fusionner_export_atc.py        ⚠️ NE JAMAIS OUBLIER après un changement de mapping
        ↓
02_interactions_ansm/construire_interactions_ansm.py
        ↓
03_pipeline_symbolique/run_pipeline_reel.py
        ↓
03_pipeline_symbolique/simuler_scoring.py     (validation / résultats Marches 1 & 2)
```

Le pipeline GNN (dossier `04_gnn/`) et le microservice (`05_api_deploiement/`) sont
**en attente de ré-audit** : ils dépendent de fichiers actuellement dans
`99_a_verifier/` dont le rôle exact et la fiabilité n'ont pas encore été
confirmés (risque de contamination par le même bug de mapping ATC déjà
corrigé dans la chaîne symbolique).

## Rôle de chaque dossier

| Dossier | Contenu | Statut |
|---|---|---|
| `00_donnees_sources/` | Données brutes, jamais modifiées à la main | ✅ Stable |
| `01_mapping_atc/` | Normalisation médicament → code ATC | ✅ Vérifié |
| `02_interactions_ansm/` | Table de gravité des interactions (ATC↔ATC) | ✅ Vérifié |
| `03_pipeline_symbolique/` | Marches 1 & 2 : reconstruction + raisonnement | ✅ Vérifié |
| `04_gnn/` | Entraînement du modèle de prédiction de liens | ✅ Vérifié (train_heterognn_v2.py uniquement) |
| `05_api_deploiement/` | Microservice FastAPI d'inférence | ✅ Vérifié |
| `99_a_verifier/` | Fichiers non audités, à relire avant usage | ⚠️ À traiter |
| `archives_obsoletes/` | Anciennes versions bugguées, conservées par sécurité | ❌ Ne pas utiliser |
| `outils_externes/` | Dépôts clonés non retenus dans la chaîne finale | ➖ Référence seulement |

## Règle d'or

Si vous modifiez un fichier en amont de la chaîne (ex. `mapping_atc_final.csv`),
**tous les fichiers en aval doivent être régénérés**, pas seulement celui que
vous venez de toucher. Un fichier de sortie (`export_remed_avec_atc.csv`,
`interactions_ansm.csv`, `nodes_reel.csv`...) ne se met jamais à jour tout
seul.
