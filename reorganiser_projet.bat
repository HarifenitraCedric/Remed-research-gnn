@echo off
REM ============================================================
REM  Réorganisation du projet remed-research-gnn
REM  À lancer depuis C:\RMD\remed-research-gnn
REM  Ne supprime rien : déplace uniquement (move), rien n'est perdu.
REM ============================================================

echo Création de la structure de dossiers...
mkdir 00_donnees_sources
mkdir 01_mapping_atc
mkdir 02_interactions_ansm
mkdir 03_pipeline_symbolique
mkdir 04_gnn
mkdir 05_api_deploiement
mkdir 99_a_verifier
mkdir archives_obsoletes
mkdir outils_externes

echo.
echo === 00_donnees_sources : donnees brutes ===
move csv_export 00_donnees_sources\
move export_remed_complet.csv 00_donnees_sources\
move atc_who.csv 00_donnees_sources\
move thesaurus_ansm 00_donnees_sources\

echo.
echo === 01_mapping_atc : normalisation ATC (Semaine 1) ===
move mapping_atc.py 01_mapping_atc\
move mapping_atc_v2.py 01_mapping_atc\
move finaliser_mapping_atc.py 01_mapping_atc\
move fusionner_export_atc.py 01_mapping_atc\
move mapping_atc_final.csv 01_mapping_atc\
move export_remed_avec_atc.csv 01_mapping_atc\

echo.
echo === 02_interactions_ansm : base de connaissances (Semaine 2) ===
move construire_interactions_ansm.py 02_interactions_ansm\
move interactions_ansm.csv 02_interactions_ansm\

echo.
echo === 03_pipeline_symbolique : Marches 1 et 2 ===
move regles_medicales.py 03_pipeline_symbolique\
move run_pipeline_reel.py 03_pipeline_symbolique\
move simuler_scoring.py 03_pipeline_symbolique\
move data_prete_pour_graphe_reel.csv 03_pipeline_symbolique\
move nodes_reel.csv 03_pipeline_symbolique\
move edges_reel.csv 03_pipeline_symbolique\

echo.
echo === 04_gnn : entrainement du modele (verifie) ===
move modele_heterognn.py 04_gnn\
move train_heterognn_v2.py 04_gnn\
move graphe_heterogene_complet.pt 04_gnn\
move remed_gnn_weights.pt 04_gnn\
move mappings_remed.json 04_gnn\

echo.
echo === 05_api_deploiement : microservice ===
move app_api.py 05_api_deploiement\

echo.
echo === 99_a_verifier : JAMAIS AUDITES -- a envoyer pour relecture ===
move créer_interactions_graphe.py 99_a_verifier\
move interactions_thesaurus_global.csv 99_a_verifier\
move assembler_graphe.py 99_a_verifier\
move preparer_aretes_gnn.py 99_a_verifier\
move med_interactions_edge_index.npy 99_a_verifier\
move med_interactions_edge_attr.npy 99_a_verifier\
move train_heterognn_v3.py 99_a_verifier\
move predict_service.py 99_a_verifier\
move verifier_csv.py 99_a_verifier\

echo.
echo === archives_obsoletes : versions perimees, gardees par securite ===
move train_heterognn.py archives_obsoletes\

echo.
echo === outils_externes : clones non utilises dans le pipeline final ===
move parseur-thesaurus-interactions-ansm outils_externes\

echo.
echo === Nettoyage des fichiers vides accidentels (verifiez avant si besoin) ===
REM Ces 3 fichiers de 0 octet ressemblent a des accidents de terminal
REM (ex: "commande > A" au lieu du nom de fichier voulu). Deplaces, pas supprimes.
mkdir a_supprimer_probablement
if exist A move A a_supprimer_probablement\
if exist B move B a_supprimer_probablement\
if exist Index move Index a_supprimer_probablement\

echo.
echo ============================================================
echo  Reorganisation terminee. Verifiez 99_a_verifier et
echo  a_supprimer_probablement avant de continuer a travailler.
echo ============================================================
pause
