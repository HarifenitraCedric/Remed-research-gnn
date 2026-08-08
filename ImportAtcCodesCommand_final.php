<?php

namespace App\Command;

use App\Entity\Medication;
use App\Repository\MedicationRepository;
use Doctrine\ORM\EntityManagerInterface;
use Symfony\Component\Console\Attribute\AsCommand;
use Symfony\Component\Console\Command\Command;
use Symfony\Component\Console\Input\InputArgument;
use Symfony\Component\Console\Input\InputInterface;
use Symfony\Component\Console\Output\OutputInterface;
use Symfony\Component\Console\Style\SymfonyStyle;

/**
 * Importe mapping_atc_final.csv dans la table medication.
 *
 * ⚠️ AUCUN FALLBACK. Un médicament non apparié DOIT rester NULL.
 *
 * Garde-fou intégré : après l'import, la commande vérifie elle-même
 * qu'aucun code ATC ne représente une part anormalement élevée des
 * médicaments (seuil : 5%). Si c'est le cas, elle échoue explicitement au
 * lieu d'annoncer un succès -- ce contrôle est apparu nécessaire après
 * plusieurs réintroductions accidentelles de fallback (N02BE01, puis
 * V03AX) par des versions antérieures de cette commande.
 */
#[AsCommand(
    name: 'app:import-atc-codes',
    description: 'Importe les codes ATC (sans fallback, avec auto-vérification anti-régression)'
)]
class ImportAtcCodesCommand extends Command
{
    private const SEUIL_ALERTE_DOMINANCE = 0.05; // 5% d'un seul code = suspect

    public function __construct(
        private readonly MedicationRepository $medicationRepository,
        private readonly EntityManagerInterface $em,
    ) {
        parent::__construct();
    }

    protected function configure(): void
    {
        $this->addArgument('csv_path', InputArgument::REQUIRED, 'Chemin vers mapping_atc_final.csv');
    }

    protected function execute(InputInterface $input, OutputInterface $output): int
    {
        $io = new SymfonyStyle($input, $output);
        $csvPath = $input->getArgument('csv_path');

        if (!file_exists($csvPath)) {
            $io->error("Fichier introuvable : {$csvPath}");
            return Command::FAILURE;
        }

        // --- Remise à zéro systématique avant import, pour ne jamais hériter
        // d'une contamination précédente (fallback d'une exécution antérieure)
        $this->em->createQuery('UPDATE App\Entity\Medication m SET m.codeAtc = NULL')->execute();
        $io->note('Colonne code_atc réinitialisée avant import.');

        $handle = fopen($csvPath, 'r');
        $headers = fgetcsv($handle);
        $idxId        = array_search('medicament_id', $headers, true);
        $idxNom       = array_search('medicament_nom', $headers, true);
        $idxSubstance = array_search('denomination_substance', $headers, true);
        $idxAtc       = array_search('code_atc', $headers, true);

        if ($idxAtc === false) {
            $io->error("Colonne 'code_atc' introuvable dans le CSV.");
            fclose($handle);
            return Command::FAILURE;
        }

        $codesExclus = ['ATC_INCONNU', 'DISPOSITIF_NON_MEDICAMENTEUX'];
        $compteurCodes = [];
        $miseAJourId = $miseAJourSubstance = $miseAJourNom = $ignores = $nonTrouves = 0;

        while (($row = fgetcsv($handle)) !== false) {
            if (!isset($row[$idxAtc])) continue;
            $codeAtc = trim($row[$idxAtc]);

            if (in_array($codeAtc, $codesExclus, true) || $codeAtc === '') {
                $ignores++;
                continue;
            }

            $medication = null;
            $methode = null;

            if ($idxId !== false && isset($row[$idxId]) && is_numeric($row[$idxId])) {
                $medication = $this->medicationRepository->find((int) $row[$idxId]);
                if ($medication) { $methode = 'id'; }
            }
            if (!$medication && $idxSubstance !== false && !empty($row[$idxSubstance])) {
                $substance = trim($row[$idxSubstance]);
                $medication = $this->medicationRepository->findOneBy(['denominationSubstance' => $substance])
                    ?? $this->medicationRepository->createQueryBuilder('m')
                        ->where('m.denominationSubstance LIKE :val')->setParameter('val', '%' . $substance . '%')
                        ->setMaxResults(1)->getQuery()->getOneOrNullResult();
                if ($medication) { $methode = 'substance'; }
            }
            if (!$medication && $idxNom !== false && !empty($row[$idxNom])) {
                $nomOriginal = trim($row[$idxNom]);
                $nomEpure = trim(explode('(', $nomOriginal)[0]);
                $premierMot = preg_split('/\s+/', $nomEpure)[0] ?? '';
                $medication = $this->medicationRepository->findOneBy(['name' => $nomOriginal])
                    ?? $this->medicationRepository->findOneBy(['name' => $nomEpure]);
                if (!$medication && strlen($premierMot) >= 3) {
                    $medication = $this->medicationRepository->createQueryBuilder('m')
                        ->where('m.name LIKE :val')->setParameter('val', '%' . $premierMot . '%')
                        ->setMaxResults(1)->getQuery()->getOneOrNullResult();
                }
                if ($medication) { $methode = 'nom'; }
            }

            if ($medication) {
                $medication->setCodeAtc($codeAtc);
                $compteurCodes[$codeAtc] = ($compteurCodes[$codeAtc] ?? 0) + 1;
                match ($methode) {
                    'id' => $miseAJourId++,
                    'substance' => $miseAJourSubstance++,
                    'nom' => $miseAJourNom++,
                };
            } else {
                $nonTrouves++;
            }
        }
        fclose($handle);
        $this->em->flush();

        $totalMedications = $this->medicationRepository->count([]);
        $totalMappes = $miseAJourId + $miseAJourSubstance + $miseAJourNom;

        $io->section('Résultat de l\'import');
        $io->table(['Métrique', 'Valeur'], [
            ['Mappés par ID', $miseAJourId],
            ['Mappés par substance', $miseAJourSubstance],
            ['Mappés par nom', $miseAJourNom],
            ['Total mappés', $totalMappes],
            ['Ignorés (ATC_INCONNU/dispositif)', $ignores],
            ['Non trouvés en base', $nonTrouves],
            ['Total médicaments en base', $totalMedications],
            ['Couverture', sprintf('%.0f%%', $totalMedications > 0 ? $totalMappes / $totalMedications * 100 : 0)],
        ]);

        // --- GARDE-FOU ANTI-RÉGRESSION : détection automatique de fallback ---
        $io->section('Vérification anti-régression (détection de fallback)');
        arsort($compteurCodes);
        $codeDominant = array_key_first($compteurCodes);
        $nbCodeDominant = $compteurCodes[$codeDominant] ?? 0;
        $partCodeDominant = $totalMappes > 0 ? $nbCodeDominant / $totalMappes : 0;

        $io->writeln(sprintf(
            "Code le plus fréquent : %s (%d occurrences, %.0f%% des médicaments mappés)",
            $codeDominant, $nbCodeDominant, $partCodeDominant * 100
        ));

        if ($partCodeDominant > self::SEUIL_ALERTE_DOMINANCE) {
            $io->error(sprintf(
                "ÉCHEC VOLONTAIRE : le code '%s' représente %.0f%% des médicaments mappés (seuil : %.0f%%).\n" .
                "C'est la signature typique d'un fallback générique réintroduit quelque part.\n" .
                "AUCUNE donnée n'est considérée fiable tant que ce point n'est pas résolu.\n" .
                "Vérifiez qu'aucune autre version de cette commande, ni aucun trigger SQL, n'assigne " .
                "de valeur par défaut à code_atc.",
                $codeDominant, $partCodeDominant * 100, self::SEUIL_ALERTE_DOMINANCE * 100
            ));
            return Command::FAILURE;
        }

        $io->success(sprintf(
            "Import terminé et vérifié : %d/%d médicaments mappés (%.0f%%), aucun fallback détecté.\n" .
            "Les %d médicaments non mappés ont code_atc = NULL (volontairement).",
            $totalMappes, $totalMedications, $totalMedications > 0 ? $totalMappes / $totalMedications * 100 : 0,
            $totalMedications - $totalMappes
        ));

        return Command::SUCCESS;
    }
}
