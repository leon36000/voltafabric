# Contrat de preuve

## Niveaux à ne jamais fusionner

1. SOURCE_REVIEWED : code précis consulté, version et licence identifiées.
2. CPU_TESTED : exécution CPU réelle, logs, commande, code et environnement identifiés.
3. ANALYTICAL : calcul ou simulation avec hypothèses explicites.
4. CUDA_COMPILED : binaire SM70 produit; ne prouve pas son exécution.
5. V100_EXECUTED : calcul réel sur une V100 identifiée; oracle et tolérances publiés.
6. FABRIC_MEASURED : deux directions, tailles et échanges simultanés mesurés.
7. SERVICE_QUALIFIED : modèle complet, baseline identique, qualité et énergie mesurées.

Le pack livré est CPU_TESTED et ANALYTICAL. CUDA, PCIe, RDMA et service restent ouverts.
Un retour de modèle est UNTRUSTED_MODEL_PROPOSAL même s'il affirme être correct.
L'accord de plusieurs modèles ne remplace pas une expérience indépendante.

## Reproductibilité

Pour chaque lot, conserver : scope, hash source, toolchain, commande exacte, code de
sortie, logs, données/seed, tolérances et statut des tests ignorés. Les preuves sont
liées au code testé; toute modification fonctionnelle exige une nouvelle exécution.
Les secondes de compilation/oracle ne sont jamais des benchmarks V100.

L'oracle Python utilise `math.fsum` et n'importe pas l'implémentation Rust. Les tests
contre le DP énumèrent toutes les coupes des petits cas; ils ne prouvent pas un
optimum de placement général. Le contrôle de sensibilité corrompt délibérément une
sortie; il ne constitue pas une campagne complète de mutation-testing.

## Promotion / investissement

Baseline corrigée, mêmes poids, quantification, contexte, batch, concurrence et
réglages connus. Comparer qualité, TTFT, inter-token/p95, requêtes réussies, mémoire,
joules/tâche et coût de tout le système. Séparer charge, chauffe et régime stable.
Ne pas multiplier entre eux les gains d'articles ou tirer un gain service d'un microkernel.

Seuils stratégiques proposés : environ ×2 de débit utile, ou 30 % d'énergie économisée,
ou modèle nettement plus utile devenu exploitable. Ce ne sont ni des prévisions ni
une condition empêchant un petit correctif bon marché d'être intégré.

## Mutation plutôt que comptage de couverture

Une garde modifiée ou ajoutée exige un mutant compilable qui la retire, et un test qui
meurt avec elle. La raison est mesurée, pas théorique : sur 48 mutants bornés (A26–A33),
15 gardes avaient un test qui *ressemblait* à de la couverture mais dont l'assertion était
satisfaite par une règle voisine — la liste blanche d'extensions au lieu du filtre de
secrets, un gateway absent au lieu de la garde d'autorisation, un fichier illisible au lieu
de l'acquittement matériel, une longueur au lieu de la borne de K. Six sous-systèmes
différents ont montré la même forme. Compter les tests ne la détecte pas; `scripts/rust_mutations.py`
la détecte, avec `--control` pour qu'une copie non mutée passe d'abord, sans quoi aucun
verdict n'est valide. Un mutant équivalent pour l'entrée sondée n'est ni un défaut ni une
raison de retirer la clause; un mutant non compilable n'est pas un mutant tué; une erreur
d'environnement n'est pas un test réussi.

## Blocages

Un blocage matériel n'arrête pas les tâches CPU indépendantes. Il interdit seulement
leur promotion matérielle. Un budget/alias cloud inconnu bloque les appels payants,
pas le développement local. Un test réellement échoué reste visible avec sa cause,
sa correction et le passage rouge→vert; ne pas l'effacer pour embellir la livraison.
