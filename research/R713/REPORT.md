# R713 — préflight atomique des six owners et chargement CPU

## Tranche livrée

R712 raccordait un owner MXFP4 résidant au LOAD Session, mais le propriétaire pouvait encore appeler cette API avec un seul owner sans preuve que le reste du top‑6 était prêt. R713 ferme cette frontière : chaque owner reçoit une empreinte SHA‑256 du routage exact, qui comprend l’identité run/modèle/plan/couche/batch/epoch/attempt, les six IDs dans l’ordre donneur, les bits IEEE‑754 exacts des six coefficients et les rank/local choisis par `routing::dispatch`.

`preflight_pinned_mxfp4_route_residents` refuse avant connexion s’il manque un expert, si un expert apparaît deux fois, si un coefficient ou binding a changé, si l’inventaire à six tensors ou l’artefact vérifié ne correspond pas, ou si le budget agrégé déclaré est dépassé. Il retourne un reçu privé à construction contrôlée en ordre de route. `load_into_session` exige ce reçu complet et vérifie que l’owner consommé y figure avec la même identité, taille et empreinte. Le rejeu automatique n’est pas ajouté.

## Preuve CPU

La fixture épinglée fournit l’index et l’en‑tête officiels DeepSeek; les octets de poids restent synthétiques. Elle crée les six owners au format MXFP4 puis choisit six rangs CPU distincts. Les refus incomplet, budget insuffisant, coefficient modifié et rank binding modifié sont exercés avant de créer une Session. Après un préflight valide, le test ouvre six processus locaux `vf-rank-session`, envoie HELLO et un LOAD par owner, vérifie les reçus liés à la même empreinte, puis ferme chaque session avec `bye_execs = 0`.

Commandes et reçus : voir `BUILD_TEST.txt`. Debug et release passent. Clippy `-D warnings`, rustfmt et vérification diff passent également.

## Réemploi et limites

Réemploi interne effectif : `routing::dispatch`, l’artefact `session::mxfp4::Artifact`, le protocole Session HELLO/LOAD, `LoadReceipt` et `close`. L’idée du préflight de toutes les bindings avant le démarrage reprend la frontière éprouvée du client `session::mxfp4::run_series`; la nouvelle empreinte lie en plus précisément la route donneuse et ses coefficients à chaque owner. Aucun code tiers n’a été copié dans R713. Le sondage arXiv R712 reste une source de pistes : Apex/DALI sont des algorithmes à mesurer localement; Korch est Apache‑2.0 mais son périmètre publié est un ordonnanceur DNN expérimental, pas un noyau V100 LLM prêt à intégrer.

Ce test prouve seulement six LOAD CPU indépendants avec des payloads synthétiques pour une route à un token. Il n’envoie aucun EXEC et ne prouve ni calcul de couche, génération, payload réel, partage de rang multi‑expert, mémoire GPU, V100, PC4↔PC6, RDMA, exactitude numérique de sortie ni performance. Le budget est une admission d’octets d’artefacts; il ne mesure pas le RSS ni la capacité mémoire d’un équipement. La réduction atomique de résultats reste à construire/qualifier avant toute exécution multi‑expert.

## Publication

Une relecture authentifiée du projet privé `root/voltafabric` a confirmé le 2 novembre 2026 que la branche R710 pointait vers `f4c73db055f63b5db014c334d082ae414efb2541`, que les branches R711, R712 et R713 étaient absentes et que les seules opérations actives étaient `[]`. R713 a ensuite été créée à partir de ce SHA, sans mise à jour de `main`, dans le commit `df056168ce7d0506bfa18cef5b379243281df4cd`. Les 16 fichiers publiés, dont le code, les rapports et le handoff, ont été relus exactement depuis cette branche.

Le pipeline R713 #2503 est `success`; il ne couvre que les scans configurés et le fallback, pas la compilation Rust ni les tests CPU locaux. Les tests/debug/release/clippy ci-dessus ont été exécutés séparément en local. Le certificat HTTPS direct PC4 reste non approuvé et n'est pas contourné; lecture et publication ont utilisé le connecteur GitLab PC4 authentifié. Aucun push forcé ni fusion dans `main` n'a eu lieu. Les rapports R711 et R712, le sondage arXiv, les reçus de tests et le handoff courant sont inclus pour préserver la chaîne de reprise; leurs résultats historiques restent datés.

Le manifeste `research/R712/SHA256SUMS` conserve les digests des fichiers source/test avant les durcissements R713. Ses entrées pour ces deux chemins ne sont donc pas censées vérifier le contenu successeur présent dans cette branche; les autres fichiers historiques qu'il couvre restent inchangés. `research/R713/SHA256SUMS` fournit les digests actuels du code et des preuves inclus.

## Suite

R714 : relier le reçu du préflight aux requêtes EXEC de chaque Session et au Collector Rust existant, puis vérifier la sortie de couche contre un oracle indépendant qui reproduit l’ordre de réduction contractuel. Garder le cas synthétique CPU et zéro reprise UNKNOWN; ne prétendre aucune génération complète avant raccord de l’attention, des propriétaires réels, du checkpoint et de toutes les couches requises.
