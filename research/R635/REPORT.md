# R635 — contrôler le plan avant toute réservation R53

## Changement

Le mode explicite `reserve_r53=True` du runner de prompt calcule maintenant le plan effectif avec le même exécutable épinglé et les mêmes entrées via `--plan-only`, puis vérifie l’identité attendue avant d’importer et d’appeler le module de réservation. Un plan différent interrompt donc le flux avant consommation d’identités R53. Le test ajouté vérifie aussi que l’invocation complète du moteur n’est jamais lancée après ce refus.

Cette tranche durcit le pont déjà construit dans R631–R634; elle ne copie pas de code d’un nouveau moteur. Les adaptations donneuses déjà conservées restent R626 (composant GGML de llama.cpp, commit `3cf03257f219afbe7334045ff7c6a06ac68c627d`, MIT) et R627 (tête Strata, commit `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`, MIT). R630 a terminé une génération CPU SC117 locale à travers 48 couches et les experts VoltaFabric. Ces résultats ne prouvent ni exécution GPU/inter-PC, ni intégration au runtime complet de Strata.

## Vérifications

- 43 tests `adapters.strata_tokenizer.test_prompt_runner` réussis dans l’environnement Python existant de Strata sur PC1.
- Vérification du coordinateur réel en mode préflight uniquement : deux appels `--plan-only`, le second avec une identité attendue volontairement différente; refus exact `engine receipt differs from expected plan`.
- Le lanceur de test a interdit la création de sockets IPv4/IPv6; seul AF_UNIX était admis. Le worker configuré était le stub `/usr/bin/false`.
- Le registre testé était synthétique et est resté vide. Aucun R53 réel, aucune lease de production, aucun worker, aucune génération complète et aucun GPU n’ont été utilisés.
- Le reçu et les empreintes des artefacts observés figurent dans `receipt.json`. Les 768 réservations et 2006 claims inconnus historiques de R630 demeurent `RESERVED_UNKNOWN_NO_REPLAY`.

## Limites et suite

Le préflight réduit un échec prévisible après réservation, mais ne garantit pas que la lease, le registre partagé, le checkpoint ou les admissions matérielles soient valides; ces contrôles restent nécessaires et une panne après démarrage reste `UNKNOWN`, sans rejeu automatique. Prochaine tranche proposée : raccorder `PoolMultiFn` de Strata à la frontière VoltaFabric réelle et vérifier cette intégration sur CPU avec un oracle indépendant. GPU et R53 réel restent suspendus jusqu’aux admissions distinctes.
