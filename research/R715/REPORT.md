# R715 — top-6 DeepSeek avec experts co-localisés

## Changement

R714 reliait six propriétaires MXFP4 à six Sessions distinctes. R715 accepte aussi plusieurs experts sélectionnés sur le même rang. Le préflight vérifie désormais l'unicité `(rank, local)` tout en conservant l'unicité des six experts et l'identité complète de la route.

`group_pinned_mxfp4_route_residents` combine les propriétaires sélectivement lus en un artifact Safetensors par rang. L'ordre des locaux doit correspondre exactement au `resident_order` du dispatch; un rang contenant des locaux supplémentaires est refusé par cette frontière. Les six noms de tenseurs, types, formes, octets et empreintes par tenseur sont vérifiés avant la sérialisation. L'artifact groupé reste sous la limite existante de 64 MiB. La provenance conserve, pour chaque expert, le nom de shard et le SHA-256 de son en-tête source; le regroupement ne suppose donc pas une source unique, mais le test de cette tranche n'a utilisé qu'un seul shard synthétique.

Le budget de construction est explicite. La borne préflightée compte les artifacts sources retenus, deux copies du payload et deux marges d'en-tête. Le dépassement du budget de pic ou de la limite d'artifact refuse avant la connexion de Sessions. Les artefacts assemblés doivent aussi tenir dans le budget agrégé du préflight.

`execute_pinned_mxfp4_route_grouped` consomme un reçu LOAD par rang, vérifie qu'il couvre exactement les propriétaires choisis et que le jeu de Sessions correspond aux rangs actifs. Il prévalide toutes les Sessions avant le premier EXEC. Une Session peut recevoir plusieurs slots; le Collector R9 conserve son contrat de résultats FP32 prépondérés et chaque coefficient de route n'entre qu'une fois dans la réduction. Une sortie n'est publiée qu'après la complétion de tous les slots et `collector.finish()`. Un résultat ambigu n'est jamais renvoyé et aucune sortie partielle n'est exposée.

## Preuve CPU

La fixture conserve six experts MXFP4 synthétiques de 18 800 640 octets chacun. Deux experts sélectionnés partagent le rang 4; quatre autres utilisent chacun leur rang. Cinq processus CPU chargent donc cinq artifacts et exécutent les six slots de la route. La sortie est comparée à `Mxfp4Expert::forward_reference`, avec une tolérance maximale imposée de `5e-4`; la fixture utilise des coefficients exactement représentables en FP32, appliqués une seule fois. Le test confirme la complétion des six slots et un EXEC par rang.

Vérifications effectuées :

- tests d'intégration ciblés debug : 4 passés; test groupé isolé : 1 passé;
- tests d'intégration ciblés release : 4 passés; test groupé isolé : 1 passé;
- tests de lecture sélective debug et release : 6 passés dans chaque configuration;
- Clippy avec `-D warnings`, `cargo fmt --check` et `git diff --check` : passés.

## Réemploi, portée et limites

Cette tranche réutilise le routeur, le format MXFP4/Safetensors, les Sessions CPU, le protocole LOAD/EXEC et le Collector R9 déjà dans VoltaFabric. Aucun fichier de code tiers n'a été copié. Les composants donneurs examinés et le sondage arXiv restent des références de conception; cette preuve ne valide pas leur compatibilité générale ni leurs performances. Les formats QPN2/NVFP4 K16, MXFP4 K32 et IQ3 restent distincts.

Les poids sont synthétiques. La topologie deux-plus-quatre est une fixture et ne provient pas d'un registre/plan de production. Aucun payload de checkpoint réel, génération complète, attention, cache KV, tokenizer, GPU, V100, Spark/RTX 5080, transport inter-PC, RDMA ou R53 de production n'est couvert. La CI GitLab ne prouve pas les tests CPU locaux.
