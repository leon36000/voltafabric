# R695 — enveloppe mémoire par rang pour les routes MoE

## Modification

Le planificateur R694 savait réserver une quantité opaque de mémoire par appareil. R695 ajoute `PerRankMemoryEnvelope` et `plan_qwen2moe_advisory_with_memory`: chaque rang fournit séparément KV, état récurrent, workspace, graphes, communication, marge de sécurité et temporaire de chargement. Le plan refuse les rangs dupliqués/manquants, les additions qui débordent, et toute affectation dont le pic poids + réserves dépasse la capacité. Le temporaire est pris en compte pendant le placement, pas seulement dans le rapport final.

Le résultat conserve le Stamp, l'attempt, l'ordre des experts et les bits de coefficients du batch routé. Le registre de propriétaires et le dispatch sont en lecture seule. Un échec renvoie une erreur avant de rendre un avis partiel. Le budget récurrent est distinct du KV : il peut représenter l'état d'une couche GDN sans le traiter comme un cache attention.

## Réutilisation et sources

- Réemploi interne effectif : le calcul vérifié `memory::Budget::check` et les capacités/réserves de `Topology::Device`; aucun type de format quantifié n'est converti.
- Idée adaptée : la prépublication InplaceKVCache (arXiv:2609.14507v1) sépare la résidence physique du KV de la planification CPU/GPU. R695 n'en reprend aucun code et ne qualifie ni son format ni ses résultats de débit; il transpose uniquement la séparation des catégories mémoire dans le plan VoltaFabric.
- FluxMoE (arXiv:2604.02715v3) reste une référence de pagination d'experts, non réutilisée: son évaluation annoncée est sur H20/L40S et aucun code/licence donneur n'a été vérifié dans cette passe.
- Le chemin llama.cpp hérité de R693 reste épinglé à `ggml-org/llama.cpp` commit `3cf03257f219afbe7334045ff7c6a06ac68c627d`, MIT; R695 n'ajoute ni fichier upstream ni appel runtime dans `llama_decode`.

Les évaluations de ces prépublications sont celles de leurs auteurs; elles ne sont pas une preuve indépendante pour le parc VoltaFabric.

## Vérification

Sur les modules Rust exacts extraits de GitLab R693/R694, le test ciblé local a passé 6 cas: conservation des routes/poids/identités; ensemble de profils exact; refus faute de chemin; budgets KV/état/temporaire admis; refus de capacité sans avis partiel; rangs incomplets/dupliqués et overflow refusés. `rustfmt --check` et Clippy ciblé passent. Le crate racine complet n'a pas été compilé dans ce harnais; serde_json 1.0.145 du verrou racine n'était pas présent dans le cache offline, donc l'extraction a utilisé la version locale en cache 1.0.154.

## Limites et suite

Les entrées et topologie sont synthétiques. Pas de checkpoint entraîné, de session `llama_decode` reliée à cet avis, de KV/GDN d'un vrai modèle, de GPU, de réseau, de comparaison moteur ou de mesure de performance. Prochaine tranche: appeler ce conseiller de manière optionnelle sur le batch réellement capturé par la frontière llama.cpp, avant le commit d'exécution, en gardant l'avis sans autorité sur les propriétaires et les routes.

## Publication

The R695 source commit `dc8e737738042a7d6d2e025307a093b74846d830` is on the private GitLab branch and both Rust files were read back byte-exact. Pipeline 2370 passed its four configured scans/fallback jobs; this is not the CPU test or a Rust compilation result.
