# R365 — Priorité des experts résidents inspirée de Maya

## Source et périmètre

Project Maya, commit `171077787036ccc2784250ca644760f654e8d4a3`, [src/core/expert_cache.cpp](https://github.com/mw00/project-maya/blob/171077787036ccc2784250ca644760f654e8d4a3/src/core/expert_cache.cpp), SHA-256 `f8bbe8008771391274a2b1e12424fb4dde894514290d2655c78f38165520b325`. La licence racine Maya est [MIT](https://github.com/mw00/project-maya/blob/171077787036ccc2784250ca644760f654e8d4a3/LICENSE), les composants tiers peuvent avoir leurs propres licences. Aucune ligne du donneur n'est copiée : adaptation indépendante de l'idée de placer les experts résidents avant les autres, puis de trier par chaleur et ordre antérieur. Maya utilise des scores flottants et une politique concrète de cache ; VoltaFabric utilise des compteurs u64 et produit seulement un classement de métadonnées.

## Raccord et garanties

`routing::rank_resident_candidates` valide d'abord le vrai `RoutingProblem` et son registre, puis classe les experts par résidence sur le rang demandé, compteur décroissant, rang antérieur et identifiant. Les identifiants inconnus, doublons dans l'ordre antérieur, rang absent et métadonnées invalides sont refusés. Le classement ne modifie pas le dispatch, les coefficients, les copies résidentes ou les poids ; aucune admission, migration, éviction ou allocation n'est exécutée. Ne pas utiliser le résultat pour modifier un plan actif sans une admission de capacité et une transition d'époque atomique. Le contre-exemple R172 impose notamment de ne pas placer deux copies d'un expert sur le même rang.

## Vérification

- Reconstruction partielle CPU du socle GitLab `ef4e5d5d3536e73897967d9d25049194cbea5270` ; rustc 1.95.0, dépendances Cargo en cache.
- `cargo check --offline --locked --lib` : succès.
- `cargo test --offline --locked --test routing_heat` et la même commande avec `--release` : 3 tests réussis dans chaque profil. Les tests vérifient ordre et égalités, préservation du dispatch et des coefficients, invalidité de l'époque et du registre, noms inconnus et duplication.
- `rustfmt --check` ciblé : succès.
- Clippy ciblé avec `-D warnings -A clippy::manual_checked_ops` : succès. Sans l'exemption, rustc/Clippy 1.95 relève deux avertissements préexistants dans `src/session.rs` ; ils ne sont pas corrigés ici.
- `cargo test --lib` ne compile pas dans cette reconstruction partielle : fixtures binaires et fichiers d'évidence absents localement. Ce n'est pas un passage de la suite complète.

Aucun GPU, réseau inter-PC, chargement de poids, génération complète ou mesure de performance n'est qualifié. Prochaine étape : relier ce classement à une proposition d'admission bornée en mémoire et valider atomiquement son résultat avec le registre Rust ; restaurer les fixtures nécessaires aux tests plus larges.
