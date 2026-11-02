# R704 — Liaison du gate DeepSeek vers le routage VoltaFabric

Date : 2026-11-02  
Branche : `r704-deepseek-gate-route-identity-20261102`  
Parent : reçu R703 `bcaa7ee42752abfa597d1333fd1e20f9021d7a3c`

## Tranche intégrée

`DeepSeekV41Gate::prepare_route_batch` convertit une décision top-6 d’un token en
`RoutingProblem` VoltaFabric et appelle le validateur commun avant de rendre le lot.
Le nouveau `DeepSeekV41RouteBatch` garde le problème privé et transporte l’attempt
non nul avec l’identité run/modèle/plan/couche/batch/epoch du `Stamp`. La méthode
refuse un attempt nul, une révision ou une couche différente du gate chargé, une
route qui n’est pas top-6 et toute incohérence registry/rank/local/order détectée
par `routing::validate`.

Les coefficients retournés par le gate sont transmis tels quels. Aucun tri,
renormalisation ou second `route_scale` n’est appliqué. Le lot peut être fourni au
planner puis à une Session par l’appelant, après la validation de mémoire et la
présence effective des poids.

## Vérification CPU

Le test de projection R702 a été étendu pour construire un lot à partir du fixture
synthétique Safetensors, vérifier l’identité complète, l’attempt 29, l’ordre des six
experts et la somme des coefficients `ROUTE_SCALE`, puis vérifier le refus du mauvais
layer et de l’attempt zéro.

- `CARGO_BUILD_JOBS=1 cargo test --offline --locked --test r702_deepseek_gate_projection` — 3 réussis ;
- `CARGO_BUILD_JOBS=1 cargo clippy --offline --locked --test r702_deepseek_gate_projection -- -D warnings` — réussi ;
- `rustfmt --edition 2021 --check src/deepseek_v41_gate.rs tests/r702_deepseek_gate_projection.rs` — réussi ;
- `git diff --check` — réussi.

## Réutilisation et limites

Cette tranche réutilise l’API `RoutingProblem` et `routing::validate` de VoltaFabric,
et s’appuie sur l’adaptation du gate DeepSeek V4.1 Flash MIT à la révision
`314113f8a7dadb388215396ba5fbee2c610a04d6`. Aucun code donneur supplémentaire n’a
été copié. Le parser Safetensors reste la dépendance Apache-2.0 déjà épinglée.

Le test reste synthétique et CPU. R703 a parcouru 7 920 entrées GitLab et établi que
les poids réels du gate et les experts 379–382 sont absents des artefacts suivis. Ce
raccord ne lance pas le planner, ne démarre pas de Session et ne qualifie pas la
génération. Le dispatch doit rester refusé tant que le gate réel, les six profils,
leurs poids résidents et l’admission mémoire complète ne sont pas disponibles.
Pas de téléchargement du checkpoint entier, GPU, réseau de production, rejeu
UNKNOWN ou relance automatique.
