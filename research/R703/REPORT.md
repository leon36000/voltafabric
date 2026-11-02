# R703 — Inventaire des poids épinglés et garde Safetensors

Date : 2026-11-02  
Branche : `r703-gate-artifact-inventory-20261102`  
Parent : reçu R702 `d4a20ec8d8bc91e2c4cb9f45e1fc7159d9553c2f`

## Résultat

L’inventaire en lecture seule a parcouru l’arbre GitLab privé R702 complet : 7 920
entrées en 80 pages. Il confirme que les octets réels du gate DeepSeek et les quatre
profils nécessaires aux routes top-6 `[0, 383, 382, 381, 380, 379]` ne sont pas dans
les artefacts suivis de cette branche.

Présents :

- le header et l’index safetensors épinglés pour `model-00003-of-00048.safetensors` ;
- les sources et la licence MIT du donneur DeepSeek à la révision
  `314113f8a7dadb388215396ba5fbee2c610a04d6` ;
- des fixtures de poids réels pour les experts 0 et 383 ;
- une fixture de poids groupée experts 96–192, avec oracles pour 96 et 192.

Absents dans l’arbre complet parcouru : le shard gate annoncé à 7 389 759 032 octets
et les fichiers `fixtures/deepseek_v41_layer0_expert379_314113f8.safetensors`,
`...expert380...`, `...expert381...` et `...expert382...`. Les experts 96 et 192 ne
remplacent pas les quatre choix du gate. Le plan de Session reste donc atomiquement
refusé ; aucune route n’a été changée pour faire entrer les seuls profils présents.

## Code et vérification

`tests/r703_safetensors_fail_closed.rs` exerce le lecteur sélectif ajouté en R702 avec
des enveloppes Safetensors minimales construites en mémoire. Cinq tests vérifient une
lecture exacte, les clés JSON dupliquées (y compris dans `__metadata__`), des offsets
qui se chevauchent, des octets finaux non indexés, un header déclaré au-delà de la
borne sans corps, ainsi qu’une sélection dupliquée. Les erreurs sont des refus ; aucun
objet partiel n’est rendu.

Commandes locales, dans le worktree disponible :

- `CARGO_BUILD_JOBS=1 cargo test --offline --locked --test r703_safetensors_fail_closed` — 5 réussis ;
- `CARGO_BUILD_JOBS=1 cargo clippy --offline --locked --test r703_safetensors_fail_closed -- -D warnings` — réussi ;
- `rustfmt --edition 2021 --check tests/r703_safetensors_fail_closed.rs` — réussi ;
- `git diff --check` — réussi.

Ces tests CPU et l’inventaire GitLab ne chargent aucun poids du checkpoint, ne lancent
pas de Session, de GPU ou de transport inter-PC. Le pipeline GitLab de sécurité n’est
pas une compilation Rust.

## Réutilisation et limites

Le test traverse la frontière existante `safetensors = 0.8.0` (Apache-2.0), déjà
épinglée, et le lecteur Rust de VoltaFabric. Aucun code donneur n’a été copié. Les
artefacts DeepSeek sont des références MIT versionnées ; cette provenance ne certifie
ni un checkpoint complet, ni la parité bit à bit, ni une génération entière.

Prochaine tranche : poursuivre côté moteur CPU avec les composants déjà suivis et
maintenir le refus avant dispatch tant que le gate et tous les experts choisis ne sont
pas disponibles. Pas de téléchargement de checkpoint entier, GPU, réseau de
production, rejeu UNKNOWN ou relance automatique.
