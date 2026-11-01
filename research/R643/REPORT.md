# R643 — fermeture de source du pont Strata/GGML (CPU)

## Changement

La branche GitLab R642 contient et suit bien `fixtures/strata_sc117_native_experts_v3.txt`, référencé par le Rust via `include_str!`. La lecture distante confirme 49 lignes et 3 316 octets; la copie locale utilisée pour la compilation a le SHA-256 `adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24`, identique à l'identité de registre consignée dans R604. Ce fichier est un petit index de types, offsets et tailles, sans données de tenseurs ni poids. R643 ajoute la preuve de compilation et la portée de validation; le code source du moteur reste celui de R642.

Le snapshot de validation a été relu depuis le commit R642 `a19ab47b67e7973100ec1375eba77bdf8d709e50`. Les 143 fichiers suivis sous `src/`, `crates/` et les trois manifestes Cargo/toolchain correspondent aux identifiants d'objets GitLab. Les six entrées C++ pertinentes (`CMakeLists.txt`, `bridge.cpp/.h`, `provider.cpp`, `provider.hpp`, `provider_test.cpp`) correspondent aussi aux blobs de R642.

## Validation CPU

- `cargo check --locked --offline --lib` — PASS, snapshot exact R642 plus registre.
- CMake a configuré le pont contre le checkout llama.cpp propre, révision MIT `3cf03257f219afbe7334045ff7c6a06ac68c627d`; backend CPU uniquement.
- `cmake --build ... --target vf_native_ggml_test --parallel 1` — PASS.
- `ctest ... -R '^native_provider$' --output-on-failure` — 1/1 PASS, vecteurs synthétiques CPU.
- `cargo build --locked --offline --bin vf-strata-native-bridge --features native-ggml-cpu` — PASS avec ces bibliothèques fraîchement construites.
- L'exécutable réel refuse `{}` avant `ready` (code 2, aucune sortie stdout; erreur `missing field template`). C'est uniquement un contrôle de rejet d'initialisation mal formée.

Un avertissement Rust non bloquant subsiste pour deux champs `CapturedQsa` inutilisés.

## Portée

Cette tranche ferme un trou de source et vérifie la compilation du chemin Rust–GGML CPU. Elle ne prouve ni le chargement d'un checkpoint réel, ni une sortie numérique d'expert via le pont Rust, ni le branchement au callback Strata, ni une génération complète. Aucun modèle, GPU, RDMA, réseau de production ou réservation R53 n'a été utilisé. Les adaptations Strata/llama.cpp restent expérimentales; leur licence MIT et leurs limites sont décrites dans `adapters/native_ggml/README.md` et `adapters/strata_native_bridge/NATIVE_BRIDGE.md`.

## Suite

Écrire une qualification CPU bornée du protocole Rust avec propriétaires synthétiques, en conservant les contrôles d'identité, l'application unique des coefficients et le refus atomique. Le démarrage avec le modèle complet attend une admission et les ressources fraîches requises; les performances restent en attente.
