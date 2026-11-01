# R655 — frontière GGML groupée vers le Collector VoltaFabric

## Adaptation

À la suite de R654, la frontière native ajoute un résident Rust `vf_ggml_native_ffi::NativeRoutedMatrix` pour une pile bornée de matrices GGML de même format. Le chargement copie les poids une fois dans le résident C++; chaque appel synchrone emprunte ce tampon et renvoie les lignes FP32 brutes dans l'ordre des slots. Le pont valide les dimensions, le nombre d'experts, les identifiants et les valeurs finies, puis ne publie les résultats qu'après le calcul complet.

Le test d'intégration compose le dispatch VoltaFabric, cette frontière FFI, `completion::accept_routed_output` et `moe::Collector::from_routing`. Il utilise une permutation de l'ordre physique des experts, deux tokens, des coefficients distincts et un numéro de tentative. Le Collector reçoit les sorties non pondérées arrondies en FP16; son accumulation f64 restitue les sorties attendues `3.5` et `2.0`, ce qui vérifie que le coefficient de chaque slot est appliqué une seule fois. Une tentative périmée est refusée avant admission; un lot avec un slot manquant ne produit pas de sortie et finit en `BATCH_INCOMPLETE`.

Le donneur reste llama.cpp/GGML, commit MIT `3cf03257f219afbe7334045ff7c6a06ac68c627d`. VoltaFabric appelle `ggml_mul_mat_id`; aucun fichier d'implémentation amont n'est copié. Le format testé est Q8_0. Il reste distinct de QPN2/NVFP4, MXFP4 et IQ3.

## Vérification

- CMake sur PC6 AArch64 avec le checkout amont exact : configuration et construction de `vf_native_ggml` et `vf_native_ggml_test` — PASS.
- `ctest --test-dir build -R '^native_provider$' --output-on-failure` — 1/1 PASS; l'oracle scalaire FP64 indépendant de R654 est conservé.
- `cargo test --offline --manifest-path harness/Cargo.toml --features native-ggml-cpu --test native_matrix` — 2/2 PASS en debug et 2/2 PASS en release.
- `rustfmt --edition 2021 --check crates/ggml-native-ffi/src/lib.rs tests/native_matrix.rs` — PASS.
- Clippy ciblé — PASS après autorisation de deux avertissements présents dans les sources inchangées (`dead_code` et `clippy::too_many_arguments` dans le callback Strata). Le premier passage strict a signalé ce dernier avertissement préexistant.
- Un premier essai Cargo sur le manifeste complet du dépôt isolé a tenté de construire les binaires historiques dont les fixtures n'avaient pas été copiées dans ce scratch. Aucun test n'a été lancé par cet essai; le test ciblé a ensuite été exécuté dans un petit workspace isolé contenant les sources R654 exactes.
- Les sorties GGML brutes de la fixture sont à environ `1.99988` et `3.99976` avant conversion FP16; les valeurs passent le contrat de tolérance quantifiée et le résultat FP16 du Collector correspond exactement à l'oracle de pondération par slot.

## Portée

Cette preuve relie réellement l'opération groupée GGML à l'API du Collector VoltaFabric dans un test CPU isolé avec identités run/modèle/plan/couche/batch/epoch et tentative. Elle n'appelle pas la Session persistante, le callback Strata ou un chemin de génération de modèle. Elle ne charge aucun checkpoint, ne qualifie ni V100/GPU ni réseau inter-PC, et n'utilise pas R53 de production.

## Suite

R656 : lier le résident matriciel groupé à une session CPU VoltaFabric identifiée, avec plusieurs batches frais et un refus atomique de l'échec d'un propriétaire, puis évaluer les formats admissibles sans les confondre. Pas de checkpoint, GPU ou réseau de production.

## Sources primaires épinglées

- [Déclaration de `ggml_mul_mat_id`](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/ggml/include/ggml.h)
- [Construction et dimensions de l'opération](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/ggml/src/ggml.c)
- [Dispatch CPU GGML](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/ggml/src/ggml-cpu/ggml-cpu.c)
- [Licence MIT](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/LICENSE)
