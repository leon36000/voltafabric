# R653 — essai de compilation du callback Strata exact

## Résultat

Le générateur de patch R641 a été rejoué sur le checkout local exact du donneur Strata `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`. Il a reproduit le patch de 21 616 octets avec le SHA-256 publié `9dfe17d046858f41bb85fbccde55b8c460cd969b0b42ae799786a874a7b63e2c`. Les patches R639 et R640 ont aussi correspondu à leurs hashes publiés. Les trois changements ont été appliqués dans un clone isolé; `git diff --check` a réussi et le checkout donneur source est resté propre.

La configuration CMake opt-in a réussi avec CUDA Toolkit 13.0 et `CMAKE_CUDA_ARCHITECTURES=80`. La compilation du binaire `strata` a échoué avant la compilation de `src/core/expert_source.cpp` et `src/program/generate.cpp`: sur l’hôte PC6 `aarch64`, `src/core/graph.cpp:4` inclut `<immintrin.h>`, absent de cette plateforme. La commande a arrêté à 9%; aucun exécutable adapté n’a été produit.

Le CMake du commit épinglé annonce que ses cibles CUDA requièrent SM80 ou supérieur. Le backend CUDA de ce donneur ne convient donc pas au V100 SM70, même si le problème d’intrinsics ARM était corrigé. Strata reste un donneur d’idées de dispatch et de traitement des échecs; cette version n’est pas un moteur CUDA cible qualifié pour V100 ou pour compiler nativement sur PC6 ARM64.

## Commandes reproduites

- `python3 -B research/R641/make_generator_patch.py <checkout-fb58> <R639-patch> <R640-patch> <generated-r641.patch>` — PASS; hash exact du patch publié.
- Application des patches R639, R640 et R641 dans le clone isolé puis `git diff --check` — PASS.
- `cmake -S <Strata-patched> -B <build> -DSTRATA_ENABLE_CUDA=ON -DSTRATA_VOLTAFABRIC_POOL=ON -DVOLTAFABRIC_ADAPTER_ROOT=<VoltaFabric> -DCMAKE_CUDA_ARCHITECTURES=80 -DSTRATA_BUILD_TESTS=OFF -DSTRATA_MMQ_KQUANTS=OFF -DSTRATA_Q6K_EXPERTS=OFF` — configuration PASS.
- `cmake --build <build> --target strata --parallel 2` — FAIL à 9%: `src/core/graph.cpp:4:10: fatal error: immintrin.h: No such file or directory` sur `aarch64`.

Les sources Strata au commit indiqué portent la licence MIT. Aucune source donneuse n’a été copiée dans le moteur VoltaFabric par cette tentative; la frontière d’adaptation R641 reste un patch opt-in et n’a pas été compilée dans cette tentative.

## Sources donneuses consultées

- `src/core/expert_source.cpp` au commit épinglé documente le pinning comme opt-in et signale le risque de corruption IQ3_S associé aux buffers; voir [Strata expert_source.cpp](https://github.com/Niko1221/Strata/blob/fb58e0dbc8399662c0e47c76578c6e878b14f6cf/src/core/expert_source.cpp).
- `CMakeLists.txt` au même commit fixe les cibles CUDA à SM80 ou supérieur; voir [Strata CMakeLists.txt](https://raw.githubusercontent.com/Niko1221/Strata/fb58e0dbc8399662c0e47c76578c6e878b14f6cf/CMakeLists.txt).
- La licence du commit donneur est [MIT](https://github.com/Niko1221/Strata/blob/fb58e0dbc8399662c0e47c76578c6e878b14f6cf/LICENSE).

## Portée

Pas de modèle, de checkpoint, de génération, d’appel de session, de reçu de travail, de GPU, de réseau inter-PC ou de réservation R53. La configuration a demandé la compilation pour SM80; aucun code n’a été exécuté sur GPU. Le blocage intervient dans un fichier du donneur avant les fichiers contenant le callback VoltaFabric. La compilation partielle et le clone sont conservés sous `.runs/r643-exact-r642-20261101/.runs/r653-strata-callback-20261101/`.

## Prochaine tranche

Examiner le backend CUDA de llama.cpp épinglé par R643 pour SM70, puis identifier une fonction native à intégrer à la frontière VoltaFabric et la vérifier sur CPU ou par compilation ciblée. Garder les contrats IQ3, MXFP4 et NVFP4 distincts; aucune comparaison de performance ni exécution matérielle n’est demandée par cette preuve.
