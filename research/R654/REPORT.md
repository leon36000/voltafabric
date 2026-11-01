# R654 — primitive CPU de matmul d'experts groupée avec GGML

## Adaptation

Le checkout llama.cpp/GGML MIT épinglé à `3cf03257f219afbe7334045ff7c6a06ac68c627d` expose `ggml_mul_mat_id`. Cette opération reçoit un tensor de matrices d'experts `[colonnes, lignes, experts]`, des activations `[colonnes, slots, tokens]` et des indices d'experts `[slots, tokens]`; le résultat conserve un output par slot. Son opérateur CPU regroupe les lignes par matrice sélectionnée avant d'exécuter les produits vectoriels.

VoltaFabric ajoute `routed_matrix_forward` dans l'adaptateur natif. La fonction borne les dimensions, les indices et le volume, appelle l'opérateur GGML épinglé et renvoie des lignes FP32 brutes, sans coefficients de route. Le calcul est synchrone et emprunte les buffers que le propriétaire maintient résidents; le code ne duplique pas les poids. L'appelant reste responsable de la validation du plan, de l'identité et de la résidence, du routage entre rangs et de l'application unique des coefficients dans le Collector.

Ce code appelle une API donneuse existante; il ne copie aucun fichier d'implémentation llama.cpp/GGML. Les sources amont et cette version portent MIT, avec les avis conservés dans l'adaptateur.

## Vérification

- Donneur exact : `3cf03257f219afbe7334045ff7c6a06ac68c627d`; clone de build propre avant compilation.
- `cmake -S adapters/native_ggml -B build -DGGML_SOURCE_DIR=<checkout-pinné> -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF` — PASS.
- `cmake --build build --target vf_native_ggml_test --parallel 2` — PASS sur PC6 AArch64 avec C++17, `-Wall -Wextra -Werror`.
- `ctest --test-dir build -R '^native_provider$' --output-on-failure` — 1/1 PASS.
- Le test envoie quatre slots à trois matrices Q8_0, avec deux slots dirigés au même expert et mêmes activations. Chaque élément est comparé à un oracle scalaire FP64 après déquantification indépendante. Les sorties des slots répétés sont égales. Un identifiant hors plage et une activation non finie sont refusés sans modifier la sortie.

Sources donneuses primaires, toutes au commit épinglé :

- [Déclaration de ggml_mul_mat_id](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/ggml/include/ggml.h)
- [Dimensions et construction de l'opération](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/ggml/src/ggml.c)
- [Dispatch CPU, regroupement des slots par expert et kernels](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/ggml/src/ggml-cpu/ggml-cpu.c)
- [Licence MIT](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/LICENSE)

## Portée

La configuration CMake a désactivé CUDA; cette preuve est CPU seulement. Le nouveau calcul n'est pas encore appelé par Rust Fleet, le protocole Session ou le callback Strata. Il n'ouvre ni checkpoint ni modèle, ne qualifie aucun format QPN2/NVFP4, MXFP4 ou IQ3, et ne prouve aucune génération complète, distribution GPU/inter-PC ou performance. Seul le chemin groupé Q8_0 a été testé ici; les autres formats restent à vérifier sur cette nouvelle frontière.

## Suite

R655 : relier les sorties groupées non pondérées au Collector CPU VoltaFabric sur un lot synthétique à identités fraîches, vérifier l'application des coefficients une seule fois et le refus atomique des sorties incomplètes. Aucun checkpoint, GPU ou réseau de production.
