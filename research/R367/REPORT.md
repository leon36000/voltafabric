# R367 — Admission exacte d'un lot de type Strata vers les sessions VoltaFabric

## Frontière observée et provenance

Source Strata installée sur PC1 : commit `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`, contrôlée en lecture seule. [PoolMultiFn](https://github.com/Niko1221/Strata/blob/fb58e0dbc8399662c0e47c76578c6e878b14f6cf/include/strata/core/verify.hpp) fournit les activations FP32, les identifiants, la couche et les dimensions ; sa signature ne fournit pas les coefficients. Ils doivent être récupérés séparément du gate de la même couche et du même batch. Les callbacks Strata sont `void` et le moteur peut poursuivre avant que le pilote ne lise le témoin d'erreur : ce lot ne corrige pas ce point dans Strata.

Empreintes de la source observée : `session.hpp` SHA-256 `ffe954333948aee170399252afaad2e8b2b1e81fb31ed9c4358597eba4ca2e3b`, `verify.hpp` `81e7b9c5eb7686f1c16d3a89ffd1b497443401d12cbc7a685b0f378e35be70d2`, `session.cpp` `b1479da93e913a0c944b1c61bb1fd623935cf68d7d02ae9ee1e32d7bb567625f`. La licence amont est MIT ; aucune ligne du donneur n'est copiée. Le lien du binaire Strata actif à ce checkout n'est pas établi.

## Adaptation

`strata_ingress::admit_pool_batch` reçoit séparément l'identité attendue, les lignes FP32, les IDs et les coefficients réels fournis par l'appelant. Elle vérifie le registre et le Stamp, la couche, les dimensions bornées, les experts, les doublons et les coefficients ; elle refuse tout ID négatif, dont les sentinelles éventuelles de hits GPU. Chaque activation doit être exactement représentable en FP16, zéros signés inclus ; aucune conversion approximative ni renormalisation n'est faite. Elle retourne le `RoutingProblem` et les bits FP16 seulement après validation complète.

Aucun worker n'est lancé et aucun attempt n'est attribué par l'admission ; l'authenticité et la provenance des coefficients fournis restent à établir par le futur raccord Strata. Le calcul et la pondération restent dans la Session et le Collector existants. L'API ne traite pas encore les hits GPU, ni le fournisseur IQ3 ; ces formats ne sont pas réinterprétés en NVFP4 ou MXFP4.

## Tests frais

Clone GitLab du commit parent `3f5ba664af878b2a02bbd783ad983147a3115ec3` quant au contenu du moteur ; Rust 1.90, Cargo hors ligne verrouillé, un job CPU. Le clone local était issu de R365, dont les sources sont identiques ; R366 ajoute seulement son rapport.

- `cargo test --offline --locked --test strata_ingress` en debug et release : **4 tests réussis** chacun. Les 63 488 motifs FP16 finis sont testés, ainsi que les activations FP32 inexactes, NaN, infini, débordement, mauvaise identité, formes, IDs, doublons et coefficients invalides.
- `cargo test --offline --locked --test a53_compressed_session admitted_strata_style_batch_reaches_real_local_sessions_and_dense_oracle` en debug et release : succès. Un lot synthétique est admis, les bits vont à deux vrais processus Session Rust locaux avec leurs propriétaires QPN2 de test, et la sortie correspond bit à bit à l'oracle dense. Ce ne sont pas des poids d'un modèle complet.
- `cargo test --offline --locked --test routing --test routing_heat` : 13 + 3 succès.
- `cargo test --offline --locked --lib` : 106 succès, 3 ignorés.
- `cargo clippy --offline --locked --lib --test strata_ingress --test a53_compressed_session -- -D warnings`, `rustfmt --check` ciblé et `git diff --check` : succès.

Les tests n'appellent pas Strata, ne lisent pas de vraies activations Strata et n'établissent pas la fréquence d'admission exacte en pratique. Aucun GPU, transport inter-PC, génération complète ni gain de performance n'est qualifié. Avant un raccord réel : fournir les coefficients authentiques, une voie numérique appropriée pour les activations FP32 inexactes, les hits et l'IQ3, et un contrôle d'échec avant copie/combinaison avec annulation/empoisonnement sûr du contexte. Les leases, les identités complètes et les UNKNOWN restent soumis aux contrats du projet.
