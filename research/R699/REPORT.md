# R699 — conseil mémoire générique avant dispatch MXFP4 DeepSeek

## Résultat

Le planificateur VoltaFabric expose maintenant son conseil de routage et son budget mémoire via `RoutingProblem`, au lieu d’exiger le wrapper Qwen2MoE. Les wrappers Qwen2MoE existants délèguent à cette API. Le conseil conserve le `Stamp`, l’`attempt`, l’ordre des experts et les bits des coefficients, exige un profil pour chaque expert routé et un envelope mémoire pour chaque rang, et reste en lecture seule.

Le test utilise les fragments épinglés des experts DeepSeek-V4.1-Flash 0 et 383, MXFP4 K32, dans deux Sessions CPU persistantes et le Collector FP32 prépondéré. Il calcule le budget avant de créer les Sessions; une capacité inférieure d’un octet au pic de chargement refuse l’admission. La route utilise le stamp R699 et le même attempt 903 dans le conseil et les appels Session. Le test compare l’agrégation aux vecteurs-oracle BF16 épinglés avec un seuil absolu de 5e-4.

## Réutilisation et provenance

- Réutilisation interne: `RoutingProblem`/dispatch, profils d’experts, `Budget::check`, envelopes par rang, `Mxfp4ExpertShard`, Session MXFP4 persistante et Collector FP32 prépondéré. Le conseil générique ne modifie ni le gate, ni le registre, ni les coefficients.
- R699 ne copie aucun code d’un moteur tiers. Les vrais octets d’experts viennent du dépôt modèle DeepSeek-V4.1-Flash, révision `314113f8a7dadb388215396ba5fbee2c610a04d6`, licence MIT consignée à R698. Les artefacts sont des poids de test, pas une intégration logicielle DeepSeek.
- R697 a exploré séparément la frontière publique llama.cpp pour Qwen2MoE; cette tranche n’appelle pas `llama_decode` et ne revendique pas de réutilisation nouvelle de llama.cpp, vLLM, SGLang, TensorRT-LLM, Strata, Maya ou DeepInfer.
- Le test R694 révélait aussi un import de crate inexistant (`voltafabric_r694_exact_source_check`). Il pointe maintenant vers le nom réel `voltafabric` du `Cargo.toml`; ses sept tests ciblés repassent.

## Vérification

Dans le worktree isolé basé sur le commit local `604764703f0a12af4fe264041ebecc0ed9c21a90`, avec le planner et routing R698 relus depuis GitLab et le patch R699: `rustfmt --edition 2021 --check ...` passe; `cargo test --offline --locked --test r699_deepseek_memory_admission --test r694_routed_advice -- --nocapture --test-threads=1` passe (2 tests R699, 7 tests R694; 10,36 s pour R699); `cargo clippy --offline --locked --lib --test r699_deepseek_memory_admission --test r694_routed_advice -- -D warnings` passe. Le pipeline GitLab ne compile pas Rust; son état sera consigné après achèvement.

## Limites

La route et la topologie mémoire sont synthétiques; les coefficients 0,375 sont ceux des fixtures oracle, pas les sorties du gate DeepSeek. Les capacités utilisent les tailles réelles des deux fichiers comme profils d’experts, mais aucun capteur ne mesure la VRAM/RAM disponible. Le planner est advisory: le test prouve l’ordre « conseil puis création Session » dans ce chemin, pas une réservation durable ou transaction multi-rang généralisée. Il n’y a ni checkpoint complet, ni tokenizer, ni attention/KV, ni routeur/gate, ni génération, ni accélérateur, ni échange inter-PC ou réseau de production.

## Suite

Brancher les sorties du gate réel à cette API typée, alimenter les envelopes avec les allocations résidentes et les mesures d’exécution autorisées, puis concevoir le chargement multi-rang transactionnel qui refuse toute publication de poids partielle. Continuer en parallèle l’examen source/licence des moteurs donneurs; ne pas présenter les artefacts de modèle comme du code réutilisé.
