# R737 — composition attention + FFN pour une couche fenêtrée

## Résultat

`src/deepseek_v41_window_block.rs` relie les frontières R736 en un bloc token
identifié : HC-pre attention, Q/KV et attention fenêtrée, HC-post attention, HC-pre
FFN, gate et plan top-6, expert partagé, reçu de route, somme routée+partagée, puis
HC-post FFN. Les composants du checkpoint et les propriétaires de couche sont
fournis déjà admis; ce code n'effectue aucun accès disque aux poids.

L'exécuteur de route est un appel `FnOnce` explicite. Il doit soumettre exactement
le plan retourné via les Sessions VoltaFabric et retourner le reçu terminal. Le
composeur compare Stamp, attempt, experts et fingerprint au plan privé avant toute
composition; la fonction de combinaison existante refuse les routes Aborted ou
UNKNOWN et garde les coefficients dans le Collector prépondéré existant. L'expert
partagé est ajouté une seule fois.

La passe opère sur une copie du cache KV. Le cache d'origine n'avance qu'après le
reçu `Completed`, la combinaison routed+shared et la validation du HC-post FFN. Une
erreur ou un reçu UNKNOWN ne publie pas de résiduel partiel et ne modifie pas la
position de l'appelant.

## Réemploi et limites de la frontière

Cette adaptation réutilise les types et opérateurs déjà présents dans VoltaFabric :
recevabilité HC R736, projection et cache fenêtré, gate et plan de charge indexé,
Collector MXFP4 FP32 prépondéré, somme de l'expert partagé et identités de couche.
Les sémantiques du modèle donneur MIT DeepSeek-V4.1-Flash restent une référence; aucun
code donneur n'est copié. Cette tranche n'intègre pas de composant vLLM, TensorRT-LLM,
llama.cpp, SGLang, Maya ou Strata.

La fermeture d'exécution est une frontière de confiance : son type ne prouve pas que
les experts sont toujours résidents, ni qu'un appelant ne les recharge pas. Le
contrat exige une exécution unique contre les propriétaires déjà résidents; la
preuve de résidence doit venir de l'adaptateur Session concret. Aucun exécutant réel
ou checkpoint n'a été branché ici. R736 reste la seule qualification de la passe
d'attention CPU.

## Vérification et limites

- `cargo fmt --all -- --check` : réussi.
- `cargo check --locked --lib` : réussi.
- `cargo clippy --locked --lib -- -D warnings` : réussi.
- Aucun test ajouté ou lancé, aucun oracle numérique indépendant.
- Aucun payload de checkpoint lu, aucun GPU, réseau inter-PC ou benchmark.
- Ce n'est ni une couche entière exécutée ni une génération complète. L'attention KV
  compressée, les logits, tokenizer, échantillonneur et séquence multi-token restent
  à intégrer et qualifier.

## Suite

R738 : remplacer la fermeture abstraite par un adaptateur qui démontre la résidence
et l'identité des propriétaires MXFP4 dans les Sessions, sans LOAD de poids sur le
chemin token; établir ensuite un oracle numérique indépendant avant toute extension.
