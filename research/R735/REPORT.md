# R735 — profil RoPE DeepSeek borné et raccordé à la fenêtre R733

## Résultat

Ajout de `src/deepseek_v41_rope.rs`. Le chargeur n'accepte que la configuration officielle épinglée par son SHA-256, un layer backbone dont `compress_ratios[layer] == 0`, et le profil d'attention fenêtrée Q/KV correspondant. Il lit `qk_rope_head_dim=64`, `head_dim=512`, 64 têtes, fenêtre 128, theta 10000 et limite de positions 1,048,576. Les couches compressées sont refusées par cette frontière.

La sortie R734 est exacte BF16 pré-RoPE. R735 fait tourner les 64 dernières composantes de chaque tête Q et du KV partagé en paires complexes adjacentes, caste chaque résultat en BF16 RNE, conserve le même Stamp et l'attempt, et expose `append_and_attend` vers le cache atomique R733. Le cache doit avoir la fenêtre épinglée de 128; l'échelle softmax est fixée en interne à `head_dim^-0.5` au lieu d'être choisie par l'appelant. Les contrôles d'identité et de position du cache restent l'autorité avant mutation.

## Source et décision de configuration

Donneur MIT `DeepSeek-V4.1-Flash`, révision `314113f8a7dadb388215396ba5fbee2c610a04d6`; `model.py` SHA-256 `e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d`. La config officielle épinglée a SHA-256 `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879`.

Le snapshot `ModelArgs` a un défaut `rope_head_dim=32`, tandis que la config du checkpoint indique `qk_rope_head_dim=64`. Le snapshot ne contient pas de chargeur qui mappe ces deux noms. L'adaptateur n'utilise donc pas ce défaut : il exige la config épinglée et sélectionne explicitement ses champs pour ce profil. Le mapping concret vers l'instance d'inférence du donneur reste non qualifié. Le `model.py` donne aussi `compress_ratio=0` → `original_seq_len=0` et `rope_theta=args.rope_theta`, ce qui désactive YaRN pour la voie purement fenêtrée; le theta sélectionné est 10000. Le scaling YaRN de la config HF et le theta de KV compressé ne sont pas appliqués à cette frontière.

Le helper Qwen CPU local utilise une rotation split-half avec theta 10,000,000. Cette convention ne correspond pas aux paires complexes adjacentes de DeepSeek et n'est pas réutilisée directement. Aucun code donneur n'a été copié.

## Réemploi et limites

Réutilisés : cast BF16 RNE et identité `Stamp` de VoltaFabric; Q/KV résidents de R734; cache et contrôle atomique de position/identité R733. C'est une adaptation CPU locale d'un contrat précis, pas un kernel issu du moteur donneur.

La vérification R735 s'est limitée à `cargo check --locked --lib`, `cargo fmt --all -- --check` et `cargo clippy --locked --lib -- -D warnings`, tous réussis après l'ajout du contrôle de fenêtre et de l'échelle fixée. Aucun test fonctionnel ou oracle numérique RoPE n'a été exécuté dans cette tranche; aucune parité bit à bit Torch/CUDA n'est revendiquée. Aucun payload modèle, GPU, génération, transport inter-PC ou mesure de performance n'a été utilisé.

Sink attention, lecture réelle du payload, chemin KV compressé/indexer, projection `wo_a/wo_b`, composition résiduelle HC, tokenizer, tête de logits, échantillonneur et génération restent à faire.

## Suite

R736 : charger par index le sink d'attention et `wo_a/wo_b` de la couche épinglée, puis relier la passe d'attention à la composition résiduelle HC sous le même Stamp/attempt. Maintenir les poids résidents et l'admission bornée; pas de téléchargement complet du modèle ni d'exécution GPU.
