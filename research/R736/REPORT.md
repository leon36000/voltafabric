# R736 — couche d’attention fenêtrée DeepSeek et résiduel HC

## Résultat

Ajout de `src/deepseek_v41_attention_output.rs`, une frontière CPU scalaire pour
le chemin fenêtré non compressé de la révision DeepSeek-V4.1-Flash épinglée. Elle
charge par l’index Safetensors `attn_norm.weight`, `attn.attn_sink`, `wo_a.weight`,
`wo_b.weight` et `wo_b.scale`, avec un plafond de payload de 112 Mio. Les cinq
tenseurs doivent être dans le même shard; l’en-tête est relu après l’import pour
refuser un shard modifié. Le mapping index local de la couche 0 place ces noms
dans `model-00003-of-00048.safetensors`; le payload du shard n’a pas été lu.

La nouvelle passe liée à un `Stamp`/attempt fait, dans cet ordre : HC-pre sur le
résiduel quatre copies, RMSNorm BF16 avec epsilon `1e-20`, projections Q/KV R734,
RoPE Q/KV R735, cache/fenêtre et sink R733, RoPE inverse sur la sortie d’attention,
quantification-déquantification d’activation FP8 E4M3/UE8M0 en blocs de 32,
`wo_a` BF16 groupé (8 groupes de 4096 vers 1024), `wo_b` FP8 E4M3/E8M0, puis
HC-post de la branche attention vers quatre copies BF16. La réduction CPU de
`wo_a` est en FP32 et celle de la primitive FP8 existante en f64; les ordres de
réduction ne prétendent pas reproduire PyTorch/Torch ou TileLang bit à bit.

La mutation du KV est différée : la passe clone le cache de fenêtre (256 Kio pour
128 × 512 valeurs FP32), exécute la couche et ne remplace le cache original
qu’après validation de la projection et du HC-post. Une erreur garde donc le KV
et la position d’origine.

## Réemploi et source

Le donneur MIT est [DeepSeek-V4.1-Flash, révision
`314113f8a7dadb388215396ba5fbee2c610a04d6`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/inference/model.py).
Le fichier `model.py` téléchargé en mémoire depuis cette révision fait 61 669
octets et son SHA-256 est
`e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d`.
La source confirme `attn.attn_sink`, l’inverse complexe adjacent appliqué à la
sortie, la vue groupée de `wo_a`, son type BF16 et l’enchaînement `wo_b`. Aucun
code DeepSeek n’est copié.

Correction de la suite R735 : son texte appelait `wo_b` « BF16 ». Le code épinglé
construit cette projection avec le type FP8 E4M3 par défaut et les poids admis
utilisent les échelles E8M0; R736 suit ce format FP8/E8M0.

VoltaFabric réutilise son import Safetensors borné et son contrôle de hashes, sa
quantification FP8/E8M0 existante, la projection Q/KV R734, le RoPE et le cache
atomique R733, ainsi que le HC-pre et la réception exacte run/modèle/plan/couche/
batch/epoch/attempt. R736 ajoute la conservation du reçu de shard aux mixes HC et
refuse d’assembler QKV, HC et sortie si leurs révisions, index, shards ou en-têtes
diffèrent.

La taille brute demandée pour ces cinq tenseurs est 109 103 360 octets (~104,05
Mio), sous le plafond de 112 Mio. Le plan indexé identifie le shard mais ne fournit
pas les hashes de payload. L’appelant doit fournir les hashes tensoriels admis;
aucun poids réel n’a été chargé ici.

## Vérification et limites

- `cargo fmt --all -- --check` : réussi.
- `cargo check --locked --lib` : réussi.
- `cargo clippy --locked --lib -- -D warnings` : réussi.
- Aucun test, oracle numérique, poids réel ou génération n’a été exécuté dans
  cette tranche.
- Aucun GPU, transport inter-PC ou changement de service n’a été utilisé.
- Seule la voie purement fenêtrée est couverte; l’attention KV compressée,
  indexer, autres couches, gate/routage complet, FFN, tête de logits, tokenizer,
  échantillonneur et génération restent à relier et qualifier.
- Cette adaptation CPU ne bénéficie pas automatiquement de la qualité ou des
  garanties de performance du moteur donneur.

## Suite

R737 : relier l’attention R736 au gate, aux propriétaires d’experts et au FFN
VoltaFabric pour une couche complète identifiée, puis établir un oracle numérique
indépendant avant d’étendre la couverture aux chemins KV compressés. Garder les
comparaisons de performance en attente et les essais matériels sous les gates
existants.
