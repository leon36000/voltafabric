# R734 — Projections Q/KV résidentes de l’attention DeepSeek-V4.1

## Résultat

Ajout d’un propriétaire CPU immuable pour les projections attention du backbone
DeepSeek-V4.1 épinglé : `wq_a` FP8, `q_norm` BF16, `wq_b` FP8, `wkv` FP8 et
`kv_norm` BF16. Le chargeur indexé sélectionne uniquement ces huit tenseurs,
vérifie l’index, le shard, le type, la forme, la taille et chaque empreinte fournie,
puis garde les matrices résidentes. L’opération à un token rend Q et KV BF16 avant
RoPE avec le même `Stamp` et l’attempt.

Les matrices restent au format source F8_E4M3 + échelles E8M0 32×32; ce n’est ni
MXFP4 K32, ni QPN2/NVFP4 K16, ni IQ3. Le produit CPU réutilise l’activation FP8
E4M3/E8M0 de VoltaFabric. `wq_b` est découpée en blocs de lignes pour que chaque
matvec respecte le plafond de travail CPU déjà appliqué à la référence FP8.

## Vérification de la source épinglée

- Donneur : `DeepSeek-V4.1-Flash`, commit MIT
  `314113f8a7dadb388215396ba5fbee2c610a04d6`; aucun code donneur copié.
- `model.py` SHA-256 :
  `e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d`.
- Index Safetensors SHA-256 :
  `74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8`.
- Métadonnées du shard de layer 0 SHA-256 :
  `71213f444f9c15a4af6dbcbddc36f7611542884ebb3aadef54de650071357449`.
- Les huit noms de layer 0 sont dans le shard
  `model-00003-of-00048.safetensors`; le header local confirme les formes :
  `wq_a [1280,5120]`, `wq_b [32768,1280]`, `wkv [512,5120]`, `q_norm [1280]`,
  `kv_norm [512]`, avec les échelles F8_E8M0 correspondantes.
- Le chargeur exige les huit SHA-256 de payload du propriétaire appelant; aucun
  payload réel n’a été lu dans ce lot. Il refuse l’absence, le changement de shard,
  les formes ou digests incorrects et la modification du shard durant la lecture.

## Adaptation et limites

Réutilisés : `Fp8E4m3Matrix`, `fp8_activation_quantize_reference`, le cast BF16
RNE, `Stamp`, le lecteur indexé Safetensors et les refus fermés de VoltaFabric.
Le code adapte l’ordre de `Attention.forward` du donneur sans copier ses kernels.
Le GEMV accumule ses blocs en f64 puis retourne f32, casté BF16 aux frontières
linéaires. Cela ne prouve pas la parité bit à bit avec `fp8_gemm` CUDA/Torch.

Le résultat est explicitement pré-RoPE : il ne peut pas être passé directement au
cache d’attention R733, qui reçoit Q/KV après RoPE. RoPE, fenêtre/KV append,
attention sink, chemin KV compressé/indexer, projection `wo_a/wo_b`, HC-post,
composition d’une couche et génération ne sont pas raccordés ici. Le modèle et ses
poids n’ont pas été téléchargés; aucun GPU, réseau de production, R53 ou transport
inter-machine n’a été utilisé.

## Tests CPU

Sept tests ciblés passent debug et release : formes exactes depuis l’index/header
épinglés; formule scalaire indépendante E4M3/E8M0; résultat BF16 et identité; refus
d’une couche, tentative ou activation incorrecte; découpage du GEMV identique au
matvec direct; expansion des échelles 32×32; rejet de SHA malformés avant I/O.
Les 16 tests combinés des modules DeepSeek passent aussi en debug et release,
incluant la fenêtre/sink R733 et les tests HC/RMSNorm préexistants.
`cargo fmt --all -- --check` et `cargo clippy --locked --lib -- -D warnings` passent.

## Suite

R735 : vérifier les paramètres RoPE effectifs admis et relier Q/KV pré-RoPE à la
fenêtre R733 par une étape post-RoPE identifiée. Garder le KV compressé, `wo_a/wo_b`
et HC-post séparés jusqu’à leurs propres preuves. Aucun téléchargement ou GPU sans
admission fraîche.
