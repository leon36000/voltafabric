# R709 — plan borné des plages de poids MXFP4 DeepSeek

## Changement

`src/deepseek_v41.rs` expose maintenant `plan_pinned_mxfp4_expert_payloads`. À partir de l'index Safetensors épinglé, du JSON d'en-tête d'un shard, de leurs identités/hashes attendus et d'une limite d'octets fournie par l'appelant, le plan résout les six noms `w1/w2/w3 × weight/scale` par expert vers le shard et les plages absolues exactes. Il refuse les identifiants dupliqués ou hors domaine, les révisions/index non épinglés, les noms de shards ambigus, les headers/hash incohérents, les types/formes incorrects, les plages hors fichier ou chevauchantes et tout dépassement de la borne donnée.

Le plan ne lit pas les plages de poids. Il apporte la correspondance tensor → shard → offsets dont le lecteur sélectif et le futur propriétaire résident ont besoin, sans ouvrir le checkpoint complet ni traiter les métadonnées comme une preuve de résidence.

## Vérification sur le modèle épinglé

Le test reprend le top‑6 synthétique `[0, 383, 382, 381, 380, 379]`, l'index officiel SHA‑256 `74b0686a…b98fa8` et l'en-tête officiel du shard `model-00003-of-00048.safetensors`, SHA‑256 `71213f44…57449`. Les 36 tenseurs attendus (six par expert) appartiennent au shard indiqué par l'index. L'en-tête mesure 258 136 octets; les poids commencent à l'offset fichier 258 144. Pour ces formes MXFP4 K32, les plages sélectionnées totalisent 112 803 840 octets de payload dans un shard de 7 389 759 032 octets.

- Test ciblé debug : 2 réussis.
- Test ciblé release : 2 réussis.
- `cargo clippy --offline --locked --lib -- -D warnings` : réussi.
- `rustfmt --edition 2021 --check src/deepseek_v41.rs` et `git diff --check` : réussis.
- Le blob de source local correspond au blob GitLab R709 (`1c1a64f5…2997c8`).

## Réutilisation et limites

La tranche réutilise la grammaire de noms MXFP4 déjà contrôlée par VoltaFabric, les métadonnées Safetensors et les preuves d'en-tête/index officielles déjà épinglées. Elle ne copie aucun code d'un moteur tiers. DeepSeek‑V4.1‑Flash reste sous MIT à la révision `314113f8a7dadb388215396ba5fbee2c610a04d6`; les sources du donneur sont des références de format, pas une garantie de qualité du moteur.

Les identifiants sélectionnés sont une fixture de route, pas une génération mesurée. L'index et l'en-tête ne prouvent ni présence ni hash des octets de poids. Les artefacts locaux suivis ne couvrent toujours pas les experts 379–382; l'admission complète des six reste refusée. Aucun shard complet ni checkpoint n'a été téléchargé. Pas de Session/worker lancé, GPU, transport inter‑PC, R53 de production, UNKNOWN rejoué ou sortie de modèle complète.

Pipeline GitLab #2456 terminé avec succès. Ses quatre jobs réussis couvrent les scans/fallback configurés, pas une compilation ni les tests Rust locaux. Relance automatique arrêtée; aucune fusion vers `main`.

## Traçabilité

- Branche privée : `r709-deepseek-expert-range-plan-20261102`.
- Parent : `ac541f796253351da0f75289fd95d6641f5c8c91` (handoff R708).
- Commit source : `a68ee0e01931af18084ad52b78af8d2ebbd56701`.
- Fichier : `src/deepseek_v41.rs`.
- SHA‑256 de la source : `c889a6676a725a52418abff737837773c52535bbbfcf079e939fa9903fa22c51`.
