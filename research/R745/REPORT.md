# R745 — réduction greedy du vocabulaire partitionné de DeepSeek V4.1

## Source examinée et décision

La révision officielle épinglée [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/inference/model.py) (`314113f8a7dadb388215396ba5fbee2c610a04d6`, [licence MIT épinglée](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/LICENSE)) découpe le vocabulaire de `ParallelHead` par rang. Chaque rang calcule ses logits FP32 locaux; le chemin de référence rassemble ensuite les lignes complètes avec `all_gather`, les concatène, puis applique le sampler. Le vocabulaire de la configuration épinglée est 129 280 tokens. La source n'est pas copiée; VoltaFabric adapte uniquement la frontière greedy de cette tête partitionnée.

R744 avait déjà établi la règle de départage: remplacement strict `>` et premier ID conservé. R745 réduit cette idée au rang propriétaire des logits: chaque partition calcule localement son meilleur couple `(score FP32, token_id u32)`; le collecteur compare ces couples sans recevoir les tranches complètes. À égalité, l'ID global le plus petit gagne, comme dans le parcours complet en ordre de token.

## Adaptation VoltaFabric

`src/deepseek_v41_head.rs` ajoute un plan explicite des plages de vocabulaire. L'admission exige une couverture contiguë, complète et sans rang répété de `[0, vocab_size)`, ainsi qu'un digest tokenizer. Le digest du plan lie vocabulaire, tokenizer, rangs et bornes.

`ShardedGreedyPlan::local_argmax` transforme une tranche locale en un petit enregistrement typé contenant l'identité run/modèle/plan/couche/batch/epoch, l'attempt, la position du token, les digests tokenizer/partition, le rang et au plus un couple `(u32, f32)`. Une partition entièrement masquée reste représentée sans candidat afin que son rang compte dans la couverture. `ShardedGreedyReducer::accept_partial` vérifie l'identité, l'attempt, la position, les digests, l'appartenance du token à la plage, l'unicité du rang et la finitude du score avant toute mutation. `finish` refuse toute couverture partielle ou tout vocabulaire entièrement masqué. Le candidat complet passe ensuite dans le sampler R744, qui vérifie encore l'identité et l'étape du décodeur avant d'ajouter le token.

Le transport sérialisé et son authentification ne sont pas implémentés. Le type privé empêche la mutation accidentelle dans le processus Rust; une frontière entre processus ou machines devra décoder, authentifier puis révalider les mêmes champs. Aucun socket ni worker n'est ajouté dans ce lot. La réduction offre une forme de message minuscule en comparaison d'une ligne de logits, mais aucun gain d'octets ni de latence n'a été mesuré ici.

Les routes d'experts, leurs coefficients, les propriétaires, les poids et les contrats QPN2/NVFP4, MXFP4 ou IQ3 ne sont pas touchés. Cette tranche ne calcule pas la projection LM head et ne charge pas de checkpoint.

## Vérification CPU

Le harnais reproductible `research/R745/cpu-harness` compile les véritables modules prompt, sampler, nouveau reducer et SHA-256 de VoltaFabric, avec `deepseek-recipe` 0.2.0, `deepseek-recipe-core`/`encoding` 0.2.0 et `tokenizers` 0.23.2 verrouillés.

Les tests couvrent le départage entre rangs hors ordre, les identités et tentatives, les digests de plan/tokenizer, doublons, plages invalides, candidats mal formés, NaN/+inf, masques `-inf`, refus de sortie partielle et passage au décodeur. Un oracle plat indépendant recalcule l'argmax sur 97 lignes complètes déterministes, puis le résultat partitionné doit être identique.

Les tests debug et release avec le feature stream actif passent chacun: 18 réussis, 0 échoué, 3 ignorés faute du tokenizer épinglé. Clippy termine avec succès; ses deux avertissements restants viennent de compteurs de test inutilisés du module SHA-256 partagé. `rustfmt --check` passe sur les modules et le harnais. Ce harnais isolé ne remplace pas le build complet du crate VoltaFabric.

## Limites et suite

R745 ne qualifie ni la sérialisation/authentification VFB3/VFR3, ni la génération complète, ni le producteur réel de logits, ni un checkpoint chargé, ni GPU, ni réseau inter-PC, ni R53 de production, ni une performance. L'étape suivante est une frontière de transport typée qui conserve le refus atomique, puis son raccord à une projection issue d'un checkpoint compatible avec les plages vérifiées. Aucune réservation GPU ou réseau n'est demandée par ce lot.

