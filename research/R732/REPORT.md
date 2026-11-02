# R732 — Lier HC FFN au chemin route / experts partagés

État local : intégration CPU ciblée vérifiée; publication GitLab en attente de la fin du pipeline de reçu R731.

## Tranche intégrée

`DeepSeekV41IndexedMoeLayer::prepare_ffn_token_from_hc` prend les mixes HC FFN et le flux résiduel qui les a produits. Il refuse la branche Attention, une différence de couche ou d'identité complète, puis applique HC-pre et le `ffn_norm` résident une fois avant d'appeler la route déjà indexée. Le plan conserve le Stamp, l'attempt, l'activation BF16 et les coefficients de route du gate existant.

`DeepSeekV41HcMixes::apply_ffn_post` lie ensuite les mêmes mixes à l'après-sublayer FFN. Le chemin exercé est : HC-pre → ffn_norm/gate/top-6 → lecture sélective des experts MXFP4 → cinq Sessions CPU regroupées sur les rangs propriétaires → Collector complet → expert partagé E4M3/E8M0 → une addition et un arrondi BF16 → HC-post vers les quatre copies. Les coefficients du gate restent appliqués par le Collector une seule fois; aucun coefficient HC d'attention n'entre dans le chemin FFN.

## Preuve

Le test `completed_cpu_session_receipt_joins_shared_then_composes_hc_post_once` utilise le format, l'index et l'en-tête DeepSeek épinglés, avec tensors synthétiques aux offsets indiqués. Il charge réellement les deux branches HC synthétiques; la branche FFN alimente la route, et le test vérifie le refus de la branche Attention et d'un résidu d'un autre batch. Le reçu de route est produit par cinq Sessions CPU et six slots MXFP4, puis joint à l'expert partagé.

L'oracle de route compare l'exécution aux six experts résidents avec coefficient appliqué une seule fois. L'addition shared et le HC-post sont vérifiés par une formule distincte et un arrondi BF16 RNE indépendant. R731 vérifie déjà séparément projection/split/Sinkhorn par des sommes f64; R732 exerce les mêmes sorties HC dans le chemin de couche. Aucune parité Torch/TileLang n'est revendiquée.

## Résultat et limites

- Test ciblé R712, debug : 1/1, 96,91 s.
- Test ciblé R712, release : 1/1, 6,05 s.
- Clippy sur la cible et la bibliothèque avec `-D warnings`, rustfmt et un scan explicite des espaces blancs : succès.
- Aucun nouveau paquet. Le chemin reste CPU uniquement et les payloads HC, gate, experts et partagé sont synthétiques.

Cette tranche qualifie un sous-bloc FFN pour un token; elle ne qualifie ni l'attention/GDN, ni le modèle complet, ni une génération autorégressive. Il reste à connecter le flux résiduel issu de l'attention/GDN et à porter les logits vers un tokenizer/sampler vérifiable. Aucun GPU, transport inter-PC, ledger R53 de production, test de performance ou replay UNKNOWN n'a eu lieu.

## Réutilisation et provenance

Réutilisés : l'import indexé Safetensors R721/R731, gate et ffn_norm R724–R726, propriétaire MXFP4/Collector et Sessions CPU R712–R719, expert partagé et join atomique R727–R730, HC-pre/post R250/R730/R731. Le donneur DeepSeek-V4.1-Flash reste épinglé en révision MIT `314113f8a7dadb388215396ba5fbee2c610a04d6`; aucun code donneur copié et aucune qualité externe héritée n'est présumée.

Branche prévue : `r732-deepseek-hc-ffn-session-20261102`, issue du reçu R731 `f12139a2b46dcabeb5c7164f85ac33e3bc973f4f`. Le reçu R731 source (`ff771291bc7530504996215d46ecb7900bcbfa16`) a pipeline #2570 vert et lecture exacte 7/7. Son pipeline de reçu #2571 est encore en cours au moment de ce rapport.
