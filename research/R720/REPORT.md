# R720 — conseil de rang principal depuis des mesures de liaison importées

## Résultat

R720 ajoute `planner::recommend_primary_rank_advisory`. Pour un rang source local et une liste de rangs accélérateurs, l’API refuse une topologie synthétique, un rang inconnu ou dupliqué, un lien hôte→accélérateur non mesuré, une ambiguïté de liens actifs et un candidat sur un autre hôte. Elle choisit le lien mesuré le plus rapide; les candidats dans la bande de tolérance explicitement fournie par l’appelant sont départagés par mémoire non réservée, puis par rang. Elle renvoie les références de mesure utilisées.

Cette adaptation transforme en entrée typée et testable une idée de placement de DSX. Elle reste consultative : aucun expert, propriétaire, route, coefficient, registre, chargement ou allocation n’est modifié. Le plan d’experts existant reste la décision séparée; cette fonction ne revendique ni débit mesuré ni qualification d’un GPU. Le détenteur d’appel doit fournir une source de mesure réelle, puisque l’API ne vérifie pas le contenu du reçu.

## Donneur inspecté et décisions d’adaptation

- Dépôt public `lxe/dsx`, tête `main` observée le 2 novembre 2026 : `aa2dc5c6b12c24720c55a63e1da7c8ae31a18471`.
- Licence racine : MIT, Copyright © 2026 Aleksey Smolenchuk. SHA-256 `LICENSE`: `21606d2231fc37d5d042843402e7a51ccfcf1b4eb0f465363248c37e012a3a54`.
- L’aide de placement du donneur recommande le GPU principal selon la bande passante hôte→GPU, avec la mémoire comme départage pour les mesures proches; elle garde dense/KV sur ce GPU et les poids experts sur les GPU auxiliaires. Références épinglées : [`README.md`](https://github.com/lxe/dsx/blob/aa2dc5c6b12c24720c55a63e1da7c8ae31a18471/README.md), [`docs/placement.md`](https://github.com/lxe/dsx/blob/aa2dc5c6b12c24720c55a63e1da7c8ae31a18471/docs/placement.md). Les seuils propres à DSX ne sont pas repris comme défaut : VoltaFabric exige un seuil fourni explicitement.
- Le README annonce un port SM70 sous CUDA 12.9 et des résultats DSX sur 2×V100 PCIe 32 Go. Ces nombres restent des annonces du donneur, non reproduites ici et non comparées au moteur VoltaFabric.
- L’inspection CUDA nuance la formule « chaque kernel SM80 a une version SM70 » : [`sm70.cuh`](https://github.com/lxe/dsx/blob/aa2dc5c6b12c24720c55a63e1da7c8ae31a18471/nvidia/src/kernels/sm70.cuh) indique que son helper n’a pas d’appelant. [`expert_fast.cuh`](https://github.com/lxe/dsx/blob/aa2dc5c6b12c24720c55a63e1da7c8ae31a18471/nvidia/src/kernels/expert_fast.cuh) documente une voie DP4A pour SM70 en regard des instructions IMMA; [`t16.cuh`](https://github.com/lxe/dsx/blob/aa2dc5c6b12c24720c55a63e1da7c8ae31a18471/nvidia/src/kernels/t16.cuh) distingue les kernels SM80 à copie asynchrone d’un chemin simple SM70. Aucun de ces kernels n’est copié.
- Le modèle livré par DSX emploie T16 pour les experts, Q8_0 pour les poids denses et IQ4_NL pour Engram. Ces formats sont distincts de MXFP4 K32, NVFP4 K16 et IQ3 VoltaFabric; aucun quantizer, poids ou décodeur n’est réutilisé.

L’arXiv scout GPT‑6 Luna a aussi signalé **Director** (arXiv:2607.08782), qui combine prédiction des prochains experts, migration en ligne et optimisation avec contraintes de capacité. Son évaluation utilise 4×8 RTX 4090 ou 4×8 H200, PCIe 4.0/NVLink et InfiniBand, avec Python/Colossal‑AI/SciPy/CUDA 12.4. Cette piste n’est pas importée : elle prévoit des migrations d’experts selon une route prédite, tandis que VoltaFabric doit conserver la route et le coefficient fournis par le gate et refuser une exécution partielle. Voir [résumé arXiv](https://arxiv.org/abs/2607.08782) et [texte intégral](https://arxiv.org/html/2607.08782v1). Aucun de ces bancs ne qualifie notre V100 ou liaison inter-PC.

## Vérification locale

- `cargo test --locked --offline --test r720_primary_rank_advice` : 4/4 réussis.
- `cargo test --release --locked --offline --test r720_primary_rank_advice` : 4/4 réussis.
- `cargo clippy --locked --offline --test r720_primary_rank_advice -- -D warnings` : réussi.
- `cargo fmt --check` et `git diff --check` : réussis.
- Les fixtures importées du test sont synthétiques et signalées comme telles; elles vérifient les règles d’admission, l’ordre, le départage mémoire et le refus des profils non mesurés. Aucun appareil matériel n’a été sondé.

## Limites et suite

Aucune mesure hôte→V100 fraîche n’a été importée par R720; le conseil n’a donc pas choisi de rang pour PC1, PC4 ou PC6. R720 ne qualifie ni génération complète, ni checkpoint réel, ni GPU, ni R53, ni transport de production, ni amélioration de performance. Aucun UNKNOWN n’a été relancé. Le drapeau de relance automatique reste `STOPPED_BY_OWNER`.

Prochaine étape : examiner les frontières d’adaptation et licences de vLLM/SGLang pour le chemin V100 et la résidence des experts, puis intégrer une tranche seulement si elle apporte un raccord CPU testable à la génération VoltaFabric. Pas de téléchargement de poids, GPU ou migration de route.
