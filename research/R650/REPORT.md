# R650 — raccord du lanceur local au vrai pont binaire

## Vérification

Le harnais `cpu-harness/test_native_bridge_launcher.py` importe le lanceur Python existant `adapters/strata_native_bridge/voltafabric_launch.py` et appelle son vrai helper `_start_bridge`, puis `_finish_sessions`. Celui-ci démarre le binaire Rust compilé `vf-strata-native-bridge`; le coordinateur admet un index SC117, un fichier pack sparse de taille déclarée 50 292 326 400 octets, un seul expert synthétique de 2 176 000 octets et un vrai processus local `vf-rank-session`.

Une trame VFB3 unique avec run `r650-actual-bridge-launcher`, batch `9650001`, epoch `77`, couche `0`, attempt `30` a traversé le socket AF_UNIX du lanceur. Le test a vérifié l'identité complète et les dimensions de la réponse VFR3, une ligne de 2560 valeurs FP32 finies et non nulles, puis l'arrêt gracieux. Un seul reçu durable `RESERVED_UNKNOWN_NO_REPLAY` est conservé dans `receipts/`; ne jamais rejouer cette identité.

Le pack de test était sparse et temporaire; son allocation physique a été limitée par le harnais et le fichier a été supprimé à la fermeture de son répertoire temporaire. Le reçu durable n'est pas un registre R53 partagé ou de production.

## Commandes et portée

- `cargo build --locked --offline --features native-ggml-cpu --bin vf-strata-native-bridge` — PASS.
- `python3 -B research/R650/cpu-harness/test_native_bridge_launcher.py` avec `VF_R650_BRIDGE`, `VF_R650_RANK_SESSION`, `VF_R650_TMP_ROOT` et un nouveau `VF_R650_LEDGER_DIR` vide — PASS.
- Processus Rust terminés après le test; aucun worker VoltaFabric actif.

Cette tranche teste le helper de session du lanceur existant et le vrai pont Rust en un seul rang/couche. Elle ne lance pas la fonction publique complète `launch()` qui construit 48 sessions, ni le binaire donneur Strata, ni son callback de génération. Aucun checkpoint réel, modèle complet, GPU, RDMA, inter-PC, bail R53 de production ou performance n'est qualifié. La vérification de sortie ici ne constitue pas un oracle mathématique; R649 reste la comparaison du même fournisseur à `Resident::forward`.

Réutilisation démontrée : GGML CPU natif épinglé de R643 dans un résident VoltaFabric, avec le framing AF_UNIX/VFB3/VFR3 du pont et le lifecycle launcher de R642. Ce résultat ne revendique pas une intégration runtime de vLLM, SGLang, TensorRT-LLM, Maya ou DeepInfer, ni l'utilisation du moteur Strata donneur complet. La prochaine frontière est de raccorder l'appel réel du donneur au flux à 48 sessions sans ouvrir le checkpoint avant d'avoir un plan d'admission et une fixture sûre.
