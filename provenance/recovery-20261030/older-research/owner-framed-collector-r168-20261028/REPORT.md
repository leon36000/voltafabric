# R168 — cadrage hors ligne des sorties Owner vers Collector

Les deux lignes FP16 capturées du vrai owner CPU R158 sont vérifiées par SHA, reconstruites sous la classe `OwnerOutput` réelle extraite par AST, puis emballées avec `as_raw_output_bytes` dans des trames `completion_batch_fp16_raw` utilisant les helpers R162. Les métadonnées de routage et le stamp proviennent de l'ancien problème R158, non d'une Session active. Chaque trame est relue avec les helpers R162 et les octets reçus sont identiques. Une identité locale étrangère est refusée pour chaque owner.

Les deux lignes décodées ont ensuite été données au vrai CLI Rust `collect-fp16`, rang 9 avant rang 2 : deux lignes acceptées et 2048 valeurs égales à `0,75 * ligne` ; exit 0. SHA sources, trames, captures et exécution dans `receipt.json`; script reproductible `run.py`, Python environnement CPU local, sans GPU. Le test démontre un raccord de représentation hors ligne ; il ne prouve pas que Rust ait lu ces trames, ni une communication en direct entre Session et owner, ni un modèle complet.

Suite : implémenter le worker de pont réel (HELLO, LOAD, EXEC, SHUTDOWN), importer l'owner sans altérer les installations et relier ses trames à la véritable Session ; isoler les échecs avant/après admission et ne jamais rejouer UNKNOWN.
