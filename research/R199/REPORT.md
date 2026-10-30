# R199 — oracle CPU indépendant des poids NVFP4 d'origine

`nvfp4_oracle.py` lit directement les codes E2M1, les échelles E4M3 par 16 et les facteurs scalaires du fichier safetensors ModelOpt. Il décode les matrices dans leur ordre d'origine, sans utiliser les permutations QPN2, le propriétaire ou le collecteur Rust, puis effectue les projections et l'activation avec des intermédiaires FP16. Il pourra être appliqué aux huit experts distincts après vérification et extraction du shard R188.

Sur l'expert complet 0 déjà présent, dont les codes correspondent à l'échantillon R132, le résultat est comparé à la capture indépendante du propriétaire R158 : 510 valeurs sur 2 048 sont identiques en FP16 et l'écart absolu maximal est `1.239776611328125e-05`. Le test utilise une borne illustrative de `2e-5` pour ce cas et confirme aussi le refus d'entrée non finie. Deux tests passent, exit 0. La différence est compatible avec les chemins et arrondis différents ; sa cause précise reste à confirmer ; cette borne ne qualifie ni l'ensemble du modèle ni une qualité GPU.

Aucun nouveau poids, GPU, génération ou performance ; le fixture est un seul expert authentique déjà présent et le SHA complet du shard R188 n'est pas encore vérifié.
