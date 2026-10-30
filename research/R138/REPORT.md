# R138 — différence entre déquantisation FP16 émulée et poids idéaux

Sur les trois plans réels de l'expert R132, le calcul CPU a comparé chaque poids idéal binaire64 `E2M1 × E4M3 × facteur global` avec l'émulation des étapes FP16 du kernel QPN2 construites dans R137. Les 3 145 728 valeurs ont été examinées, en conservant l'ordre des codes/scales et les facteurs propres à chaque projection. Torch CPU sert à organiser les tableaux ; aucun kernel CUDA n'est exécuté.

| Projection | Valeurs dont l'écart numérique n'est pas zéro | Écart absolu maximum | Valeurs idéales non nulles émulées à zéro |
|---|---:|---:|---:|
| down | 448 001 | 3,487663343548775e-05 | 0 |
| gate | 440 293 | 3,4875236451625824e-05 | 0 |
| up | 441 960 | 2,6155728846788406e-05 | 0 |

Ces chiffres décrivent **les poids déquantisés**, pas une erreur de sortie, une perte de qualité ou une différence observée sur V100. Le plan et les données sont liés par SHA, les 65 codes/scales possibles ayant été croisés entre deux émulations CPU dans R137. Les conventions de format et l'émulation sont issues du code donneur ; aucun calcul CUDA ou modèle complet n'est prouvé. Le shard complet n'a pas été haché. Une future tolérance de sortie doit être déclarée et justifiée à partir d'un oracle adapté et de la vraie accumulation, pas choisie après observation des sorties GPU.
