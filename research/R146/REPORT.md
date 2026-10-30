# R146 — chemin d'attention du candidat Qwen

Les champs de configuration extraits en R129 pour `nvidia/Qwen3.6-35B-A3B-NVFP4` à la révision `1355db6a052410cfd62085d94b58866fd0f2c3c5` indiquent l'architecture `Qwen3_5MoeForConditionalGeneration` et 40 types de couches : 30 `linear_attention` et 10 `full_attention`. La couche 0 est de type `linear_attention`. Le contrôle hors ligne examine les champs conservés en R129, pas une nouvelle vérification du config.json complet.

Le membre Python `qwen3_5.py` extrait du wheel 1Cat existant sur PC1 a été relié au SHA du wheel et contrôlé par AST. La classe conditionnelle construit le langage `Qwen3_5MoeForCausalLM` ; le décodeur sélectionne `Qwen3_5GatedDeltaNet` pour une couche de type `linear_attention`, classe dérivée de `QwenGatedDeltaNetAttention`. R145 a identifié une branche de préremplissage SM70 qui importe paresseusement TileLang dans ce dernier module. Cette chaîne source rend cette dépendance pertinente pour le candidat existant, mais ne prouve pas que la branche SM70 particulière soit exécutée avec une configuration réelle.

La commande SSH a extrait le membre puis une seconde lecture a vérifié l'empreinte du wheel et celle du membre ; les deux sorties 0. `python3 -B check.py` : sortie 0. Source wheel conservé dans `.runs/r146-qwen-route`, empreintes dans les reçus ; pas d'import du moteur, modèle, GPU ou mesure. La validation du SHA complet des shards reste absente.

Prochaine action : tracer les conditions d'appel du préremplissage GDN pour ce modèle, avec les données locales déjà acquises, ou tester la frontière MoE indépendante sur CPU ; le conflit 1Cat/TileLang reste explicitement ouvert.
