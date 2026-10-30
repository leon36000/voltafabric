# R152 — registre Qwen 1Cat avec AutoConfig CPU

Le wheel 1Cat 1.2.2 épinglé associe `qwen3_5_moe` à `Qwen3_5MoeConfig` dans son registre et associe cette classe au module de configuration Qwen dans son index. Les octets du wheel et des trois membres ont été hachés ; le wheel correspond à R125.

Sur PC1 CPU, le test a exécuté la classe Qwen exacte du wheel ainsi que la classe `LazyConfigDict` et le dictionnaire de registre extraits par AST du code exact. Une hiérarchie de modules vLLM simulée fournit cette classe au registre ; l'enregistrement dans le vrai `transformers.AutoConfig` est effectué manuellement selon la séquence observée dans le source 1Cat. Ensuite `AutoConfig.from_pretrained(..., local_files_only=True, trust_remote_code=False)` a chargé le config R149 comme `Qwen3_5MoeConfig` et sa classe texte, dtype BF16, 30 couches linéaires. Commande SSH et probe : exit 0, journal d'erreur vide.

Ce résultat établit que la couture registre/classe/AutoConfig fonctionne avec ces simulations CPU et explique l'échec de l'AutoConfig standard R151. Il ne montre pas l'import ni l'exécution de la fonction complète de chargement vLLM, la sélection effective du dtype sur V100, l'installation intégrale des dépendances 1Cat ou le chargement des poids. Aucun GPU ni modèle complet n'a été exécuté. Le conflit TileLang/TVM reste ouvert.

Suite : tester sur CPU la frontière MoE vers le collecteur Rust à partir des octets R132, avec coefficients et identités explicites ; ou qualifier le chargement réel 1Cat seulement après résolution de ses dépendances et isolation adaptée.
