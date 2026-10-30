# R133 — admission CPU des formes et échelles d'un expert réel

L'adaptateur Skinny VoltaFabric épinglé et les trois fichiers d'échantillon R132 ont été transmis à l'environnement Python 3.12 / Torch CPU existant de PC1, puis vérifiés par SHA-256 avant utilisation. Les trois matrices codes et les trois matrices scales ont été assemblées selon les formes de l'en-tête R131, en ordre down, gate et up ; les facteurs F32 gate/up restent distincts dans la structure, même si leurs valeurs dans cet expert réel sont égales.

Le vrai `prepack_experts` de l'adaptateur a accepté ces formes et son contrôle des échelles SM70 a passé. Les deux appels de prépacking ont toutefois utilisé **une fonction CPU de test injectée qui clone les données sans permutation** ; le résultat n'est donc ni un prépacking QPN2 réel ni un calcul GPU. La structure propriétaire créée a les dimensions `(1, 1024, 2048, 2048, 512)` et les facteurs globaux attendus aux emplacements gate/up/down.

Pour les 65 codes E4M3 distincts observés dans les 196 608 octets d'échelles, une formule Python indépendante et la conversion `torch.float8_e4m3fn` CPU donnent des valeurs exactement égales. Cette comparaison porte sur les échelles décodées uniquement ; elle ne vérifie pas le décodage E2M1, un produit matriciel, l'ordre d'accumulation, les coefficients d'experts ou les sorties du modèle. L'échantillon provient de plages TLS et le SHA LFS du shard complet reste non vérifié. Aucun GPU, moteur 1Cat ou modèle complet n'a été exécuté.

Prochaines portes : oracle indépendant des codes E2M1 et facteurs avec exemples et sortie d'expert, vrai prépacking natif puis comparaison numérique sous les ressources réellement réservées ; environnement et dépendances du moteur toujours incomplets.
