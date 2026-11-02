# R698 — fragments DeepSeek-V4.1 réels dans deux Sessions CPU locales

## Résultat

Le chemin MXFP4 K32 natif de VoltaFabric relie maintenant les fragments d’experts épinglés à deux Sessions persistantes distinctes et au Collector FP32 prépondéré. Le test consomme les vrais fragments safetensors des experts 0 et 383 de la couche 0 et compare la somme du Collector aux vecteurs-oracle BF16 épinglés.

Le callback public llama_decode de R697 exerce Qwen2MoE. Pour DeepSeek-V4.1, R698 emprunte le propriétaire MXFP4 K32 et la Session VoltaFabric existante au lieu de forcer une architecture différente dans llama.cpp.

## Données et licence

- deepseek-ai/DeepSeek-V4.1-Flash, révision 314113f8a7dadb388215396ba5fbee2c610a04d6, licence MIT selon la provenance locale épinglée.
- En-têtes Safetensors inspectés : expert 0 (18 801 312 octets), experts 96 et 192 (37 602 576 octets), expert 383 (18 801 288 octets). Cela couvre quatre experts d’une couche, pas un checkpoint complet.
- Les plans w1/w2/w3 sont des codes I8 empaquetés E2M1 avec échelles F8_E8M0 par K=32. Ce format reste MXFP4 K32, distinct de NVFP4 K16 et d’IQ3.
- La provenance A58 indique des extractions par plages d’octets d’un shard officiel de 7,39 GB ; le checkpoint complet de 510 GB n’a pas été téléchargé. Aucun poids d’embedding, routeur, attention, normalisation, KV ni tokenizer complet n’est présent dans cet essai.

## Réutilisation et adaptation

- Réutilisés depuis VoltaFabric : Mxfp4ExpertShard pour le SHA-256, l’admission bornée et les plans immuables ; le worker CPU vf-rank-session et le protocole persistant MXFP4 ; RoutingProblem et le Collector FP32 prépondéré.
- Le test mappe l’expert global 0 au rang/local 0/17 et l’expert 383 au rang/local 1/4, puis envoie un token BF16 de 5 120 valeurs à deux processus persistants.
- Les coefficients 0,375 proviennent des vecteurs-oracle individuels. Cette route de test n’est pas la sortie du gate et sa somme 0,75 ne qualifie pas le routeur. Chaque coefficient est appliqué une seule fois par le producteur R9 ; le Collector additionne les résultats déjà prépondérés.
- Aucun code externe n’a été copié dans cette vague. Les sources DeepSeek servent de référence épinglée ; le propriétaire, la Session et le Collector restent ceux de VoltaFabric.

## Vérification

Commande ciblée cargo test --offline --locked --test r698_deepseek_real_mxfp4_session -- --nocapture : 1 réussi, 0 échec, deux Sessions CPU, deux chargements exacts, une route à deux rangs et une sortie de 5 120 valeurs. L’écart absolu maximal est inférieur ou égal à 5e-4, selon l’assertion contre la somme des deux sorties d’oracle. Durée : 10,14 s dans le checkout local r367-strata-exact-ingress ; le pipeline GitLab n’est pas cette preuve.

Une prévalidation s’est arrêtée avant la création de toute Session : la constante SHA attendue de l’expert 383 avait un caractère manquant. Après correction, le test concluant a reçu un run_id et un plan_id distincts. Aucun LOAD/EXEC partiel ni claim inconnu n’a été créé par la prévalidation.

## Limites

Ce résultat qualifie localement un fragment de couche avec vrais experts, pas une génération DeepSeek complète. Le gate est simulé par une route de test, le checkpoint est incomplet, le batch contient un seul token et les deux workers sont sur le même hôte. Aucune GPU, V100, RTX 5080, Spark, réseau inter-PC/RDMA, registre R53 de production, performance ou tokenizer n’a été exercé. Aucun UNKNOWN antérieur n’a été rejoué.

## Suite

R699 reliera le conseil mémoire par rang R694–R697 au dispatch natif MXFP4 de DeepSeek. Le gate restera seule autorité de route ; le chargement et la résidence devront refuser atomiquement une admission incomplète. Dériver l’enveloppe des experts du dispatch courant à partir des poids résidents déjà qualifiés, CPU seulement. Aucun téléchargement du checkpoint, GPU, réseau de production, nettoyage ni relance automatique.
