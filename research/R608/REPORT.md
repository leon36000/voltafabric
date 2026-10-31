# R608 — génération complète locale avec experts scalaires

Une exécution neuve sur PC1 a traité un prompt de 13 tokens sur les 48 couches du checkpoint SC117 observé, avec deux workers CPU locaux par couche et le backend scalaire GGML. Elle a généré les tokens `[9419, 0]`, décodés `Hello!`, puis s'est arrêtée à la limite de deux tokens demandés. Les 96 workers ont confirmé leur fermeture. Ce résultat est une courte génération locale, pas une qualification GPU ou inter-PC.

Le vérificateur contrôle l'identité de plan et du run, le backend, les tokens du prompt, les admissions et évictions, les compteurs de fermeture et les 1 529 claims conservés : 384 batches et 1 145 évictions. Chacun est `RESERVED_UNKNOWN_NO_REPLAY` et son nom est recalculé à partir de son identité ; aucun claim antérieur n'a été rejoué. Le résultat brut et les valeurs dérivées du modèle restent sur PC1 et dans `.runs`, tandis que les claims et les métadonnées nécessaires à la reprise sont conservés ici.

Les tokens générés et les hashes d'embeddings correspondent à l'essai vectoriel R588 ; les hashes de sorties de couche et de logits diffèrent. Cette comparaison ne démontre pas l'équivalence numérique des deux chemins ni une qualité de génération générale. L'essai utilise des composants GGML partagés avec les références antérieures et n'est pas un oracle complet indépendant du modèle. Aucune mesure de performance comparative n'est revendiquée.

Le plan `r600-24498628aa5e7846f93dce55daeb38042481f8093c023c9ddb57f8af34babcc0` a été recalculé séparément par `verify_plan.py`. Le lanceur limite la portée à 13 Gio de mémoire, zéro swap, deux cœurs et 320 tâches. Son marqueur exclusif et les reçus attestent d'une seule soumission R608 ; l'échec préalable sans worker R607 reste documenté séparément. Aucun GPU, worker réseau, registre R53 ou lease partagée n'a été utilisé.

À poursuivre : qualification numérique des sorties et des seuils de quantification, vérification sur plusieurs prompts et raccordement inter-PC uniquement après réconciliation R53 et admission matérielle. Le checkpoint observé n'est pas authentifié par cette expérience.
