# R607 — raccorder le plan scalaire à l'adaptateur de prompt

Le programme Rust R600 ajoute `native_math_backend` à son reçu de plan et produit un plan scalaire `r600-…`. L'adaptateur de prompt Strata rejetait à la fois ce champ et ce préfixe, y compris le champ explicite ajouté au plan vectoriel. Il ne pouvait donc pas utiliser le nouveau binaire pour préparer une génération.

L'adaptateur accepte les anciens reçus vectoriels ainsi que les nouveaux reçus explicitant leur backend. Il vérifie la correspondance entre backend demandé, backend reçu et domaine du plan ; le mode scalaire exige aussi un plan attendu avant de démarrer l'exécutable. Les plans mal formés et les incohérences restent refusés. Cette comparaison ne constitue ni une authentification ni une réservation matérielle.

Deux tests ajoutés ont d'abord reproduit le refus des nouveaux reçus, puis passent avec le correctif. La suite tokenizer sur PC1 a exécuté 48 tests, dont huit ignorés faute de données externes. Une préparation réelle sans worker ni génération utilise le checkpoint et le tokenizer SC117 présents sur PC1 ; son plan scalaire a été reconstruit indépendamment par `verify_plan.py`.

Une première soumission s'est arrêtée avant le moteur : l'environnement nettoyé avait été passé à `systemd-run` lui-même, qui ne trouvait plus le bus de session. La sortie était vide, aucun worker ni claim n'a été créé, et son marqueur reste conservé. Une nouvelle préparation sous identité distincte, avec l'environnement nettoyé appliqué uniquement au processus enfant, est suivie en R608 ; aucune identité soumise n'a été rejouée.
