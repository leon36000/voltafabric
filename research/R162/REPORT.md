# R162 — garde de trames expérimentale pour le pont CPU

Un petit module privé implémente la lecture bornée de l'en-tête JSON et du corps, le décodage d'entrées FP16 finies et l'écriture de trame. Il refuse notamment clés JSON dupliquées, troncation, tailles hors limites, indices de token dupliqués ou booléens et FP16 non fini. La limite d'en-tête de 64 Kio est volontairement plus stricte que le protocole général : ce module n'est pas un worker de production et ne prétend pas accepter toutes les requêtes Session valides.

Le premier lancement des tests a échoué car le module n'existait pas. Après implémentation, cinq tests de garde et trois tests supplémentaires passent. Ces derniers reconstruisent l'en-tête et les activations du vrai EXEC FP16 R160 et retrouvent leurs SHA, testent une écriture fractionnée et le refus de clés dupliquées imbriquées. Commandes : `python3 -B test_wire.py` et `python3 -B test_real_capture.py`, exit 0. Aucun ancien runner ni GPU n'a été relancé.

Ce lot prépare une pièce de parsing, pas une Session connectée au propriétaire : il ne gère ni la machine d'états HELLO/LOAD/EXEC/BYE, ni l'identité complète d'un résultat, ni poison/UNKNOWN, ni les capacités négociées. Il reste local et ne modifie pas les 370 sources canoniques. Le module et ses tests référencent les preuves R160 locales ; ils ne constituent pas une bibliothèque portable livrée ou admise en production.

Suite : réaliser une couture CPU minimale à partir de ces fonctions et du vrai `PersistentQPN2Owner`, puis l'exercer depuis une vraie Session Rust, avec contrôle indépendant des erreurs et sans appliquer deux fois les coefficients.
