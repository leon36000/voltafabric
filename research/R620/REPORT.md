# R620 — invalider une session après refus du chargement initial

Les chargements initiaux d’experts natifs vectoriels et scalaires laissaient la session marquée utilisable après une réponse `Refused`, alors que le serveur peut terminer la session après ce refus. Les deux méthodes la rendent maintenant inutilisable avant de retourner l’erreur. Les refus locaux avant envoi ne changent pas ce comportement et restent sans écriture. Les ajouts d’experts invalidaient déjà la session en cas de refus.

Un pair synthétique lit intégralement le corps envoyé, refuse puis reste ouvert. Le nouveau test échouait sur la session encore utilisable avant correction ; il passe ensuite pour les deux backends et vérifie qu’une seconde tentative ne transmet aucun octet. Les 150 tests de bibliothèque passent en debug et release ; 10 tests à activation explicite restent ignorés. Clippy ciblé passe avec l’avertissement historique dead_code autorisé et rustfmt passe.

Cette correction ne qualifie pas le réseau, les GPU, le registre R53 ou une génération distribuée. Les résultats incertains restent UNKNOWN et ne sont jamais rejoués. Les tests utilisent des données synthétiques et aucun worker réel.
