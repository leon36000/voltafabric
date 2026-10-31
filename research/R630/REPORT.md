# R630 — réponse locale SC117 terminée par EOS

Avec les exécutables et les experts réels de R627, une nouvelle exécution CPU locale bornée du prompt « Hi » génère `Hello! How can I help you today?`, puis le token EOS 248046, avant la limite de 16 tokens. Dix tokens ont été générés en comptant EOS ; les 96 workers sont fermés et le processus s’est terminé normalement.

Les tokens, entrées et empreintes des sorties, embeddings et logits du préfixe de R629 correspondent exactement. Ceci n'est pas un oracle indépendant du modèle complet ; R623 comparait indépendamment les logits du premier token, avec décodeurs GGML partagés. Le résultat ne qualifie ni GPU ni worker inter-machine.

Les 768 réservations de batch et 2006 réservations d’éviction demeurent `RESERVED_UNKNOWN_NO_REPLAY`. Les 2774 claims exacts et la réponse sont conservés en fragments vérifiables, avec leurs hashes ; aucune identité consommée n'a été rejouée. Le registre partagé R53 d'origine et la lease coopérative restent à réconcilier avant exécution distribuée.

Les adaptations Strata et llama.cpp, leurs licences et leurs limites figurent dans les preuves R626 et R627 ; PLAN.json lie les exécutables, les sources et les paramètres à cette nouvelle identité.
