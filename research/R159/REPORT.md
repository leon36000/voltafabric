# R159 — sessions Rust comprimées avec expert échantillonné CPU

Les deux artefacts résidents Rust R158, contenant chacun une copie synthétique du même expert échantillonné R132, ont alimenté deux vrais workers `vf-rank-session` locaux sous la commande `session-run-compressed`. Le CLI et le worker sont identifiés par SHA dans le plan. Deux lots ont exécuté quatre requêtes : un seul transfert LOAD par rank, deux EXEC et BYE par worker, deux sorties normales code 0, sans processus de worker restant aux PID enregistrés. Les reçus contiennent les compteurs et les résultats complets. Aucune communication entre machines ni GPU.

Le collecteur Rust rapporte deux lots exacts selon son oracle dense interne. Une vérification Python scalaire distincte lit les octets R132, reconstruit l'entrée par lot selon la règle de variation du source et applique les coefficients 0,25 et 0,5 : les 4 096 valeurs concordent dans les tolérances R134 (abs et rel 1e-10), écart maximal 7,8063e-18. Les deux oracles partagent l'interprétation du format mais pas leur boucle de calcul. La session utilise le chemin CPU f64, pas les étapes FP16 du fournisseur R158.

Commande une fois, plan SHA préétabli, timeout 20 s par worker et 60 s externe : exit 0. Ce test qualifie la session comprimée Rust **CPU**, sans raccord au propriétaire Python R158. Il ne prouve ni CUDA, ni réseau de production, ni modèle entier, ni SHA complet du shard, ni performance.

Suite : raccorder explicitement le propriétaire Python au protocole Session, en testant sur CPU les formats d'activation/résultat, identités, reçus et poison ; conserver la distinction des contrats f64 et FP16 et ne pas rejouer les UNKNOWN.
