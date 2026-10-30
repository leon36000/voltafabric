# R172 — EPLB DeepInfer face au routage VoltaFabric réel

Deux fonctions de placement du code DeepInfer épinglé (`_batch_greedy_indices` et `rebalance_experts_global`, Apache-2.0) ont été exécutées sur CPU, en les extrayant du source par AST sans installer DeepInfer. Quatre experts logiques, six slots physiques, deux rangs de capacité trois ; entrées synthétiques et SHA des sources et du vrai CLI Rust dans `receipt.json`. Aucun code donneur n'est intégré au moteur.

- Historique équilibré `[10,10,10,10]` : proposition DeepInfer `[0,2,3 | 1,3,2]`. Convertie en registre VoltaFabric, elle passe le vrai CLI `routing`, conserve les routes experts 0 et 1 et coefficients 0,25 et 0,75.
- Historique déséquilibré `[100,20,10,5]` : proposition `[0,0,3 | 0,1,2]`. Deux copies du même expert logique 0 sont sur le même rang ; le vrai routeur Rust refuse `INVALID_INPUT: invalid replica or rank/local collision`. C'est conforme à son contrat actuel : une paire (expert, rang) ne peut être répétée. Ce cas concret écarte l'intégration directe de la politique donneuse sans adaptation et contrôle d'admission.

Autre distinction : `src/routing.rs` choisit un propriétaire préféré par expert ou le plus petit rang ; il ne répartit pas dynamiquement chaque token entre copies comme peut le faire la correspondance logique/physique DeepInfer. Un plan accepté ne démontre donc pas un équilibrage actif. Capacités homogènes ici, pas de topologie ni coût réseau, migration ou état actif ; pas de GPU, modèle, performance, ni publication.

Suite : développer une couture bornée entre proposition de placement et registre existant : contraintes de capacité réelles, interdiction des doublons même rang, stabilité déterministe, préservation routes/coefficients et fail-closed ; traiter l'affectation aux répliques comme problème séparé et ne rien migrer avant validation d'époque et résidence.
