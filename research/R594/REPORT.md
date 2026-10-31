# R594 — frontière scalaire CPU vers le Collector Rust

La référence R592/R593, adaptée des décodeurs et quantificateurs GGML épinglés par R593, est exposée à Rust via une ABI C strictement optionnelle (`native-scalar-reference`). Elle renvoie une ligne f64 non pondérée. La frontière vérifie l'index, la tranche et le SHA-256 des octets par rapport au `Binding` fourni ; ce dernier doit lui-même provenir d'une source authentifiée. L'appelant reste responsable des identités complètes et de l'admission. Elle ne remplace pas le worker résident.

Un test synthétique non nul vérifie le rejet d'identités erronées, de données corrompues et d'une activation NaN, puis convertit f64 vers f32 et FP16 avant l'admission par le Collector Rust, avec un coefficient appliqué une fois.

Un second diagnostic CPU local utilise trois experts SC117 réels de la couche 2 (75, 12 et 255) et l'activation de R574 ; les sept autres lignes sont nulles et les reçus sont synthétiques. Les 3 × 2 560 valeurs scalaires correspondent bit à bit aux captures R590. Après conversion f32 puis FP16 et réduction par le Collector Rust, un oracle Python indépendant reproduit exactement les 2 560 f64, SHA-256 `535c8bcc17bcad4ac9e01a58f00f9ee2e4fe880dd84dd6804825930120aed08b`. Le plan et les hashes des entrées sont locaux dans `.runs/vf-r594-work/plan.json`. Les captures et poids ne sont pas publiés.

Le contrôle porte sur ces entrées ; il ne prouve ni l'équivalence générale des noyaux vectoriels PC1/PC6, ni un réseau de production, une exécution GPU ou une génération complète distribuée. Aucune mesure de performance. Le contrat partagé R53 doit être réconcilié avant toute tentative matérielle ou inter-PC.

Les tests ciblés passent en debug et release (2/2 chacun), y compris les sept couples de formats observés avec des blocs synthétiques nuls. Le test non nul du passage au Collector porte sur IQ2_S/IQ4_NL. Clippy de la bibliothèque passe avec l'avertissement historique `dead_code` désactivé ; le contrôle de la bibliothèque avec `native-ggml-cpu` seul réussit sans bibliothèque scalaire. Le formatage Rust ciblé passe.
