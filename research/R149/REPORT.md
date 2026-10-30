# R149 — dtype racine du checkpoint et convertisseur 1Cat

Une requête bornée a récupéré le `config.json` épinglé du Qwen R129, omis en entier des artefacts précédents, puis a vérifié son SHA-256 contre R129. Les champs racine `dtype` et `text_config.dtype` valent tous deux `bfloat16`. L'empreinte prouve l'égalité au fichier reçu en R129 ; elle ne valide pas les poids du checkpoint.

Le convertisseur 1Cat exact, membre du wheel identifié par SHA, a été extrait et son corps exécuté par AST avec des objets de configuration simulés. Quatre tests réussis vérifient la priorité du dtype racine, puis les replis texte et vision. La simulation d'un dtype racine BF16 le retourne sans lire les métadonnées des poids. Combiné aux essais R148, si Transformers construit effectivement ce dtype Torch depuis le JSON, si `auto` est utilisé et si la plateforme indique SM70, le chemin de sélection aboutit à FP16. Cette chaîne reste conditionnelle : le parseur Transformers et le runtime réel n'ont pas été exécutés. Une option de lancement explicite peut changer ce résultat.

Plan externe : `plan.json` ; une requête, exit 0, 58 110 octets, sans poids ni GPU ; `python3 -B test_converter.py` : 4 tests, exit 0. Le JSON intégral et le source tiers restent sous `.runs/r149-config-dtype`. Aucun moteur, modèle complet ou GPU n'est qualifié.

Suite : tester le chargement de configuration réel dans un environnement CPU isolé disposant de Transformers, puis établir un chemin MoE avec collecteur et données de l'expert R132. Ne pas masquer le conflit déclaré 1Cat/TileLang.
