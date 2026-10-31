# R629 — génération locale autorégressive SC117

À partir du parent GitLab R627, génération CPU locale bornée avec le backend scalaire et les experts réels SC117 IQ3 distincts des formats MXFP4 et NVFP4. Prompt « Hi », 13 tokens de prompt, limite de huit nouveaux tokens : `Hello! How can I help you today` (9419, 0, 2500, 628, 353, 1438, 488, 3242). Arrêt à la limite, pas à EOS. Les 96 workers se sont fermés normalement.

Le préfixe commun avec R616 correspond pour les tokens et les empreintes des embeddings, sorties et logits ; ce contrôle ne constitue pas un oracle indépendant du modèle complet. R623 avait comparé indépendamment les 248 320 logits du premier token, avec les décodeurs GGML partagés. Aucune performance ni distribution réseau ou GPU n'est qualifiée par cet essai.

Le registre local neuf contient 672 batches et 1752 évictions : 2424 opérations `RESERVED_UNKNOWN_NO_REPLAY` préservées intégralement dans les fragments CLAIMS et jamais rejouées. Le registre partagé R53 et une lease coopérative réelle restent à réconcilier avant tout worker inter-machine ou GPU.

La première préparation R628 a refusé les liens symboliques avec ELOOP ; aucun calcul n'y a été soumis, aucun claim créé. R629 emploie des copies vérifiées des exécutables R627, une identité et un registre neufs. Les sources et dépendances sont épinglées par PLAN.json ; réutilisation des adaptations Strata et llama.cpp documentée en R626 et R627.
