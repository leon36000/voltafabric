# R154 — adaptateur Skinny vers collecteur Rust avec sortie CPU pré-calculée

Le proxy expert scalaire R153 a été recalculé à partir des octets R132 vérifiés ; les 2 048 valeurs FP16 sont toutes non nulles et leur SHA est identique à R153. Le vrai CLI Rust a produit deux RankDispatch pour deux identités synthétiques et des coefficients 0,25 et 0,5.

Sur PC1, le code exact `adapters/skinny_moe.py` identifié par SHA a exécuté `moe_apply_dispatch_local` pour chaque RankDispatch sous Torch CPU. Un fournisseur simulé retourne des zéros pour gate/up puis injecte la ligne pré-calculée à la sortie down ; les plans QPN2 synthétiques ne servent pas au calcul. Le test vérifie deux appels par rank, les identités complètes et deux refus d'epoch périmé avant appel fournisseur. Les sorties brutes non pondérées ont été renvoyées à PC6 puis fournies au vrai collecteur Rust en ordre inverse : 2 048 valeurs correspondent exactement à `0,75 × ligne FP16`, exit 0.

Il s'agit d'une vérification de la couture de routage, des identités et de la pondération, avec vrai adaptateur et vrai collecteur mais calcul expert injecté. Elle ne prouve ni calcul des matrices par l'adaptateur, ni kernel GPU, ni modèle ou transport de production. Les deux experts représentent des copies de test du même échantillon et non deux experts réels ; le SHA du shard complet reste non vérifié. Scripts, plan, résultats et reçus sont joints ; la ligne binaire reste dans `.runs/r154-adapter-sample-collector`.

Suite : remplacer le fournisseur qui injecte la sortie par un fournisseur CPU calculant effectivement les projections à partir des trois plans R132, avec un oracle indépendant et un contrat d'arrondis explicite avant tout GPU.
