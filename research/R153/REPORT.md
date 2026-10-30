# R153 — échantillon expert vers collecteur Rust sur CPU

Un nouveau cas CPU utilise les trois plans de poids échantillonnés R132, vérifiés par SHA. Le calcul expert est un substitut scalaire : déquantisation FP16 selon la formule R139, produits et activation en binaire64, puis sortie arrondie FP16. Il n'exécute ni l'adaptateur Skinny ni un kernel natif. L'échantillon ne vérifie pas le SHA complet du shard.

Pour exercer deux routes, le test crée explicitement deux experts synthétiques avec les mêmes poids d'un seul expert réel et des identités distinctes, affectés à deux ranks. Le vrai CLI Rust R61 crée le routage, puis reçoit les deux résultats non pondérés dans l'ordre inverse, avec les coefficients 0,25 et 0,5. Les 2 048 valeurs retournées correspondent exactement au calcul scalaire indépendant `0,75 × valeur FP16`. Les variantes résultat manquant, identité dupliquée et epoch périmé sont refusées, chacune avec exit 2 ; le routage et le collecteur nominal sortent 0. L'empreinte de la sortie FP16 et les empreintes d'entrée/sortie des appels sont dans `result.json`.

Commande `python3 -B run.py` : exit 0. Le plan SHA est antérieur à l'exécution ; les poids, buffers et résultats détaillés ne sont pas inclus dans ce commit. Ce test établit une frontière CPU bornée avec un échantillon authentifié par plage et des copies synthétiques, pas un calcul modèle, un test de l'adaptateur, de l'ordre d'arrondi GPU ou du transport entre machines.

Suite : relier les mêmes identités et coefficients au véritable `moe_apply_dispatch_local` sous Torch CPU, puis au collecteur Rust, en explicitant le fournisseur de calcul simulé et en conservant les tests de refus.
