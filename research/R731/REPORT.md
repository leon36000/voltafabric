# R731 — DeepSeek HC projection, Sinkhorn et HC-pre

État local : CPU vérifié, publication GitLab en attente du commit source et de sa lecture exacte.

## Adaptation

À partir de DeepSeek-V4.1-Flash épinglé `314113f8a7dadb388215396ba5fbee2c610a04d6` (MIT), VoltaFabric charge par l’index Safetensors seulement les trois tenseurs d’une branche et d’une couche : `hc_attn_fn` ou `hc_ffn_fn`, `*_base` et `*_scale`. Les formes sont `[24, 20480]`, `[24]` et `[3]`, tous en F32, pour un total de **1 966 188 octets résidents par branche**. L’import réutilise le lecteur indexé VoltaFabric; il vérifie l’index, le nommage, la forme, les hashes des trois payloads, l’en-tête avant et après lecture, la borne de taille et la complétude avant de construire l’objet résident.

La référence CPU calcule le RMS sur les quatre copies BF16, la projection F32 scalaire, puis reproduit le découpage `pre/post/comb`, la sigmoïde, le softmax de ligne avec `eps`, la normalisation de colonne et les 19 paires de normalisations Sinkhorn par défaut. `apply_pre` réutilise le HC-pre VoltaFabric existant et exige l’identité exacte du même jeton. La branche `Attention` ou `Ffn` accompagne les coefficients; le HC-post MoE refuse maintenant les coefficients `Attention`.

Les opérations F32 sont ordonnées explicitement sur CPU. Elles n’établissent pas une parité bit à bit avec les réductions PyTorch/TileLang. Les valeurs de configuration sont des entrées explicites; les constantes exposées reprennent les valeurs par défaut `norm_eps=1e-20`, `hc_eps=1e-6`, `hc_sinkhorn_iters=20` du `ModelArgs` épinglé.

## Vérification

Le test d’intégration utilise l’index officiel et l’en-tête officiel du shard `model-00003-of-00048.safetensors`, dont le hash est `71213f444f9c15a4af6dbcbddc36f7611542884ebb3aadef54de650071357449`. Il écrit des payloads **synthétiques** avec leurs hashes dans les offsets correspondants d’un fichier sparse de 7 389 759 032 octets. Ce test vérifie les deux branches, les reçus, le HC-pre, les seuils de l’oracle f64 et le refus d’un budget insuffisant et d’identités incorrectes. Le fixture n’est pas le payload du checkpoint du modèle.

L’oracle de test calcule séparément en f64 la projection, la sigmoïde, le softmax et Sinkhorn; les coefficients sont comparés avec des seuils explicites (`2e-6` pour pre/post et `2e-5` pour comb). Le HC-pre est comparé à une somme f64 suivie d’un arrondi BF16. Le runtime PyTorch n’est pas disponible dans l’environnement courant, donc aucune comparaison à son backend n’est revendiquée.

Résultats locaux :

- `r731_deepseek_hc_projection`: 1/1 debug et 1/1 release.
- tests unitaires HC-post R730 : 3/3.
- intégration existante `r712_deepseek_session_owner`: 12/12 debug, 12/12 release.
- Clippy sur la cible et la bibliothèque avec `-D warnings`, rustfmt et `git diff --check` : succès.

## Réutilisation et limites

Réutilisés : lecteur indexé Safetensors R721, identité/stream HC R730, HC-pre BF16 R250, registre/session/route partagé R712–R730. Aucun code donneur copié et aucune dépendance ajoutée.

Pas encore réalisé : payload réel HC, comparaison runtime PyTorch/TileLang, couche complète attention/GDN et FFN alimentée par ces paramètres, génération autorégressive, GPU, transport inter-PC, R53 de production ou mesure de performance. Cette tranche valide la frontière HC, pas une génération complète ni la qualité héritée du donneur.

Le prochain lot utile est de brancher les sorties HC réelles sur les couches attention/GDN et FFN, en conservant séparément les branches et les identités, puis de raccorder leur sortie à une étape de décodage CPU vérifiable.
