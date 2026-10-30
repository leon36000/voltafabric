# R157 — mutations locales des buffers QPN2 CPU

Dans une nouvelle exécution CPU avec identité R157, l'adaptateur réel, le prépacking du donneur et le fournisseur CPU R156 consomment l'échantillon d'expert R132. Deux copies synthétiques de cet expert alimentent deux RankDispatch. Le témoin scalaire sur les octets originaux retrouve exactement les 2 048 sorties FP16 ; le collecteur Rust vérifie la somme pondérée et l'ordre inversé des résultats. Deux epochs périmés sont refusés avant fournisseur.

Deux mutations distinctes sont exercées sur des copies de buffers **après admission**, dans la projection down : mise à zéro des codes ou des échelles pour la lane 0 du premier tile et tous ses groupes. Pour chacune, seule la sortie d'indice 0 change, passant d'une valeur nominale non nulle à zéro ; les 2 047 autres restent identiques. Cela complète les mutations globales R156 par un contrôle de localisation du fournisseur CPU, sans montrer la capacité du kernel CUDA à détecter une corruption. Toutes les étapes sortent 0 et les identités et empreintes figurent dans les reçus.

Le calcul utilise déquantisation et étapes FP16 puis produits binaire64 ; l'oracle partage l'interprétation du format. Aucun GPU, runtime 1Cat, modèle complet ou SHA complet du shard n'a été qualifié. Les buffers mutés sont des copies privées du test, pas un mécanisme de correction ou de détection de corruption de production.

Suite : relier la frontière à une session et un résident durable avec tests d'échec et d'identité, ou qualifier les dépendances runtime isolées ; toute admission GPU demeure une opération distincte.
