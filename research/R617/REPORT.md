# R617 — comparaison numérique locale PC1 et PC6

Dix experts IQ3_XXS/IQ4_NL de la couche 0 et une activation réelle du checkpoint SC117 ont été comparés sur les deux architectures. R615 avait vérifié les captures scalaires sur PC1. Sur PC6 aarch64, le même adaptateur scalaire GGML a produit exactement les 25 600 valeurs f64 capturées sur PC1 : dix cas identiques, aucune différence. Les dix contrôles de taille invalide ont également préservé la sortie. Les dix blobs, l’activation, les captures et le binaire ont été vérifiés par SHA-256 avant calcul.

Ce corpus ne comporte qu’une activation, une couche et un couple de formats. Les décodeurs et quantificateurs GGML sont partagés ; cette comparaison ne démontre ni équivalence générale, ni exactitude indépendante du modèle. Aucun worker réseau, GPU ou registre R53 n’a été utilisé. Les calculs PC6 utilisaient une limite de 512 Mio, zéro swap et 50 % d’un cœur ; les données du modèle restent locales. Le registre partagé, la lease coopérative et les autres admissions demeurent nécessaires avant un essai distribué.

Sources : données et captures R612, validation PC1 R615 et adaptateur GGML issu de llama.cpp commit `3cf03257f219afbe7334045ff7c6a06ac68c627d` (MIT). Les formats IQ3_XXS et IQ4_NL ne sont ni MXFP4 ni NVFP4.
