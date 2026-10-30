# R151 — classe Qwen spécifique du wheel 1Cat sur CPU

Avec le Transformers 4.57.6 isolé R150 et le config.json R149, `AutoConfig.from_pretrained(..., local_files_only=True, trust_remote_code=False)` a échoué normalement avec `ValueError` : le Transformers sélectionné ne reconnaît pas `qwen3_5_moe`. Cette erreur est conservée ; elle ne démontre pas que le checkpoint est invalide ni que 1Cat est incapable de le charger.

Une opération distincte a vérifié le SHA du wheel 1Cat, extrait les octets de `vllm/transformers_utils/configs/qwen3_5_moe.py` en mémoire et exécuté cette classe exacte directement, hors import du package vLLM. `Qwen3_5MoeConfig.from_dict` a construit les classes racine et texte ; les deux dtypes sont `torch.bfloat16`, dimension de clé 128 et 40 couches dont 30 linéaires. Elle s'exécute avec le vrai Transformers 4.57.6 et Torch 2.10.0+cpu du contexte R150. Exit 0 ; source SHA et résultat dans les reçus.

Ceci qualifie seulement le parsing de la configuration avec la classe du wheel directement appelée. Il ne qualifie pas l'enregistrement de cette classe par le runtime 1Cat, la sélection effective FP16 sur V100, les poids, l'exécution d'un modèle ou un GPU. Le problème de dépendances TileLang demeure ouvert. Aucun téléchargement et aucune modification d'environnement actif.

Suite : vérifier que la couche de configuration vLLM emploie et enregistre cette classe, puis préparer un test CPU MoE/collecteur limité à partir des échantillons R132 et des garanties de l'adaptateur.
