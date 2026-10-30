# R150 — lecture réelle de config Transformers sur CPU PC1

Dans un répertoire privé nouveau sur PC1, les 18 wheels du sous-graphe Transformers R143 ont été téléchargés une fois puis leurs tailles et SHA R142 vérifiés : 40 095 831 octets compressés. Le plan externe précède toute lecture. Les wheels sont conservés sous `r150-transformers-config-cpu-20261028/wheels`; aucun modèle ou GPU n'a été chargé.

La première installation hors ligne a échoué (exit 1) car l'interpréteur CPU Torch existant n'a pas le module pip. Ses 18 téléchargements vérifiés et son reçu restent conservés ; ils n'ont pas été répétés. Une opération distincte a revérifié leurs octets, utilisé le pip existant du virtualenv R140 et installé `--no-index --no-deps --no-compile` dans un nouveau dossier `site-r2`. Installation exit 0, aucun package actif modifié.

En utilisant Python/Torch CPU 2.10.0+cpu existants et ce répertoire privé, `transformers.PretrainedConfig.from_json_file` a réellement lu les octets de configuration R149 : Transformers 4.57.6, `dtype` racine identifié comme l'objet `torch.bfloat16`, `model_type=qwen3_5_moe`, exit 0. C'est une classe générique : `text_config` reste un dictionnaire et aucune classe d'architecture spécifique ou classe 1Cat n'est chargée. Aucune validation du chargement de modèle, du type effectivement transmis au moteur ou de la compatibilité GPU n'est déduite.

Scripts et reçus inclus ; roues, installation et sorties détaillées demeurent sur PC1. Le contexte est limité aux dépendances du parseur et ne résout pas le conflit de versions 1Cat/TileLang. Suite : vérifier hors ligne la disponibilité des classes de configuration 1Cat et relier l'adaptateur CPU au collecteur avec des identités réelles ; préserver les données et aucun replay de soumission inconnue.
