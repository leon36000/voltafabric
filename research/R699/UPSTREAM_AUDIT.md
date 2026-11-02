# R699 — audit d’utilisation des projets amont

## Code effectivement réutilisé dans cette tranche

Aucun fichier de code tiers n’a été copié dans R699. Le changement généralise une frontière interne VoltaFabric existante des routes Qwen2MoE au `RoutingProblem`; il réutilise son dispatch validé, son planificateur, ses profils d’experts, son budget mémoire par rang, sa Session MXFP4 et son Collector FP32 prépondéré. Les wrappers et leur comportement existant sont conservés.

## Donneur d’artefacts de test

- Dépôt: `deepseek-ai/DeepSeek-V4.1-Flash`
- Révision épinglée: `314113f8a7dadb388215396ba5fbee2c610a04d6`
- Licence déclarée dans la provenance R698: MIT
- Usage R699: seulement les fichiers de poids d’experts 0 et 383 et leurs vecteurs oracle publiés par R698; pas de code source DeepSeek importé.
- Format: MXFP4 K32 avec échelles par bloc, conservé tel quel; aucune conversion NVFP4 K16 ou IQ3.

## Autres moteurs

R699 n’importe aucun code de Strata, Maya, vLLM, llama.cpp, TensorRT-LLM, SGLang ou DeepInfer. L’adaptateur llama.cpp de R697 est une frontière distincte et n’est pas invoqué par ce test. Ce lot ne leur attribue donc ni licence, ni version, ni qualité héritée; une future intégration devra épingler le commit et la licence effective du composant effectivement repris.

## Adaptation et limites

Le seul contrat amont appliqué ici concerne les poids et oracle de modèle épinglés; l’adaptation logicielle reste propre à VoltaFabric. Le planner retourne un conseil lisible, ne réserve pas la mémoire, ne modifie pas le gate et ne garantit pas un chargement distribué atomique. Le test emploie des budgets et topologie synthétiques à partir des tailles d’artefacts, sur CPU.
