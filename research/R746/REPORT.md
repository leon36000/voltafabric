# R746 — codec borné des résultats greedy de vocabulaire partitionné

## Résultat

R746 ajoute à l’adaptation de VoltaFabric de la tête vocabulaire partitionnée DeepSeek V4.1 un codec JSON versionné, à trame unique et bornée, pour le résultat top‑1 local d’un rang. Le score FP32 est sérialisé par ses bits `u32`, afin de préserver sa représentation exacte. Le réducteur vérifie ensuite que le rang annoncé correspond au rang de la Session fourni par l’appelant, puis applique ses validations d’identité, de doublon et de complétude avant d’émettre un token.

Le codec refuse les trames trop grandes, multiples ou tronquées, les champs inconnus, une version ou un type inattendus, une identité/digest invalide, un attempt nul, les gagnants incomplets, les scores non finis et les positions qui débordent. Les échecs de rang sont atomiques : un rang rejeté n’est pas consommé et peut ensuite être accepté sous la bonne identité.

## Réemploi et provenance

Cette tranche réutilise et étend le réducteur greedy VoltaFabric de R745. R745 documente la révision source DeepSeek‑V4.1‑Flash épinglée `314113f8a7dadb388215396ba5fbee2c610a04d6` et sa licence MIT. R746 ne copie pas de code DeepSeek, vLLM, TensorRT‑LLM, llama.cpp, SGLang, Strata, Maya ou DeepInfer. La tranche est une adaptation VoltaFabric, pas une intégration de ces moteurs ni une qualification de leurs licences/version dans le produit.

Le harnais CPU est déplacé sous `research/R746/cpu-harness/` pour garder les artefacts R745 historiques inchangés. Il importe directement les sources de production de R746 et utilise une structure `Stamp` réduite au contrat testé; il ne compile pas l’application complète.

## Limites actuelles

Le codec n’est pas encore une variante de `Session::Reply`, et aucun producteur de worker ni lecteur de Session n’envoie/reçoit ces octets dans le protocole VoltaFabric. `expected_session_rank` doit provenir d’une identité de Session déjà admise; le codec n’authentifie ni le pair ni le transport. Il n’y a donc pas de qualification réseau, inter‑machine, GPU, projection de tête, checkpoint, modèle complet ou génération complète. Aucun résultat de performance n’est revendiqué.

La prochaine tranche est l’ajout d’une variante stricte à `Session::Reply` et de son branchement au producteur/collecteur existant, avec tests de rejet atomique et identité de Session. Elle doit d’abord être faite contre la source exacte GitLab de `session.rs`, sans écraser d’évolution concurrente.

## Vérification

Le harnais indépendant du build principal passe 22 tests; 3 tests de référence tokeniseur restent ignorés faute du tokeniseur épinglé. `cargo clippy` passe avec deux avertissements `dead_code` hérités des compteurs de test du module SHA‑256. `rustfmt --check` passe pour les sources touchées. Les détails sont dans `TEST-RESULT.txt`.
