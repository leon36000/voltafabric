# R714 — exécution CPU du top-6 lié au routage

## Changement

R713 qualifiait six owners et leur LOAD, sans lancer de calcul. R714 relie maintenant le reçu unique de préflight aux six Sessions MXFP4 et au Collector FP32 prépondéré déjà présent dans VoltaFabric. `execute_pinned_mxfp4_route` consomme le préflight et les reçus LOAD, vérifie l'identité complète, la route, ses coefficients, les six bindings rank/local et les activations avant le premier envoi. Chaque Session est contrôlée contre son binding chargé et le contrat du Collector avant qu'un rang reçoive EXEC.

La tranche exécute un token à six rangs distincts. Elle applique le contrat R9 : le producteur prépondère une fois en FP32 et le Collector somme les résultats FP32 dans l'ordre défini. La sortie n'est exposée qu'après six reçus terminaux complets et `collector.finish()`. Une réponse ambiguë ou incomplète rend un reçu terminal `Unknown` ou `Aborted`, sans sortie partielle et sans nouvelle tentative.

Le contrôle de Session a été extrait en `preflight_preweighted_fp32_until`. Le point d'envoi répète ces vérifications, afin qu'un préflight multi-rang ne remplace pas les contrôles immédiats de capacité, de binding, de deadline, d'entrée et d'état de Session.

## Preuve CPU

Une fixture à six experts utilise les métadonnées/index DeepSeek déjà épinglés et des octets de poids synthétiques. Six processus `vf-rank-session` locaux chargent un expert chacun. La route complète s'exécute ensuite sur CPU et sa sortie est comparée à `Mxfp4Expert::forward_reference`, oracle distinct du producteur Session R9. Les coefficients de la fixture sont des puissances de deux exactement représentables en FP32 et leur somme vaut un. Le test impose un écart absolu maximal de `5e-4` et confirme un EXEC par rang, un reçu complet par rang et une fermeture avec `bye_execs = 1`.

Les tests de lecture sélective et de budget restent verts en debug et release. Clippy strict, rustfmt et `git diff --check` passent.

## Réemploi et limites

Réemploi effectif : Safetensors et le propriétaire MXFP4 existants, le dispatch du routeur, les Sessions HELLO/LOAD/EXEC, les reçus, et le Collector R9 FP32 prépondéré de VoltaFabric. Aucun code tiers n'a été copié dans cette tranche. Le sondage arXiv dirigé par GPT‑6 Luna reste consigné dans `research/R712/ARXIV_SCOUT.md`; il ne fournit pas de noyau tiers qualifié pour le parc visé.

La fixture est synthétique : aucun payload de checkpoint réel n'a été lu. La preuve ne couvre ni génération modèle complète, attention, cache KV ou tokenizer, ni GPU/V100, Spark/RTX 5080, réseau inter-PC, RDMA, R53 de production, débit ou performance. Elle ne transforme pas le pipeline GitLab en preuve de compilation/test; ce sont les reçus CPU locaux qui les établissent.

Limite structurelle découverte : l'adaptateur actuel exige six rangs distincts, car son owner et son LOAD portent un seul expert. Il refuse explicitement le cas où plusieurs experts top-6 résident sur le même rang. Une intégration au registre réel devra soit regrouper les experts d'un rang dans un owner et une Session, soit démontrer que le placement écarte ces routes; aucun des deux n'est encore établi. La réutilisation de composants donneurs ne confère pas automatiquement leurs garanties à VoltaFabric.

## Suite

R715 doit traiter le cas fréquent possible des experts co-localisés : construire un owner MXFP4 par rang contenant exactement les experts sélectionnés de ce rang, lier ce groupe au même préflight, et qualifier plusieurs slots par Session sans appliquer deux fois les coefficients. La première étape reste CPU/synthétique et refuse atomiquement tout groupe incomplet. Ne pas engager de payload réel, GPU ou transport inter-PC sans les admissions propres à ces voies.
