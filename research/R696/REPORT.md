# R696 — pic mémoire lors du remplacement atomique des experts

## Défaut corrigé

Le calcul R695 vérifiait KV, état récurrent, workspace, graphes, communication,
marge et buffer de chargement, puis le poids de destination. Il ne comptait pas
explicitement les poids experts déjà résidents pendant que la nouvelle affectation
est chargée. Un plan pouvait donc annoncer un pic inférieur à la capacité alors
que l'ancien et le nouvel ensemble coexistaient.

R696 ajoute `resident_expert_bytes` à l'enveloppe de chaque rang. Le plan réserve
cet ensemble existant pendant l'admission et vérifie le pic conservateur
`résidents existants + poids candidats + autres budgets + buffer de chargement`.
Il rapporte séparément le régime stable après remplacement et le pic transitoire.
Un dépassement retourne `CAPACITY_EXCEEDED` avant de rendre l'avis; les routes,
coefficients, Stamp, attempt et registre courant restent inchangés. L'enveloppe
doit compter tous les poids experts présents sur le rang, y compris ceux hors de
la route courante. Cette version suppose un chargement intégral du nouvel ensemble;
elle peut donc refuser un remplacement en place qui réutiliserait effectivement
les allocations courantes, mais elle n'affirme pas à tort que le pic est admissible.

## Donneurs et réutilisation

- Le planificateur interne R694/R695 est étendu; aucun code d'un autre moteur ou
  article n'est copié dans cette tranche.
- La frontière `llama_decode` de R693 continue de s'appuyer sur l'API publique de
  `ggml-org/llama.cpp` au commit `3cf03257f219afbe7334045ff7c6a06ac68c627d`,
  licence MIT. R696 ne modifie pas le code donneur et n'est pas encore relié à son
  callback.
- InplaceKVCache, arXiv [2609.14507v1](https://arxiv.org/abs/2609.14507), reste une
  idée de séparation des budgets de résidence du cache; aucun code/licence donneur
  n'a été vérifié. L'article traite du KV cache et ne prouve ni ce modèle de pic
  pour les poids experts ni ses coûts sur notre parc.

## Vérification et limites

Sur l'extraction CPU des modules exacts R693/R694 et la modification R695 locale,
7 tests ciblés passent. Ils couvrent une admission valide avec poids préexistants,
un dépassement causé par la coexistence ancien/nouveau, le refus atomique sans
mutation de route, les budgets KV/temporaire, les profils de rangs et l'overflow.
`rustfmt --check` et Clippy ciblé passent. La pipeline GitLab R695 2370 passe ses
quatre jobs configurés, mais ne compile pas le crate Rust.

Les capacités et budgets des tests sont synthétiques. Pas de modèle entraîné, de
callback `llama_decode` connecté, de mesure des allocations d'un vrai modèle, de
GPU, d'essai inter-PC ou de comparaison de performance. `resident_expert_bytes`
devra provenir d'un inventaire exact et versionné avant de servir à une admission
réelle; R696 ne déplace aucun poids et n'active pas de réessai.

## Prochaine tranche

R697 : connecter l'enveloppe à la frontière de routes llama.cpp sous forme de
conseil optionnel avec Stamp/attempt frais, vérifier les octets de poids depuis le
GGUF déjà admis, et garder le registre et le dispatch autoritaires. CPU seulement;
pas de GPU, R53 matériel, worker réseau ou UNKNOWN consommé.
