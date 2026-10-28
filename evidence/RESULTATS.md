# Résultats de livraison — 7 octobre 2026

## Conclusion
Le socle Rust est exécutable et qualifié sur CPU. Le pack prépare une intégration
Skinny/VoltaFabric, il ne livre pas un moteur distribué GPU terminé.

## Preuves
61 tests Rust passent en debug et en release. Format et Clippy (-D warnings) passent.
Le dernier contrôle CPU est lié au manifeste source SHA-256 :
`941b3d28ac98c6a9edf97789df7eec07fb3363ad785b64b7e63625dbf3f533f4`.
Consulter `remote/final_gate.json` et `remote/RUST_SOURCE_SHA256SUMS`.

L'oracle Python indépendant a exécuté 923 cas : 192 MoE aléatoires, 24 permutations,
24 collectes désordonnées, 38 collectes invalides, 192 layouts mémoire, 192 pipelines
contre recherche exhaustive, 256 codes d'échelle et 5 entrées CLI invalides. Une
corruption volontaire de résultat est détectée. Écart maximal observé :
1,4210854715202004e-14 ; aucune extrapolation vers FP16/NVFP4 GPU.

45 tests Python vérifient le client de propositions, ses chemins et réservations,
les erreurs réseau simulées et le refus de rejouer un appel au statut incertain.
32 tests de la preuve NumPy héritée ont également été rejoués. Ces populations de
tests sont séparées et ne doivent pas être additionnées comme un score de qualité.

## Corrections prouvées pendant la construction
Une allocation mémoire incorrectement évaluée a d'abord été exposée par une suite
rouge. Le client multi-agent a également été développé face à des tests rouges.
Un test a reproduit une création de dossier hors racine via un lien symbolique :
la destination est désormais résolue et contrôlée avant sa création. Une prévention
complète contre un utilisateur hostile modifiant simultanément le filesystem n'est
pas revendiquée. Les journaux correspondants sont conservés.

## Coûts et actions externes
Aucun appel à un modèle cloud n'a été effectué pour qualifier le runner ; serveur
HTTP local simulé uniquement. Il n'existe donc pas de preuve d'accès aux alias proposés.
Rust1.90.0 et ses dépendances ont été installés dans un répertoire isolé de Naos sur
PC1 pour compiler. Les GPU, leur firmware, leurs paramètres et les services du parc
n'ont pas été utilisés ou reconfigurés par ce lot.

## Prochaine action
Ouvrir le dossier extrait dans Codex et donner `PROMPT_CODEX.md`. Reproduire les preuves,
puis adapter le chemin quantifié du Skinny vivant, sans repartir d'un nouvel audit global.
Les contrats et les contre-tests fournis servent de référence aux nouvelles intégrations.
