# R693 — deux couches VoltaFabric dans la frontière publique `llama_decode`

## Résultat vérifié

Une fixture CPU synthétique Qwen2MoE à deux couches exécute trois appels publics
`llama_decode`. Pour chaque token, les deux couches font passer la sélection
top-2 du donneur dans des Sessions Rust VoltaFabric distinctes, avec projections
`gate/up/down` propres à leur couche et le Collector FP32 prépondéré existant.
Les sorties de génération correspondent à la voie donneuse : tokens `3,3,3` et
erreur absolue maximale des logits `0`. Les routes vérifiées sont `[40,3]` à la
couche 0 et `[3,40]` à la couche 1; chaque somme des coefficients top-2 vaut
`0.127954` environ. Une faute injectée après la couche 0 fait échouer
`llama_decode` avec `-3`; le test constate le succès Session de couche 0, le refus
de couche 1, aucune lecture de logits, aucune sortie publiée et aucun token
échantillonné.

Les essais intermédiaires ont exposé deux défauts du banc. D’abord, le refus
réutilisait le Stamp réussi; le registre l’a correctement refusé avant l’injection.
Ensuite, un relancement a gardé le même run/plan/batch/attempt tout en changeant de
dossier ledger. Cette sortie est conservée mais rejetée comme preuve, car changer
de dossier ne doit pas contourner la non-relecture d’une identité. Le chargeur
ouvre désormais le chemin ledger fourni sans créer de sous-ledger nonce; le banc
crée explicitement un seul dossier stable. L’essai final utilise un run/plan neufs
(`v10`) et un nouveau batch/attempt (`9901/99001`) pour le refus. Un essai `v8`
a échoué avant dispatch à cause d’identifiants C++/Rust divergents; aucune
réservation n’a été créée. Le test final v10 passe sur une racine jamais utilisée.

## Réutilisation et adaptations

- Donneur : `ggml-org/llama.cpp`, commit
  `3cf03257f219afbe7334045ff7c6a06ac68c627d`, licence MIT vérifiée dans
  `LICENSE`. Le raccord local R679 modifie `src/llama-*`; ce n’est pas un build
  upstream intact. VoltaFabric réutilise l’API publique de modèle/décodage et le
  sampler glouton du donneur; aucun fichier upstream n’est copié dans ce lot.
- Côté VoltaFabric, le lot reprend le chargeur GGUF de tranches d’experts, les
  Sessions Rust et l’agrégation FP32 déjà présentes dans la branche source
  R692/R655. Il ajoute à `adapters/native_ggml/r693` le raccord à deux couches,
  l’inventaire par couche et l’essai tardif de refus atomique.
- Chaque couche charge indépendamment ses trois matrices Q8_0, remappe les deux
  tranches de checkpoint synthétiques `[1,0]` vers les propriétaires résidents
  `[40,3]`, et inclut la couche dans son Stamp. Le banc ne confond pas NVFP4 K16,
  MXFP4 K32 ou IQ3.

## Vérification

Le script publié compile le worker Rust en mode debug hors ligne, le chargeur
FFI Rust en mode release hors ligne et le banc C++17 avec `-Wall -Wextra -Werror`.
Le résultat frais est conservé dans `BUILD_TEST.txt`; les avertissements du crate
VoltaFabric `session.rs` et `sc117_layer_chain.rs` sont préexistants et en dehors
de R693. Le test inclut la métadonnée de modèle à deux blocs du donneur, les
projections d’experts synthétiques et le chemin d’attention/KV de la bibliothèque
llama.cpp épinglée.

## Limites et suite

CPU seulement. Poids synthétiques et vocabulaire factice de quatre IDs; aucun
checkpoint entraîné, tokenizer texte, modèle complet, GPU, worker inter-PC,
transport de production, lease R53 ou comparaison de vitesse. L’identité et le
ledger de cette fixture restent locaux et ne constituent pas le ledger matériel
partagé. `llama.cpp` fournit ici la frontière d’exécution de référence; R693 ne
qualifie ni son patch d’intégration R679 sur les appareils cible ni la compatibilité
d’un checkpoint réel.

L’inventaire R692 n’a pas établi qu’un checkpoint réel compatible était lisible.
La veille arXiv ciblée de 24 travaux est jointe; elle recommande un planificateur
CPU fondé sur les routes, octets et topologie, inspiré de VarioPath, sans changer
le dispatch. Prochaine tranche R694 : implémenter ce calcul de coût/recherche sur
entrées synthétiques explicites, vérifier qu’il conserve strictement l’ensemble
des owners et la réduction, et étiqueter toute topologie locale comme inconnue
jusqu’à mesure. Aucun téléchargement ni essai matériel. Le remplacement par un
checkpoint reste prioritaire dès qu’une source déjà admise et lisible est établie.
