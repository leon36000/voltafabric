# R633 — pont synthétique lanceur–réservation–consommateur Rust

Un essai local relie désormais les composants R631 et R632 dans une seule chaîne : le lanceur de prompt réserve la base et le suffixe, transmet le même descripteur de verrou à un processus enfant, puis le consommateur Rust vérifie l'admission et crée son marqueur à usage unique. Une seconde tentative est refusée avant un nouveau lancement. Si le plan est modifié après réservation, le consommateur le refuse et ne crée pas de marqueur de démarrage ; les deux réservations restent consommées.

Le processus enfant est le binaire de tests Rust, pas un moteur génératif ni un worker. Le tokenizer est synthétique et le module `regex` est remplacé uniquement dans cet essai : aucune sémantique de tokenisation n'est vérifiée. Le verrou est un véritable flock local, conservé ouvert tout au long du test. Aucun registre R53 actif, réseau ou GPU n'est utilisé.

Les 21 tests du binaire passent en debug et release ; rustfmt et Clippy ciblé sur le binaire passent. Un essai plus large de Clippy avec `--tests` échoue sur des avertissements préexistants dans `vf-sc117-qsa-eviction-fault-check.rs`, `vf-sc117-eight-layer-dynamic.rs` et `sc117_layer_chain.rs` ; ces avertissements ne sont pas masqués. Aucun code de production n'est modifié dans ce lot.

Ce test n'établit ni un verrou partagé entre machines, ni la coopération des services voisins, ni la complétude du registre historique. Les admissions et la réconciliation réelles restent obligatoires avant de lancer un worker distribué ou GPU.
