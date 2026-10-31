# R634 — refus avant réservation des modes distants incompatibles

Le lanceur R632 pouvait réserver les deux identités R53 avant de passer au coordinateur une configuration distante que celui-ci refuse déjà : affinité non collante, backend scalaire, préremplissage groupé ou trace QSA. Une réservation restait alors consommée malgré un échec prévisible. Un test synthétique reproduit ce comportement avant correction : le moteur simulé est appelé et les deux fichiers existent.

Lorsque `--reserve-r53` est demandé, le lanceur refuse maintenant ces quatre catégories avant réservation et avant lancement. Il exige le backend vectoriel, l'affinité `sticky`, un groupe de taille un et l'absence de trace QSA, comme le coordinateur Rust actuel. Le test du pont Python–Rust R633 utilise désormais une configuration cohérente. Les contrôles existants du coordinateur demeurent en place ; cette vérification anticipée ne les remplace pas et ne couvre pas toutes les causes possibles d'échec après réservation.

Les cas incompatibles sont refusés sans fichier de réservation ni appel du moteur simulé ; les cas compatibles continuent à vérifier la transmission du verrou et le non-rejeu. Sur PC1 : 47 tests de l'adaptateur, dont un ignoré pour le pack observé non configuré. Sur PC6 : 21 tests du binaire en debug et release, Clippy ciblé et rustfmt. Le premier test ciblé échouait comme attendu avant le correctif. Aucun worker réseau, GPU, registre R53 réel ou UNKNOWN antérieur n'a été utilisé.

Les contrôles anticipés ne certifient ni une lease coopérative, ni le registre partagé, ni l'identité du checkpoint ou les admissions matérielles. La réconciliation réelle reste nécessaire avant exécution distribuée.
