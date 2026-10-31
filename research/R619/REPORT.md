# R619 — revérifier registre et lease aux frontières distribuées

Le coordinateur vérifie ensemble l’identité du registre R53 et la lease verrouillée avant chaque création de session d’experts, avant chaque position et après son calcul. Une admission incomplète, une lease perdue ou un chemin de registre remplacé provoque une erreur ; les exécutions purement locales sans admission distribuée restent permises. Ce contrôle ajoute à R618 une détection après la réservation, notamment pendant le chargement du modèle.

Le test utilise une vraie flock locale et des répertoires synthétiques : il accepte la paire valide, refuse une admission incomplète, puis refuse le registre remplacé même lorsque la lease reste valide. Les 17 tests du coordinateur passent en debug et release. Clippy ciblé passe avec l’avertissement historique dead_code autorisé ; rustfmt passe. Aucun worker distant, GPU, service ou registre historique n’a été lancé ou modifié.

Ces contrôles ponctuels ne garantissent pas la stabilité entre deux contrôles, ne prouvent pas la coopération des autres processus et ne rétablissent ni R53 partagé ni lease réelle. Une erreur après transmission peut correspondre à une opération UNKNOWN ; aucun rejeu n’est permis. La qualification numérique et toutes les admissions restent nécessaires avant distribution.
