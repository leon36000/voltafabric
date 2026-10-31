# R618 — refus si le chemin du registre R53 est remplacé

Après création et synchronisation du marqueur de démarrage dans le répertoire R53 ouvert et identifié, le coordinateur rouvre maintenant le chemin et vérifie de nouveau son identité et ses permissions. Si le répertoire est remplacé pendant la consommation, il refuse de poursuivre ; le marqueur initial reste consommé dans le répertoire ouvert, sans suppression ni rejeu.

Un test synthétique remplace le chemin entre l’écriture du marqueur et le contrôle final. Le coordinateur refuse, et le test vérifie que le marqueur existe seulement dans le répertoire d’origine. Les 17 tests du coordinateur passent en debug et release, Clippy ciblé et rustfmt passent. Aucun worker ni registre historique, GPU ou service n’a été utilisé.

Cette vérification détecte un remplacement persistant au moment du contrôle. Elle ne garantit pas que le chemin restera stable ensuite, ne prouve pas la coopération des autres processus et ne reconstitue ni le registre R53 partagé ni sa lease. Les admissions existantes et l’interdiction de rejouer UNKNOWN restent nécessaires.
