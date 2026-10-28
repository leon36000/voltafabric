# LiteLLM : propositions bornées, intégration par Codex

## Deux modes différents

Les sous-agents natifs de Codex peuvent disposer de vrais outils et d'un espace de
travail. Le client `scripts/vf_agents.py` n'a pas ces outils : il envoie un contexte
explicite à un modèle et conserve sa proposition. Ne pas confondre ces modes.
Les propositions sont mises en quarantaine dans `.runs/<run-id>/`. Aucun patch ni
commande suggérée n'est exécuté par ce client. Le statut imposé est NOT_EXECUTED.

## Mise en route

Le dry-run ne demande pas de clé et ne fait aucun appel réseau :
```
python3 scripts/vf_agents.py run --config agents/routes.example.json --queue agents/queue.example.json
```
Les alias sont volontairement `null`. Ils doivent être remplacés dans une copie de
configuration locale, après discovery via les variables déjà autorisées :
```
python3 scripts/vf_agents.py models
```
Variables : `VF_LITELLM_URL`, `LITELLM_API_KEY`. Les charger avec le mécanisme existant,
sans les écrire dans le dépôt ou les afficher. HTTP local nécessite l'option explicite
`--allow-local-http` et une adresse privée littérale/localhost. HTTPS sinon.
Le client refuse les hôtes directs OpenRouter/OpenAI/Anthropic connus, les redirections
et les retries automatiques. Ce filtre n'est pas une authentification de tout serveur
arbitraire : l'opérateur doit vérifier qu'il s'agit bien du gateway existant.

Après canary et vérification des plafonds et de l'autorisation :
```
python3 scripts/vf_agents.py run --config agents/routes.local.json --queue agents/queue.example.json --run-id wave-001 --execute --authorized-spend
```
Ne lancer cette commande que dans le périmètre autorisé. L'attestation CLI ne prouve pas
un plafond financier serveur. Une route absente est rejetée avant tout POST modèle.

## Limites réelles du client

- Réservations SQLite atomiques : nombre d'appels et tokens de sortie maximum par run.
- Pas de calcul USD garanti, pas de plafond des tokens d'entrée/reasoning cachés, pas
  de budget couvrant d'autres clients. Usage absent reste inconnu, jamais zéro.
- Limites du run immuables; identifiant déjà réservé non rejoué automatiquement.
- Une issue inconnue conserve sa réservation; aucun remboursement implicite.
- Timeout d'I/O, pas deadline murale globale assurée pour tout le pool; ne pas lancer
  un service non fiable susceptible d'envoyer indéfiniment des octets au compte-gouttes.
- Contexte sur allowlist, taille bornée, exclusion de chemins sensibles. Ce filtre
  n'est pas un outil DLP complet : vérifier aussi les secrets présents dans le contenu.
- Racine locale supposée contrôlée : tests de liens symboliques, mais pas sandbox
  hostile multi-utilisateur avec défense universelle contre toutes les courses TOCTOU.
- `write_set` coordonne les propositions; il ne constitue pas un confinement OS.
- Les tests HTTP utilisent un serveur simulé, pas les quatre modèles payants.

## Cadence de travail

Une vague = tâches indépendantes, propriétaires définis, tests d'acceptation. Un seul
intégrateur. Réduire le parallélisme si la file de patches non revus augmente. Les appels
supplémentaires doivent lever une incertitude ou livrer un diff utile. Ne pas recruter
des dizaines de reviewers qui répètent le même avis. Une canary par alias avant montée
en charge, puis preuve fonctionnelle; aucun classement définitif sur le nom du modèle.
