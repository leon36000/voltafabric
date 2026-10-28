# VoltaFabric — contrat de travail

## Mission
Étendre Skinny pour un modèle résident unifié sur V100 PCIe puis inter-PC. Rust pour
le cœur; CUDA/C++ aux frontières qui le nécessitent, Python pour les oracles/adaptateurs.
Ne pas recréer un moteur universel, une forge, Kubernetes ou un orchestrateur agentique.

## Vérité et périmètre
- Ce pack ne qualifie que du CPU et des calculs analytiques. Consulter
  `evidence/DELIVERY.json` pour les preuves réellement livrées, puis les reproduire.
- Aucun accès V100 n'est requis pour les lots CPU. Ne pas inventer de débit GPU.
- Les choix et coefficients d'experts restent identiques; aucun renormalisage caché.
- Identité complète run/modèle/plan/couche/batch/epoch; doublon, résultat manquant ou
  ancien plan provoquent une erreur, pas une sortie partielle.
- Distinguer poids/codes/échelles/padding, état/KV, espace temporaire et pic de chargement.
- Garder TP+PP corrigé comme concurrent de l'EP, sans imposer le découpage gagnant.
- Respecter les modifications existantes. Pas de reset destructif, suppression globale,
  publication, achat ou changement BIOS/VBIOS/pilote/ACS/IOMMU/puissance sans accord.

## Organisation
Un seul dossier canonique sous la racine Naos existante. Lire l'état vivant avant une
reprise; ne pas recopier le pack sur une version plus avancée. `.runs/`, `.worktrees/`,
`evidence/` et téléchargements dédiés restent dans ce dossier. Un seul intégrateur
modifie la branche principale; un propriétaire par fichier et par vague parallèle.

## Vérification
Après installation du Rust épinglé et des dépendances verrouillées :
```
python3 scripts/check_integrity.py
python3 scripts/verify.py
```
L'intégrité ne s'applique qu'avant modification volontaire. Après un patch : test ciblé,
Clippy, fmt, tests debug/release et oracle indépendant; actualiser les hashes/preuves.
Une erreur d'environnement n'est pas un test réussi. Ne jamais supprimer/ignorer un
cas gênant pour obtenir du vert. Aucun consensus de modèles ne remplace ces commandes.

## Agents
Utiliser le gateway LiteLLM existant, avec alias découverts et limites vérifiées.
Ne jamais exposer de secrets ni contourner le proxy. Les noms fournis par le propriétaire
ne sont pas des identifiants API établis. `vf_agents.py` produit des propositions sans
outils; il ne transforme pas un appel chat en agent Codex autonome. Par défaut dry-run.
Ses limites requêtes/tokens ne sont pas un plafond USD global. Pas de retry aveugle,
notamment lorsque l'issue ou la facturation d'un appel est inconnue.

## Progrès et lectures ciblées
Le prompt de mission fixe le lot courant. `docs/ARCHITECTURE.md` sert pour les frontières,
`docs/AGENT_PLAYBOOK.md` pour la délégation, `docs/HARDWARE_GATE.md` avant tout test GPU,
`docs/EVIDENCE_CONTRACT.md` pour la promotion. Ne pas relire tout à chaque modification.
Après deux boucles sans preuve nouvelle, isoler un reproducer et changer une hypothèse
ou un outil; ne pas produire un troisième plan général. Rapporter chemins, commandes,
résultats et blocages exacts. Les ressources héritées sont des pistes, pas une autorité.
