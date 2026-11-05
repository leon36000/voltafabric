# R748 — compile réelle du crate VoltaFabric dans la CI PC4

La pipeline #2878 sur le commit `18964df7f7b13d1416cbc2c3d4cfb7f1034f4154` a exposé une erreur que le harnais isolé R747 ne compilait pas : le test unitaire de `src/session/mxfp4.rs:1489` omettait le champ `attempt` requis de `preweighted::Completion`. R689 avait ajouté ce champ optionnel et son initialisation `attempt: None`; R714 l'avait retirée en extrayant le préflight Session. Pour le collecteur ordinaire MXFP4 R9, `Collector::new(problem)` attend `None`. La Session contrôle déjà l'essai positif et l'identité d'essai de chaque événement/réponse avant d'accepter les lignes.

Le correctif restaure cette initialisation. Il est dans le commit `d88073ac6fa46407e73359860d81df370bb04dea`; sa relecture GitLab a confirmé le blob source `4316f3c22460663e604ede5675d62865df5bd32d`.

La pipeline #2937 est en attente depuis sa création : les trois jobs `.pre` sont `pending`, le job Rust est `created` et les logs de jobs sont vides. K3s répond comme actif sur PC4; le kubeconfig système est inaccessible à l'utilisateur non privilégié et aucun kubeconfig utilisateur n'existe. Aucun runner ni service n'a été modifié. Aucune réussite de tests racine n'est revendiquée. La prochaine preuve est le résultat de #2937.

Scans de #2878 : Gitleaks, Semgrep et Trivy ont terminé avec succès; Semgrep a signalé 155 findings sur 6 898 fichiers, non triés. SonarQube et GitGuardian ne sont pas connectés ni exécutés. Aucun payload de checkpoint réel, modèle complet, GPU, liaison inter-PC ou performance n'est qualifié.
