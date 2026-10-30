# R160 — trames réelles Session FP16 sur worker Rust CPU

Un client CPU a lancé une fois le vrai `vf-rank-session` local lié par SHA et lui a envoyé HELLO, LOAD brut de l'artefact R158, puis EXEC avec 2 048 activations FP16 brutes. Le worker a annoncé les capacités négociées, accepté le LOAD, envoyé Started, une réponse `completion_batch_fp16_raw` avec ses 2 048 valeurs, puis Finished et BYE. Identités, longueurs, SHA des trames, absence d'erreur et exit 0 ont été vérifiés. Le PID enregistré était absent après fermeture. Le plan préétabli contient les octets et identités visés ; timeout et arrêt sans replay sont prévus.

La ligne Rust CPU, qui calcule en f64 puis arrondit la sortie en FP16, a été comparée à la ligne R158 du propriétaire Python avec étapes intermédiaires FP16 : 479 des 2 048 valeurs sont égales et l'écart absolu maximal est 1,239776611328125e-05. Ce constat explique pourquoi la sortie du propriétaire ne peut pas être présentée comme identique à l'oracle f64 de Session ; il ne définit pas une tolérance GPU et ne mesure aucune qualité modèle. Les deux octets de sortie, l'empreinte R158 et la comparaison sont conservés.

Ce lot documente le protocole réel nécessaire au raccordement ; le client est un test et ne relie pas encore le propriétaire Python à une Session Rust. Aucun GPU, modèle complet ou transport de production n'a été exécuté. Le SHA complet du shard reste non vérifié.

Suite : concevoir une couture CPU bornée qui accepte ce même protocole et délègue au vrai propriétaire, avec test indépendant des trames, refus atomique et arrêt sur UNKNOWN ; choisir explicitement une validation numérique avant d'activer le chemin FP16.
