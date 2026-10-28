# R111 — identification PC4, sans accès aux disques

Tailscale indique un pair nommé PC4 en ligne à 100.76.101.48. Cette présence ne prouve pas l'accès au disque ni une sauvegarde. L'ancien alias SSH pc4-ts pointe sur 100.124.95.18 ; lors d'un premier contrôle strict vers l'adresse observée avec cette ancienne identité épinglée, la clé différait et SSH a refusé avant authentification. Aucune clé n'a été acceptée ni modifiée.

Comparaison locale des fingerprints : la clé présentée SHA256:G25t20Qd9sRYyiNmJNzS4LsWB3Pgd0x5a+yRmZDXmO0 correspond à la clé ED25519 déjà enregistrée pour l'adresse LAN 192.168.1.161 de PC4, et non à celle de l'ancien alias Tailscale. Une seconde opération distincte, strictement épinglée à cette clé LAN, a authentifié le serveur mais a été refusée par l'authentification utilisateur : `Permission denied (publickey,password)`, exit255. La commande lsblk ne s'est pas exécutée.

Aucun volume, UUID, taille disponible, chemin de sauvegarde, fichier, copie ou restauration n'a été vérifié sur PC4. La sauvegarde reste NON VÉRIFIÉE ; aucune suppression fondée sur cette sauvegarde n'est permise. Les reçus et sorties associés sont conservés. Aucune modification SSH, clé, compte, service ou donnée distante n'a été effectuée.
