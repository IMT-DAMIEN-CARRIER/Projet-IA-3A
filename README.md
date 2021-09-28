# projet-IA-3A

---
```
Léon Bousquet - Damien Carrier - Arthur Duca - Clément Savinaud
INFRES 12
```

---

## Prérecquis

Installation de [docker](https://docs.docker.com/get-docker/) sur votre machine.

## Lancement du projet : 

* Build du projet : `docker build . -t projet_ia`

* Run du projet : `docker run -p 1111:1111 --name projet_ia -d projet_ia`
* Observer les résultats : `docker logs -f projet_ia`