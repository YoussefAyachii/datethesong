Conventions
===========

* Fichiers et dossiers: tous les fichiers `.py` sont dans le
  répertoire courant, les données sont dans le sous-répertoire `data/`.
* Les fichiers `.py` dont le nom ne commence pas par `exp_` ni pas `test_`
  contiennent des fonctions et des classes, sans lignes de code à exécuter en
  dehors: ils contiennent en particulier tous les algorithmes et les fonctions
  pour accéder aux données
* Scripts: les fichiers `.py` dont le nom commence par `exp_` sont des scripts
  que l'on peut exécuter.
* Tests: les fichiers `.py` dont le nom commence par `test_` sont des
  fichiers de test, le fichier `test_xxx.py` contient les tests pour le code
  du fichier `xxx.py`
* Les commentaires permettent de documenter le code. Les lignes de code ne
  doivent pas être désactivées/activées en mettant ou en supprimant des
  commentaire au début de ces lignes.

Tests
=====

Vous pouvez lancer les tests de plusieurs façons.

1/ installer le paquet `spyder-unittest` avec `conda` ou `pip` et utiliser le
menu ajouté à `spyder`.

2/ sans `spyder-unittest`, à partir de Spyder: ouvrez le fichier de test dans
l'éditeur, appuyez sur le bouton "Exécuter le fichier"

3/ dans un terminal, placez-vous dans le répertoire principal (le répertoire
courant), puis tapez

python -m unittest

4/ dans un terminal, placez-vous dans le répertoire principal (le répertoire
parent de `tests` et de `song_dating`), puis tapez

coverage run -m unittest

cela lance les tests et effectue également une couverture du code. Vous pouvez
ensuite voir le rapport de couverture via la commande

coverage report

qui affiche le rapport de façon synthétique dans le terminal, ou bien via les
commandes

coverage html
open htmlcov/index.html

qui construit un rapport plus détaillé sous la forme de page web (première
commande) et l'ouvre dans un navigateur web (deuxième commande).

