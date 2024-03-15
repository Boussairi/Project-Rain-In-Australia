# Projet IFT870 : *Rain In Australy*

## Aperçu

Ce projet, qui s'inscrit dans le domaine de la biodiversité, vise à résoudre le défi de la prédiction de la pluie en Australie en utilisant des techniques de machine learning. L'objectif est de développer des modèles de classification capables de prédire s'il pleuvra le lendemain en se basant sur diverses caractéristiques météorologiques.

## Structure du Projet

    ├── data/
    │   ├── weatherAUS.csv
    │   ... 
    ├── src/
    │   ├── __init__.py
    │   ... 
    ├── requirements.txt
    ├── partie 1 - projet.ipynb
    ├── LICENSE
    ├── AUTHORS.rst
    ├── .editorconfig
    ├── .travis.yml
    ├── .gitignore
    └── README.md

La structure du projet est organisée comme suit :

- **data/** : Répertoire contenant les données du projet, avec notamment le fichier `train.csv`.

- **src/** : Répertoire abritant les scripts source du projet :
  - **_init_.py** : Fichier d'initialisation du module Python.
  - ...

- **requirements.txt** : Fichier spécifiant les dépendances du projet.

- **projet.ipynb** : Fichier Jupyter Notebook contenant le code principal du projet.

- **LICENSE** : Fichier décrivant la licence du projet.

- **AUTHORS.rst** : Fichier listant les auteurs du projet.

- **.editorconfig** : Fichier de configuration pour les éditeurs de texte.

- **.travis.yml** : Fichier de configuration pour l'intégration continue avec Travis CI.

- **.gitignore** : Fichier spécifiant les fichiers et répertoires à ignorer lors du suivi avec Git.

- **README.md** : Fichier que vous êtes actuellement en train de lire, contenant des informations générales sur le projet.

## Installation

   ```bash
   # Clonez le dépôt GitHub
   git clone (https://github.com/Boussairi/Project-Rain-In-Australy)
   cd nom-du-projet
   
   # Créez un environnement virtuel (optionnel, mais recommandé)
   python -m venv venv

   # Pour activer l'environnement virtuel (sous Windows) 
   venv\Scripts\activate

   # Pour activer l'environnement virtuel (sous macOS/Linux)
   source venv/bin/activate

   #Installez les dépendances
   pip install -r requirements.txt

### Comment Contribuer

1. **Créez une Issue :** Avant de commencer à contribuer, créez une issue pour discuter des modifications que vous envisagez. Cela permet d'obtenir des commentaires et d'éviter tout chevauchement avec d'autres contributeurs.

2. **Fork et Clone :** Fork le dépôt vers votre propre compte GitHub et clonez-le sur votre machine locale.

   ```bash
   git clone https://github.com/Boussairi/Project-Rain-In-Australy 
   ```

3. **Créez une Branche :** Créez une branche pour travailler sur votre nouvelle fonctionnalité ou correction.

   ```bash
   git checkout -b nom-de-votre-branche
   ```

4. **Effectuez les Modifications :** Faites vos modifications et assurez-vous de suivre les conventions de codage existantes.

5. **Testez Localement :** Avant de soumettre une demande de fusion, assurez-vous que votre code fonctionne correctement localement.

6. **Soumettez une Pull Request :** Lorsque vous êtes prêt, soumettez une pull request (PR) depuis votre branche vers le dépôt principal. Expliquez clairement les changements que vous avez apportés et pourquoi ils sont nécessaires.



Authors
--------

- Abir Jamaly
- Hamza Boussairi
- Jinane Boufaris
- Couthon Mallory
