*** Structure du projet ***

´´
projet_neutronique/
│
├── core/                       # Moteur de calcul Éléments Finis
│   ├── __init__.py
│   ├── assembly.py             # Assemblage des matrices de rigidité (diffusion), masse et réaction
│   ├── time_integration.py     # Schéma d'intégration temporelle (ex: Euler implicite, Crank-Nicolson)
│   ├── boundary_cond.py        # Application de la condition de Dirichlet homogène sur le bord
│   └── solvers.py              # Résolution du système linéaire à chaque pas de temps
│
├── physics/                    # Définition du problème physique
│   ├── __init__.py
│   ├── materials.py            # Dictionnaire/classes liant les régions géométriques aux paramètres (D, Σa, νΣf)
│   ├── geometry.py             # Scripts Gmsh pour générer le cœur (combustible, modérateur, réflecteur, barres)
│   └── observables.py          # Fonctions pour calculer le flux total Φ(t) et le flux max φ_max(t)
│
├── cases/                      # Scripts exécutables pour les 4 cas d'étude
│   ├── case1_absorption.py     # Cas 1 : Influence de Σa dans la barre de contrôle
│   ├── case2_reflector.py      # Cas 2 : Influence de l'épaisseur du réflecteur
│   ├── case3_geometry.py       # Cas 3 : Influence de la géométrie du combustible (compact vs étalé)
│   └── case4_dynamic_rod.py    # Cas 4 : Insertion temporelle d'une barre de contrôle (Σa variable dans le temps)
│
├── utils/                      # Outils transverses
│   ├── __init__.py
│   ├── mesh_utils.py           # Fonctions utilitaires pour Gmsh (extraction des tags, DoFs, etc.)
│   └── visualization.py        # Routines Matplotlib pour générer les cartes 2D et exporter les animations
│
├── output/                     # Dossier généré localement (à ignorer via .gitignore)
│   ├── meshes/                 # Fichiers .msh sauvegardés
│   ├── plots/                  # Graphiques des observables (Φ(t) en fonction de t)
│   └── animations/             # Fichiers mp4/gif de l'évolution du flux
│
├── main.py                     # Script d'entrée générique (facultatif si utilisation du dossier cases/)
└── requirements.txt            # Dépendances (numpy, scipy, matplotlib, gmsh, etc.)
´´
