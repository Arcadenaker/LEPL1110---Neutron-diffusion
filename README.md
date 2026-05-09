# LEPL1110 — Projet d'Éléments Finis : Simulateur de Diffusion Neutronique

Ce logiciel est un simulateur complet de la diffusion neutronique non stationnaire dans un cœur de réacteur nucléaire simplifié. Il repose sur la méthode des éléments finis (FEM) P1 et utilise une accélération matérielle pour permettre des simulations dynamiques et des études paramétriques complexes.

## Fonctionnalités Principales

* **Assemblage Haute Performance** : Utilisation de **Numba** pour compiler les boucles d'assemblage en code machine (C++), permettant une exécution massivement parallèle sur tous les cœurs du processeur.
* **Moteur Géométrique Avancé** : Génération de cœurs de réacteurs à géométrie hexagonale avec fragmentation booléenne (via Gmsh) pour un maillage conforme et précis.
* **Physique Nucléaire Réaliste** : Base de données de matériaux (UOX, MOX, Eau, Graphite, Boral) et prise en compte de l'inertie neutronique ($L_{eff}$).
* **Contrôleur PD Intelligent** : Un pilote automatique gère l'insertion des barres de contrôle pour maintenir ou atteindre une puissance cible (en MW).

## Fonctionnalités de l'Interface Graphique (GUI)

L'interface graphique (`gui.py`) offre un contrôle total sur la conception et la simulation du réacteur :

### 1. Configuration Géométrique et Matérielle
* **Dimensions Globales** : Ajustez le rayon des hexagones, le rayon du noyau et l'épaisseur du réflecteur.
* **Sélecteur de Matériaux** : Choisissez les matériaux pour le combustible, le modérateur, le réflecteur et les barres de contrôle parmi la base de données intégrée.
* **Structure Interne** : Définissez le nombre de couronnes de crayons (pins) à l'intérieur de chaque assemblage de combustible ou de contrôle.

### 2. Conception du Cœur
* **Répartition Automatique** : Placez automatiquement des anneaux de barres de contrôle avec une densité spécifique.
* **Peinture Manuelle (Pinceau)** : Un outil interactif vous permet de "peindre" le cœur pour alterner manuellement entre assemblages de combustible (Bleu) et barres de contrôle (Rouge).
* **Aperçu Visuel Rapide** : Visualisez instantanément la géométrie du cœur et le placement des crayons avant de générer le maillage.

### 3. Gestion du Maillage
* **Génération Automatique** : Crée un maillage `.msh` optimisé avec un gradient de finesse (plus fin près des crayons).
* **Visualiseur Gmsh Intégré** : Lancez Gmsh directement depuis l'interface pour inspecter la qualité du maillage 2D généré.

### 4. Simulations et Analyses
* **Simulation Dynamique** : Lance la résolution temporelle avec animation en temps réel du flux neutronique et de la puissance thermique.
* **Cas 1 : Étude de Pilotabilité** : Analyse par force brute des positions d'équilibre pour différentes configurations d'anneaux de contrôle.
* **Cas 2 : Cartographie Overshoot** : Génère une Heatmap de l'overshoot du contrôleur en fonction de l'épaisseur du réflecteur et de la position des barres.
* **Cas 3 : Validation Théorique (V&V)** : Vérifie la conservation de la population, la positivité du flux et les bornes de croissance théoriques.

## Architecture du Projet

* **`core/`** : Moteurs d'assemblage (`assembly.py`), intégration temporelle (`time_integration.py`) et solveur principal (`solver.py`).
* **`physics/`** : Définition de la géométrie CAO (`geometry.py`) et propriétés nucléaires des matériaux (`materials.py`).
* **`cases/`** : Scripts dédiés aux études de cas (statique, overshoot, validation).
* **`utils/`** : Interface graphique (`gui.py`) et système de logs (`logger.py`).
* **`main.py`** : Point d'entrée pour lancer l'application.

## Installation et Lancement

1.  Installer les dépendances :
    ```bash
    ```
    pip install numpy scipy numba meshddio matplotlib gmsh pypardiso
    ```
    ```
2.  Lancer le simulateur :
    ```bash
    python main.py
    ```

## 🔬 Modélisation Mathématique
Le simulateur résout l'équation de diffusion neutronique :
$$\frac{1}{v} \frac{\partial \phi}{\partial t} - \nabla \cdot (D \nabla \phi) + \Sigma_a \phi = \nu \Sigma_f \phi$$
La résolution utilise un schéma de discrétisation temporelle de type **Theta-méthode** (Crank-Nicolson) et une factorisation LU via **Pypardiso** pour une performance maximale.
