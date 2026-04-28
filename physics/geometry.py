# -*- coding: utf-8 -*-
"""
LEPL1110 — Projet d'Éléments Finis
Projet : Diffusion neutronique non stationnaire dans un cœur de réacteur simplifié

Fichier : geometry.py
Description :
    Ce module constitue la brique de pré-traitement du simulateur. Il assure la
    définition géométrique du cœur du réacteur et la génération du maillage :
    - Calcul vectorisé des centres d'assemblages hexagonaux (NumPy).
    - Identification de la frontière extérieure par analyse topologique.
    - Construction de la géométrie CAO via l'API Gmsh (noyau OpenCASCADE).
    - Assignation des groupes physiques (Fuel, Moderator, Reflector, ControlRods)
      pour l'application des propriétés matériaux dans le solveur.
"""

import os
import numpy as np
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import gmsh


# ==============================================================================
# 1. MOTEUR MATHÉMATIQUE ET TOPOLOGIQUE (NumPy vectorisé)
# ==============================================================================
class ReactorGeometry:
    def __init__(self, R_n, R_hex, form="circle"):
        self.form = form
        self.R_hex = R_hex
        self.R_n = R_n
        self._pins_cache = {}

    def hex_centers(self):
        """Génère les centres des hexagones via grille vectorisée."""
        # Calcul des bornes de la grille (Indexation discrète).
        # On détermine le nombre de rangées verticales nécessaires.
        # La distance verticale entre deux centres est 1.5 * R_hex.
        b_lim = int(np.ceil(2 * self.R_n / (3 * self.R_hex)))

        # Bornes horizontales : on ajoute b_lim/2 pour compenser le décalage
        # horizontal des lignes impaires dans le pavage.
        a_lim = int(np.ceil(self.R_n / (self.R_hex * np.sqrt(3)) + b_lim / 2))

        # Création de la grille de coordonnées (Espace des indices).
        # np.mgrid génère deux matrices 2D contenant tous les couples (b, a) possibles.
        # C'est une manière très performante de créer un "maillage" d'indices entiers.
        # Plus rapide qu'une boucle for, cette optimisation est nécessaire pour pouvoir faire des études paramétriques.
        b_vals, a_vals = np.mgrid[-b_lim : b_lim + 1, -a_lim : a_lim + 1]

        # .ravel() transforme les matrices 2D en vecteurs 1D (aplatissement).
        # C'est l'étape clé pour la vectorisation : on passe d'une structure de grille
        # à deux listes simples de coordonnées qu'on peut traiter massivement.
        A_flat, B_flat = a_vals.ravel(), b_vals.ravel()

        # Transformation des coordonnées (Index -> Cartésien).
        # Conversion des indices entiers (a, b) en positions réelles (X, Y) en mètres.
        # Le terme "+ B_flat / 2" réalise le décalage horizontal une ligne sur deux.
        X = self.R_hex * np.sqrt(3) * (A_flat + B_flat / 2)
        Y = self.R_hex * 1.5 * B_flat

        # Filtrage géométrique (Cœur circulaire).
        # On applique le théorème de Pythagore : X² + Y² <= R_n².
        # 1e-9 est une tolérance de sécurité contre les erreurs d'arrondi flottant.
        # 'mask' est un tableau de booléens (True/False) indiquant si l'hexagone est dans le cœur.
        mask = (X**2 + Y**2) <= (self.R_n**2 + 1e-9)

        # Extraction et formatage des résultats.
        # X[mask] ne conserve que les coordonnées validées par le masque circulaire.
        # np.column_stack empile les deux vecteurs 1D pour créer une matrice (N, 2) de centres.
        return np.column_stack((X[mask], Y[mask]))

    def get_outer_perimeter(self, precision=5):
        """Extrait la frontière externe via filtrage topologique vectorisé (nombres complexes)."""
        # 1. Récupération des centres et vérification
        centers = self.hex_centers()
        if centers.size == 0:
            return np.array([])

        # 2. Génération des sommets de chaque hexagone
        # On définit les 6 angles pour les sommets d'un hexagone "pointy-topped"
        angles = np.pi / 6 + np.arange(6) * (np.pi / 3)
        cx, cy = centers[:, 0, None], centers[:, 1, None]

        # Calcul des coordonnées (x, y) de tous les sommets de tous les hexagones
        # np.round est crucial pour que deux sommets identiques tombent sur la même valeur numérique
        vx = np.round(cx + self.R_hex * np.cos(angles), precision)
        vy = np.round(cy + self.R_hex * np.sin(angles), precision)

        # Astuce : On convertit les coordonnées (x,y) en nombres complexes (x + iy)
        # Cela permet de manipuler un point comme une seule entité au lieu de deux
        z = vx + 1j * vy

        # 3. Création des arêtes (segments)
        # np.roll décale les sommets pour lier le sommet i au sommet i+1
        # edges devient une liste de segments [début, fin]
        edges = np.stack((z, np.roll(z, shift=-1, axis=1)), axis=-1).reshape(-1, 2)
        z1, z2 = edges[:, 0], edges[:, 1]

        # 4. Normalisation des arêtes (Unidirectionnel)
        # Pour identifier les doublons, l'arête A->B doit être vue comme identique à B->A
        # On force le point avec la plus petite partie réelle/imaginaire en premier
        mask_swap = (z1.real > z2.real) | ((z1.real == z2.real) & (z1.imag > z2.imag))
        z_min = np.where(mask_swap, z2, z1)
        z_max = np.where(mask_swap, z1, z2)
        undirected = np.stack((z_min, z_max), axis=-1)

        # 5. Filtrage topologique
        # np.unique compte les occurrences de chaque segment
        # Les arêtes internes apparaissent 2 fois (partagées), les arêtes de bord 1 seule fois
        _, idx, counts = np.unique(
            undirected, axis=0, return_index=True, return_counts=True
        )
        bound_idx = idx[counts == 1]  # On ne garde que les segments uniques

        if bound_idx.size == 0:
            return np.array([])

        # 6. Reconstruction de la boucle (Chaînage)
        # On crée un dictionnaire {Point de départ : Point d'arrivée} pour les segments du bord
        bz1, bz2 = z1[bound_idx], z2[bound_idx]
        next_v = {(v.real, v.imag): (n.real, n.imag) for v, n in zip(bz1, bz2)}

        # On suit la chaîne pour remettre les points dans l'ordre du périmètre
        start_v = next(iter(next_v))
        curr_v = start_v
        ordered = []

        for _ in range(len(next_v) + 10):
            ordered.append(curr_v)
            curr_v = next_v.get(curr_v)
            if curr_v == start_v or curr_v is None:
                break

        return np.array(ordered)

    def get_local_pin_offsets(self, n_rings, clearance=0.05):
        """Calcule et met en cache les décalages relatifs (dx, dy) des crayons."""
        # Optimisation par mémorisation (Cache).
        # Le calcul de la position relative des crayons est identique pour tous
        # les assemblages d'un même type. On stocke le résultat pour éviter de
        # le recalculer à chaque itération lors de la génération du maillage.
        if n_rings in self._pins_cache:
            return self._pins_cache[n_rings]

        # Cas limite : aucun crayon (assemblage vide).
        if n_rings < 1:
            self._pins_cache[n_rings] = (np.empty((0, 2)), 0)
            return self._pins_cache[n_rings]

        # Géométrie interne de l'hexagone.
        # L'apothème est le rayon du cercle inscrit (distance du centre au milieu d'un bord).
        apotheme = self.R_hex * np.sqrt(3) / 2

        # Le "pitch" est la distance d'entraxe entre deux crayons.
        # Il est calculé pour que les crayons remplissent l'assemblage sans déborder,
        # en laissant une marge de sécurité (clearance) sur les bords.
        pitch = (
            self.R_hex * 0.5 if n_rings == 1 else apotheme / (n_rings - 0.5 + clearance)
        )

        # Le rayon d'un crayon est défini à 45% du pitch pour éviter tout chevauchement.
        pin_radius = pitch * 0.45

        # Construction de la grille locale des crayons.
        r_max = n_rings - 1
        a, b = np.arange(-r_max, r_max + 1), np.arange(-r_max, r_max + 1)

        # Comme pour les hexagones, on génère l'espace total des indices possibles.
        A, B = np.meshgrid(a, b)
        A_f, B_f = A.ravel(), B.ravel()

        # Filtrage hexagonal interne.
        # En coordonnées hexagonales obliques, un hexagone est défini par la condition :
        # max(|a|, |b|, |a+b|) <= Rayon_discret.
        # np.maximum.reduce effectue cette vérification vectoriellement sur les 3 axes.
        mask = np.maximum.reduce([np.abs(A_f), np.abs(B_f), np.abs(A_f + B_f)]) <= r_max
        A_v, B_v = A_f[mask], B_f[mask]

        # Transformation de la grille locale (Indices -> Cartésien relatif).
        # Même logique de pavage que pour le cœur, mais appliquée à l'échelle du crayon.
        X_offset = pitch * (A_v + B_v / 2)
        Y_offset = pitch * np.sqrt(3) / 2 * B_v

        # Empilage et formatage des coordonnées.
        # np.round stabilise numériquement les positions pour la CAO.
        offsets = np.round(np.column_stack((X_offset, Y_offset)), 6)

        # Sauvegarde dans le dictionnaire de cache.
        self._pins_cache[n_rings] = (offsets, pin_radius)

        return offsets, pin_radius

    def get_homogeneous_pins_symmetric(self, cx, cy, n_rings, clearance=0.05):
        """Fonction de compatibilité pour l'interface graphique (Aperçu Matplotlib)."""
        # Récupère les positions relatives depuis la fonction vectorisée et le cache.
        offsets, pin_radius = self.get_local_pin_offsets(n_rings, clearance)
        if offsets.size == 0:
            return np.array([]), 0

        # Translation spatiale (Broadcasting NumPy).
        # On additionne le vecteur centre [cx, cy] à l'ensemble de la matrice (N, 2).
        # Cela replace instantanément tous les crayons de cet assemblage dans le repère global.
        return offsets + np.array([cx, cy]), pin_radius

    def get_reflector_patch(self, R_reflector):
        """Instancie le patch vectoriel du réflecteur avec son évidement."""
        # Validation géométrique : le réflecteur doit physiquement contenir le cœur.
        if R_reflector <= self.R_n:
            raise ValueError(
                f"Le rayon du réflecteur ({R_reflector}) doit englober le noyau."
            )

        # 1. Construction de la frontière extérieure (Cylindre).
        # Génération discrète d'un cercle via les coordonnées polaires (128 points).
        theta = np.linspace(0, 2 * np.pi, 128)
        outer_verts = np.column_stack(
            (R_reflector * np.cos(theta), R_reflector * np.sin(theta))
        )
        outer_path = mpath.Path(outer_verts)

        # 2. Construction de la frontière intérieure (Évidement).
        # On récupère le polygone complexe généré par le filtrage topologique du cœur.
        inner_verts = self.get_outer_perimeter()

        if inner_verts.ndim == 2 and len(inner_verts) > 0:
            # Règle topologique du bobinage (Winding rule).
            # Pour que Matplotlib comprenne que la forme interne est un "trou",
            # il faut que le chemin intérieur soit parcouru dans le sens inverse
            # du chemin extérieur. L'opération [::-1] inverse l'ordre des sommets.
            inner_path = mpath.Path(inner_verts[::-1])
            compound_path = mpath.Path.make_compound_path(outer_path, inner_path)
        else:
            # Fallback de sécurité si le périmètre interne échoue.
            compound_path = outer_path

        # Création et retour de l'objet graphique prêt à être tracé.
        return mpatches.PathPatch(
            compound_path,
            facecolor="lightgrey",
            edgecolor="black",
            linewidth=1.5,
            hatch="//",
            alpha=0.6,
            zorder=0,
        )

    def get_tagged_assemblies(self, n_cr_rings=1, cr_density=0.5):
        """Assigne les tags (FUEL/CR) via masquage booléen intégral."""
        centers = self.hex_centers()
        if centers.size == 0:
            return centers, np.array([]), np.array([])

        X, Y = centers[:, 0], centers[:, 1]

        # 1. Transformation géométrique inverse (Cartésien -> Indices spatiaux).
        # On recalcule les indices (A, B) à partir des positions physiques (X, Y).
        # Cela permet d'analyser la structure du cœur sans dépendre de la grille initiale.
        B = np.round(Y / (1.5 * self.R_hex))
        A = np.round(X / (self.R_hex * np.sqrt(3)) - B / 2)

        # 2. Distance de Manhattan hexagonale.
        # Calcule le "rang" (la distance discrète) de chaque hexagone par rapport
        # à l'assemblage central (0,0). Le centre est au rang 0, les voisins au rang 1, etc.
        d_hex = np.maximum.reduce([np.abs(A), np.abs(B), np.abs(A + B)]).astype(int)

        # Préparation du tri spatial pour la distribution des barres de contrôle (CR).
        angles = np.arctan2(Y, X)
        D_max = np.max(d_hex) if d_hex.size > 0 else 0

        # Initialisation : on considère que tout le cœur est du combustible par défaut.
        tags = np.full(len(centers), "FUEL", dtype=object)

        # 3. Placement vectorisé des barres de contrôle (Control Rods).
        if n_cr_rings > 0 and D_max > 0:
            # On détermine sur quels anneaux placer les barres.
            # Espacement régulier : on divise le rayon total en fractions.
            step = D_max / (n_cr_rings + 1)
            target_rings = np.round(step * np.arange(1, n_cr_rings + 1)).astype(int)

            for target_d in target_rings:
                # Masquage : on isole les index des assemblages situés sur l'anneau cible.
                indices = np.where(d_hex == target_d)[0]
                if indices.size == 0:
                    continue

                # Tri angulaire.
                # Essentiel pour répartir les barres de manière symétrique autour du centre.
                # On ordonne les indices selon leur position angulaire (de -pi à pi).
                sorted_indices = indices[np.argsort(angles[indices])]

                # Application de la densité.
                if cr_density >= 1.0:
                    # Remplacement total : tout l'anneau devient un anneau de barres.
                    tags[sorted_indices] = "CR"
                elif cr_density > 0:
                    # Échantillonnage fractionné (Slicing avancé).
                    # On calcule l'écart (jump) pour obtenir la proportion désirée.
                    # Exemple : cr_density=0.33 -> jump=3 -> 1 barre tous les 3 assemblages.
                    jump = int(np.round(1.0 / cr_density))
                    tags[sorted_indices[::jump]] = "CR"

        # On retourne les centres, leurs matériaux assignés, et la matrice des rangs.
        return centers, tags, d_hex


# ==============================================================================
# 2. GÉNÉRATEUR DE MAILLAGE GMSH (Noyau OpenCASCADE)
# ==============================================================================
class ReactorMeshGenerator:
    def __init__(self, geometry: ReactorGeometry, params: dict):
        self.geom = geometry
        self.params = params

    def _add_hexagon(self, cx, cy, radius, occ):
        """Construit une surface hexagonale dans le noyau OpenCASCADE."""
        pts = []
        # 1. Définition des sommets (Géométrie de base)
        for i in range(6):
            angle = np.pi / 6 + i * np.pi / 3
            px = cx + radius * np.cos(angle)
            py = cy + radius * np.sin(angle)
            # Ajout du point dans l'espace CAO (z=0 pour la 2D)
            pts.append(occ.addPoint(px, py, 0))

        # 2. Construction de la topologie filaire (Arêtes)
        # On relie les points consécutifs pour former le contour.
        lines = [occ.addLine(pts[i], pts[(i + 1) % 6]) for i in range(6)]

        # 3. Fermeture topologique (CurveLoop) et création de la surface
        # Une CurveLoop garantit que le contour est fermé et orienté.
        cl = occ.addCurveLoop(lines)
        return occ.addPlaneSurface([cl])

    def generate(self, output_filename="output/meshes/reactor.msh"):
        """Construit la géométrie CAD, applique la fragmentation et exporte le maillage."""

        # --- Gestion robuste de l'API Gmsh ---
        # Nettoyage de l'état global de Gmsh en cas d'appels successifs ou d'échec précédent.
        if gmsh.isInitialized():
            gmsh.clear()
            gmsh.finalize()

        try:
            gmsh.initialize()
        except ValueError:
            # Contournement d'un problème connu d'interaction entre le module 'signal'
            # de Python et l'initialisation de Gmsh dans des threads secondaires.
            pass

        gmsh.option.setNumber(
            "General.Terminal", 0
        )  # Désactive la sortie standard de Gmsh
        gmsh.model.add("ReactorCore")
        occ = gmsh.model.occ

        # Récupération des paramètres géométriques globaux
        R_hex = self.params["R_hex"]
        R_reflec = self.params["R_reflec"]

        # Récupération de la carte topologique du cœur (Centres et types d'assemblages)
        centers, tags, _ = self.geom.get_tagged_assemblies(
            n_cr_rings=self.params["cr_rings"], cr_density=self.params["cr_density"]
        )

        # Récupération des positions relatives pré-calculées pour les crayons
        offsets_fuel, r_fuel = self.geom.get_local_pin_offsets(self.params["pins_fuel"])
        offsets_cr, r_cr = self.geom.get_local_pin_offsets(self.params["pins_cr"])

        hex_tags = []
        pin_tags_fuel = []
        pin_tags_cr = []

        # Création du domaine global (Le réflecteur agit ici comme la matrice englobante)
        reflector_tag = occ.addDisk(0, 0, 0, R_reflec, R_reflec)

        # 1. Instanciation des entités OpenCASCADE
        for i, (cx, cy) in enumerate(centers):
            tag_type = tags[i]
            # Création de l'enveloppe de l'assemblage (Modérateur inter-crayons)
            hex_tags.append(self._add_hexagon(cx, cy, R_hex, occ))

            # Création des inclusions cylindriques (Crayons)
            if tag_type == "FUEL":
                for dx, dy in offsets_fuel:
                    pin_tags_fuel.append(
                        occ.addDisk(cx + dx, cy + dy, 0, r_fuel, r_fuel)
                    )
            elif tag_type == "CR":
                for dx, dy in offsets_cr:
                    pin_tags_cr.append(occ.addDisk(cx + dx, cy + dy, 0, r_cr, r_cr))

        # 2. Fragmentation Booléenne (Étape critique pour les Éléments Finis)
        # La fonction 'fragment' calcule les intersections entre toutes les entités.
        # Elle découpe le grand disque du réflecteur avec les hexagones, et évide
        # les hexagones pour y placer les crayons.
        # Objectif : Obtenir un maillage CONFORME (les nœuds à la frontière entre
        # le combustible et le modérateur seront partagés par les deux domaines).
        all_hex_tuples = [(2, t) for t in hex_tags]
        all_pin_tuples = [(2, t) for t in pin_tags_fuel + pin_tags_cr]
        occ.fragment([(2, reflector_tag)], all_hex_tuples + all_pin_tuples)
        occ.synchronize()  # Synchronise le modèle mathématique (OCC) avec le modèle de maillage Gmsh.

        # 3. Assignation des Groupes Physiques (Physical Groups)
        # Nécessaire pour que le solveur puisse associer des propriétés matériaux
        # (sections efficaces) et des conditions aux limites aux éléments du maillage.
        pg_reflector, pg_moderator, pg_fuel_pins, pg_cr_pins = [], [], [], []

        # 3.a Identification spatiale des Surfaces (Matériaux)
        surfaces = gmsh.model.getEntities(dim=2)
        for dim, tag in surfaces:
            # Analyse de la boîte englobante (Bounding Box) pour caractériser la surface
            bbox = gmsh.model.getBoundingBox(dim, tag)
            max_extent = max(abs(bbox[0]), abs(bbox[1]), abs(bbox[3]), abs(bbox[4]))

            # Heuristique 1 : Le réflecteur est la seule entité s'étendant à la périphérie externe.
            if max_extent > self.geom.R_n + self.geom.R_hex:
                pg_reflector.append(tag)
                continue

            com = occ.getCenterOfMass(dim, tag)
            dx = bbox[3] - bbox[0]  # Largeur de l'entité

            # Heuristique 2 : Discrimination Crayon vs Hexagone (Modérateur) via la taille.
            # Si l'entité est plus petite qu'un hexagone, c'est un crayon.
            if dx < R_hex:
                # Association du crayon à son assemblage parent via la distance au centre le plus proche.
                distances = np.sqrt(
                    (centers[:, 0] - com[0]) ** 2 + (centers[:, 1] - com[1]) ** 2
                )
                nearest_idx = np.argmin(distances)

                # Répartition selon le type de l'assemblage parent
                if tags[nearest_idx] == "FUEL":
                    pg_fuel_pins.append(tag)
                else:
                    pg_cr_pins.append(tag)
            else:
                # Si l'entité n'est ni le réflecteur ni un crayon, c'est le modérateur intra-assemblage.
                pg_moderator.append(tag)

        # 3.b Identification spatiale des Lignes (Conditions aux Limites)
        lines = gmsh.model.getEntities(dim=1)
        outer_boundary = []
        for dim, tag in lines:
            bbox = gmsh.model.getBoundingBox(dim, tag)
            max_extent = max(abs(bbox[0]), abs(bbox[1]), abs(bbox[3]), abs(bbox[4]))

            # Extraction des segments frontières dont les coordonnées atteignent le rayon extérieur.
            # Tolérance de 1e-3 pour absorber les erreurs d'arrondi géométrique.
            if abs(max_extent - R_reflec) < 1e-3:
                outer_boundary.append(tag)

        # Enregistrement formel des Physical Groups dans la base de données Gmsh.
        # Les tags entiers (100, 200...) seront les identifiants lus par le solveur.
        gmsh.model.addPhysicalGroup(2, pg_reflector, 100)
        gmsh.model.setPhysicalName(2, 100, "Reflector")

        gmsh.model.addPhysicalGroup(2, pg_moderator, 200)
        gmsh.model.setPhysicalName(2, 200, "Moderator")

        gmsh.model.addPhysicalGroup(2, pg_fuel_pins, 300)
        gmsh.model.setPhysicalName(2, 300, "Fuel")

        gmsh.model.addPhysicalGroup(2, pg_cr_pins, 400)
        gmsh.model.setPhysicalName(2, 400, "ControlRods")

        gmsh.model.addPhysicalGroup(1, outer_boundary, 1000)
        gmsh.model.setPhysicalName(1, 1000, "OuterBoundary")

        # 4. Paramétrage avancé des tailles de mailles (Gmsh Fields)

        # On définit les tailles extrêmes (très fin pour le combustible, très large pour le réflecteur)
        size_min = r_fuel / 4.0 if r_fuel > 0 else 0.05
        size_max = R_hex / 1.5

        gmsh.option.setNumber("Mesh.MeshSizeMin", size_min)
        gmsh.option.setNumber("Mesh.MeshSizeMax", size_max)

        # --- 4.1. Champ de Distance ---
        # Gmsh va calculer la distance entre chaque point de l'espace et les surfaces des crayons.
        gmsh.model.mesh.field.add("Distance", 1)
        all_pins = pg_fuel_pins + pg_cr_pins
        if all_pins:
            gmsh.model.mesh.field.setNumbers(1, "SurfacesList", all_pins)

        # --- 4.2. Champ de Seuil (Threshold) ---
        # Ce champ traduit la distance (calculée au-dessus) en taille de triangle, créant le gradient.
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(
            2, "InField", 1
        )  # Se base sur la distance aux crayons
        gmsh.model.mesh.field.setNumber(
            2, "SizeMin", size_min
        )  # Taille à l'intérieur et juste autour des crayons
        gmsh.model.mesh.field.setNumber(
            2, "SizeMax", size_max
        )  # Taille loin dans le réflecteur
        gmsh.model.mesh.field.setNumber(
            2, "DistMin", r_fuel * 0.2
        )  # Jusqu'à cette distance, le maillage reste ultra-fin
        gmsh.model.mesh.field.setNumber(
            2, "DistMax", R_hex * 1.5
        )  # Distance sur laquelle le triangle grossit progressivement

        # On active ce champ comme "règle absolue" pour générer le maillage
        gmsh.model.mesh.field.setAsBackgroundMesh(2)

        # --- 4.3. Optimisations Gmsh ---
        # On désactive la propagation par défaut pour que seul notre gradient dicte la loi
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)

        # On garde l'adaptation à la courbure pour assurer que les ronds soient parfaits
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
        gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 16)

        # 5. Génération et Exportation
        # Appel du moteur de maillage 2D de Gmsh (algorithmes de Delaunay/Frontal).
        gmsh.model.mesh.generate(2)

        # Sauvegarde sur disque de manière sécurisée (création du dossier si inexistant).
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        gmsh.write(output_filename)

        # Libération de la mémoire allouée par l'API C++ sous-jacente.
        gmsh.finalize()
