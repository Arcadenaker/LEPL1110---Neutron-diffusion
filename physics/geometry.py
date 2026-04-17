import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
import matplotlib.path as mpath
import matplotlib.patches as mpatches


class ReactorGeometry:
    def __init__(self, R_n, R_hex, form="circle"):
        self.form = form
        self.R_hex = R_hex
        self.R_n = R_n

    def hex_centers(self):
        """
        Génère les centres des hexagones contenus dans un cercle (Optimisé NumPy).
        """
        # 1. Définition des bornes globales pour englober tout le cercle
        b_lim = int(np.ceil(2 * self.R_n / (3 * self.R_hex)))

        # L'index 'a' maximum dépend de R_n et du décalage b/2
        a_lim = int(np.ceil(self.R_n / (self.R_hex * np.sqrt(3)) + b_lim / 2))

        # 2. Création de la grille 2D
        a_vals = np.arange(-a_lim, a_lim + 1)
        b_vals = np.arange(-b_lim, b_lim + 1)
        A, B = np.meshgrid(a_vals, b_vals)
        A_flat, B_flat = A.ravel(), B.ravel()

        # 3. Calcul vectoriel de toutes les coordonnées
        X = self.R_hex * np.sqrt(3) * (A_flat + B_flat / 2)
        Y = self.R_hex * 1.5 * B_flat

        # 4. Application du masque circulaire
        mask = (X**2 + Y**2) <= (self.R_n**2 + 1e-9)

        # Extraction et conversion en liste de tuples
        centers = np.column_stack((X[mask], Y[mask]))
        return [tuple(c) for c in centers]

    def get_outer_perimeter(self, precision=5):
        """
        Extrait la frontière externe par analyse topologique des arêtes.
        Garantit l'absence de boucle infinie et une orientation cohérente.
        """
        centers = self.hex_centers()
        edge_counts = {}
        directed_edges = {}

        # 1. Collecte de toutes les arêtes de tous les hexagones
        for cx, cy in centers:
            # Génération des 6 sommets dans le sens anti-horaire
            angles = [np.pi / 6 + i * np.pi / 3 for i in range(6)]
            verts = [
                (
                    round(cx + self.R_hex * np.cos(a), precision),
                    round(cy + self.R_hex * np.sin(a), precision),
                )
                for a in angles
            ]

            for i in range(6):
                v1 = verts[i]
                v2 = verts[(i + 1) % 6]

                # Clé non-orientée pour compter le nombre d'occurrences
                undirected_edge = tuple(sorted((v1, v2)))
                edge_counts[undirected_edge] = edge_counts.get(undirected_edge, 0) + 1

                # Sauvegarde de l'orientation originale de l'arête
                directed_edges[undirected_edge] = (v1, v2)

        # 2. Filtrage : les arêtes d'occurrence 1 forment la frontière
        next_vertex = {}
        for undirected_edge, count in edge_counts.items():
            if count == 1:
                v1, v2 = directed_edges[undirected_edge]
                # Le dictionnaire mappe (sommet_départ -> sommet_arrivée)
                next_vertex[v1] = v2

        if not next_vertex:
            return []

        # 3. Chaînage pour construire le polygone continu et ordonné
        ordered_vertices = []
        start_v = list(next_vertex.keys())[0]  # Point de départ arbitraire sur le bord
        current_v = start_v

        # Sécurité absolue : on ne boucle pas plus de fois qu'il n'y a d'arêtes
        max_iters = len(next_vertex) + 10
        iters = 0

        while iters < max_iters:
            ordered_vertices.append(current_v)
            current_v = next_vertex.get(current_v)

            # Condition de fin : la boucle géométrique est refermée
            if current_v == start_v or current_v is None:
                break
            iters += 1

        return ordered_vertices

    def get_homogeneous_pins(self, cx, cy, n_rings, clearance=0.05):
        """
        Génère les coordonnées des centres de crayons selon un treillis hexagonal régulier.

        Paramètres:
        - cx, cy : centre de l'assemblage hexagonal.
        - n_rings : nombre de couronnes (1 = centre seul, 2 = 7 crayons, 3 = 19 crayons, etc.).
        - clearance : marge entre le bord de l'hexagone et les crayons extérieurs (fraction du pas).
        """
        if n_rings < 1:
            return [], 0

        # Le nombre total de crayons sera 1 + 3 * n_rings * (n_rings - 1)

        # L'apothème de l'hexagone (distance centre -> milieu d'un côté)
        apotheme = self.R_hex * np.sqrt(3) / 2

        # Calcul du pas (pitch) pour que les n_rings rentrent dans l'apothème
        # n_rings - 1 est le nombre d'intervalles depuis le centre jusqu'à la couronne externe
        # On ajoute une marge (clearance) pour ne pas toucher le bord
        if n_rings == 1:
            pitch = self.R_hex * 0.5
        else:
            pitch = apotheme / ((n_rings - 1) + 0.5 + clearance)

        pin_radius = pitch / 2 * 0.9  # 0.9 pour éviter que les cercles se touchent

        pins = []
        # Couronne 0 (Centre)
        pins.append((cx, cy))

        # Couronnes suivantes
        for ring in range(1, n_rings):
            # On part du coin "en haut à droite" de la couronne
            x = cx + ring * pitch
            y = cy

            # Les 6 directions pour parcourir la couronne hexagonale
            # Angles : 120°, 180°, 240°, 300°, 0°, 60°
            angles = [2 * np.pi / 3, np.pi, 4 * np.pi / 3, 5 * np.pi / 3, 0, np.pi / 3]

            for angle in angles:
                dx = pitch * np.cos(angle)
                dy = pitch * np.sin(angle)
                for _ in range(ring):
                    pins.append((x, y))
                    x += dx
                    y += dy

        # Nettoyage des petites imprécisions flottantes
        pins = [(round(px, 6), round(py, 6)) for px, py in pins]

        return pins, pin_radius

    def get_reflector_patch(self, R_reflector):
        """
        Génère un patch Matplotlib pour le réflecteur avec un trou central
        correspondant au périmètre des assemblages hexagonaux.
        """
        # 1. Vérification physique basique
        if R_reflector <= self.R_n:
            raise ValueError(
                f"Le rayon du réflecteur ({R_reflector}) doit englober le rayon du noyau ({self.R_n})."
            )

        # 2. Construction de la frontière extérieure (Cercle lisse)
        theta = np.linspace(0, 2 * np.pi, 128)
        outer_verts = np.column_stack(
            (R_reflector * np.cos(theta), R_reflector * np.sin(theta))
        )
        outer_path = mpath.Path(outer_verts)

        # 3. Construction de la frontière intérieure (Périmètre des hexagones)
        inner_verts = np.array(self.get_outer_perimeter())

        # Inversion de l'ordre des sommets intérieurs.
        # Règle topologique : pour qu'une surface soit évidée, son contour intérieur
        # doit être parcouru dans le sens inverse du contour extérieur.
        inner_verts = inner_verts[::-1]
        inner_path = mpath.Path(inner_verts)

        # 4. Création du chemin composé
        compound_path = mpath.Path.make_compound_path(outer_path, inner_path)

        # 5. Instanciation du rendu visuel
        patch = mpatches.PathPatch(
            compound_path,
            facecolor="lightgrey",  # Couleur distincte pour le réflecteur
            edgecolor="black",
            linewidth=1.5,
            hatch="//",  # Hachures pour identifier la zone solide
            alpha=0.6,
            zorder=0,  # Z-order faible pour rester en arrière-plan
        )
        return patch


if __name__ == "__main__":
    import time

    print("[1/6] Initialisation des paramètres...")
    R_hexagone = 2.0
    R_noyau = 10.0
    R_reflec = 14.0
    n_couronnes = 3

    t0 = time.time()
    geom = ReactorGeometry(R_n=R_noyau, R_hex=R_hexagone)

    print("[2/6] Calcul des centres des assemblages hexagonaux...")
    centres = geom.hex_centers()
    print(f"      -> {len(centres)} assemblages trouvés.")

    print("[3/6] Initialisation de la figure Matplotlib...")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.add_patch(
        Circle((0, 0), R_noyau, color="red", fill=False, linestyle="--", zorder=2)
    )

    print("[4/6] Génération et tracé du réflecteur...")
    try:
        reflector_patch = geom.get_reflector_patch(R_reflector=R_reflec)
        ax.add_patch(reflector_patch)
        print("      -> Réflecteur généré avec succès.")
    except Exception as e:
        print(f"      -> ERREUR lors de la génération du réflecteur : {e}")

    print(f"[5/6] Tracé des hexagones et des crayons ({len(centres)} itérations)...")
    for i, (cx, cy) in enumerate(centres):
        hex_patch = RegularPolygon(
            (cx, cy),
            numVertices=6,
            radius=R_hexagone,
            orientation=0,
            fill=False,
            edgecolor="blue",
            alpha=0.8,
            zorder=2,
        )
        ax.add_patch(hex_patch)

        pins, pin_r = geom.get_homogeneous_pins(cx, cy, n_couronnes)

        for px, py in pins:
            ax.add_patch(
                Circle((px, py), pin_r, color="purple", fill=True, alpha=0.7, zorder=3)
            )

        # Suivi de la progression dans la boucle
        if (i + 1) % 10 == 0 or (i + 1) == len(centres):
            print(f"      -> {i + 1}/{len(centres)} assemblages traités...")

    print("[6/6] Préparation de l'affichage...")
    ax.set_aspect("equal")
    ax.set_xlim(-R_reflec * 1.1, R_reflec * 1.1)
    ax.set_ylim(-R_reflec * 1.1, R_reflec * 1.1)
    ax.set_title("Géométrie du Cœur avec Réflecteur")

    t1 = time.time()
    print(f"      -> Temps total de préparation : {t1 - t0:.3f} secondes.")
    print(
        "Lancement de plt.show() (Le script est en pause tant que la fenêtre est ouverte)..."
    )

    plt.show()

    print("Fenêtre fermée. Fin du script.")
