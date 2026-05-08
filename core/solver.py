import numpy as np
import meshio
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

from core.boundary_cond import get_dirichlet_nodes
from core.time_integration import TimeIntegrator
from core.assembly import assemble_mass_or_reaction, assemble_stiffness
from physics.materials import get_material_properties

# --- IMPORT DU LOGGER ---
from utils.logger import get_logger

logger = get_logger(__name__)
# ------------------------


def extract_p1_fem_data(mesh):
    """
    Construit la transformation Isoparamétrique P1

    En éléments finis, intégrer les équations de diffusion directement sur des triangles
    de formes et tailles aléatoires est mathématiquement trop lourd.
    La solution standard est de tout ramener à un "Triangle de Référence" parfait
    (coordonnées 0.0, 0.5, 1.0)
    """
    logger.debug("Extraction des données P1 FEM depuis le maillage...")
    triangles = mesh.cells_dict["triangle"]
    ne = len(triangles)
    points = mesh.points

    # Points d'intégration de Gauss (ngp = 3) et leurs poids (w)
    # Ils permettent de calculer l'intégrale exacte des polynômes de degré 2 sur le triangle.
    ngp = 3
    w = np.array([1 / 6, 1 / 6, 1 / 6])
    N = np.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]])

    # Gradients des fonctions de forme sur le triangle de référence.
    # Ces valeurs sont constantes car le triangle de référence ne change jamais.
    gradN_ref = np.zeros((ngp, 3, 2))
    for g in range(ngp):
        gradN_ref[g, 0, :] = [-1.0, -1.0]
        gradN_ref[g, 1, :] = [1.0, 0.0]
        gradN_ref[g, 2, :] = [0.0, 1.0]

    jacobians = np.zeros((ne, ngp, 2, 2))
    dets = np.zeros((ne, ngp))

    for e in range(ne):
        p = points[triangles[e]]
        # Vecteurs directeurs du vrai triangle
        v1 = p[1] - p[0]
        v2 = p[2] - p[0]

        # LA MATRICE JACOBIENNE
        # Elle contient les dérivées spatiales. Elle sert de transformation linéaire
        # pour passer de l'espace réel (x, y) à l'espace de référence.
        J = np.array([[v1[0], v2[0]], [v1[1], v2[1]]])

        # Le déterminant du Jacobien représente l'aire du vrai triangle
        # (à un facteur 1/2 près) par rapport au triangle de référence.
        det_J = abs(v1[0] * v2[1] - v1[1] * v2[0])

        for g in range(ngp):
            jacobians[e, g] = J
            dets[e, g] = det_J

    logger.debug(f"Extraction terminée pour {ne} éléments triangulaires.")
    return triangles, dets, w, N, jacobians, gradN_ref


def pos_rodBar_init(
    mesh, conn, det, w, N, get_props_func, K, nn, user_mapping, noeuds_bords
):
    """
    Recherche algorithmique de l'insertion optimale initiale des barres de contrôle par Dichotomie.

    Cette fonction permet d'aider le contrôleur PD à démarrer proche de l'équilibre
    pour éviter de trop fortes oscillations initiales (transitoire violent).
    """
    logger.info("Recherche du point d'équilibre initial des barres de contrôle")

    # --- 1. IDENTIFICATION DES DEGRÉS DE LIBERTÉ (DOFs) ---
    # On isole les nœuds internes (libres) en excluant les frontières de Dirichlet
    mask = np.ones(nn, dtype=bool)
    mask[noeuds_bords] = False
    free_dofs = np.nonzero(mask)[0]

    # Extraction de la sous-matrice de rigidité (fuites) pour les noeuds libres
    K_FF = K[free_dofs, :][:, free_dofs]

    # --- 2. APPROXIMATION DU FLUX FONDAMENTAL ---
    # L'utilisation d'un flux plat génère des fuites artificielles infinies aux bords.
    # On le remplace par un profil parabolique (proche du mode fondamental de Bessel J0 pour un cylindre),
    # maximal au centre et s'annulant aux frontières du réacteur.

    # Récupération des coordonnées spatiales (X, Y) des noeuds libres
    pts_free = mesh.points[free_dofs]
    x = pts_free[:, 0]
    y = pts_free[:, 1]

    # Calcul de la distance radiale au carré : r^2 = x^2 + y^2
    r_carre = x**2 + y**2
    R_max_carre = np.max(r_carre)

    # Génération du profil parabolique : 1.0 au centre (0,0) et tend vers 0.0 au bord (R_max)
    phi_test = 1.0 - (r_carre / R_max_carre)

    # Sécurité numérique : on évite d'avoir des zéros parfaits pour ne pas fausser l'évaluation matricielle
    phi_test = np.maximum(phi_test, 1e-6)

    # --- 3. RECHERCHE PAR DICHOTOMIE ---
    # Bornes de recherche de la position de la barre (0.0 = totalement retirée, 1.0 = totalement insérée)
    pos_min = 0.0
    pos_max = 1.0
    pos_critique = 0.5

    # 10 itérations garantissent une précision de 1/2^10 = ~0.001 (0.1% de la course totale)
    for i in range(10):
        pos_critique = (pos_min + pos_max) / 2.0

        # Récupération des sections efficaces mises à jour pour la position actuelle
        _, c_Sigma_a, c_nuSigma_f, _ = get_props_func(
            mesh, conn, rod_insertion=pos_critique, user_mapping=user_mapping
        )

        # La composante locale de réaction correspond à la création (nu*Sigma_f) moins l'absorption (Sigma_a)
        c_R = c_nuSigma_f - c_Sigma_a

        # Assemblage de la matrice de réaction globale, puis réduction aux DOFs libres
        R = assemble_mass_or_reaction(nn, len(conn), 3, len(w), conn, det, w, N, c_R)
        R_FF = R[free_dofs, :][:, free_dofs]

        # Matrice d'évolution A = Opérateur de Réaction (R) - Opérateur de Fuite/Diffusion (K)
        A_FF = R_FF - K_FF

        # Évaluation instantanée du bilan neutronique spatial
        # On multiplie la matrice d'évolution par notre flux spatial de test parabolique
        variation_flux = A_FF.dot(phi_test)
        bilan_neutronique = np.sum(variation_flux)

        # Ajustement des bornes selon le signe de l'évolution du flux
        if bilan_neutronique > 0:
            # Sur-critique (la puissance monte) : on doit insérer de l'absorbant (augmenter pos_min)
            pos_min = pos_critique
        else:
            # Sous-critique (la puissance descend) : on doit retirer de l'absorbant (diminuer pos_max)
            pos_max = pos_critique

    logger.info(f"Position d'équilibre trouvée : {pos_critique:.3f}")
    return pos_critique


def run_full_simulation(
    mesh_path, user_mapping=None, headless=False, save_csv=None, kd_value=0.50
):
    """
    Fonction principale du simulateur.
    Elle lit la géométrie, calibre la puissance mathématique sur une échelle physique réelle,
    lance l'intégration temporelle et anime les résultats.
    """
    logger.info(f"Démarrage de run_full_simulation sur : {mesh_path}")
    print(f"Chargement du maillage : {mesh_path}")

    # --- 1. LECTURE DU MAILLAGE ---
    try:
        # Meshio lit le fichier .msh généré par Gmsh et extrait les coordonnées des points
        mesh = meshio.read(mesh_path)
    except Exception as e:
        logger.error(
            f"Erreur lors de la lecture du maillage {mesh_path} : {e}", exc_info=True
        )
        raise

    # nn=(Number of Nodes) est la dimension de nos futures matrices (M, K, R).
    # Chaque nœud du maillage représente une inconnue spatiale pour le flux neutronique.
    nn = len(mesh.points)
    logger.debug(f"Maillage chargé. Nombre de nœuds (Inconnues) : {nn}")

    # --- EXTRACTION GÉOMÉTRIQUE ---
    # On récupère les matrices Jacobiennes et leurs déterminants.
    # C'est la base de la méthode P1 : on passe d'un vrai triangle déformé à un triangle parfait de référence[cite: 3].
    conn, det, w, N, jacobians, gradN_ref = extract_p1_fem_data(mesh)

    # --- CALCUL DE LA PUISSANCE MAXIMALE ---
    # Pour afficher une puissance en MégaWatts (MW)
    # le code va chercher/calculer directement les valeurs qu'il a besoin
    # dans le fichier GMSH

    # En éléments finis, l'aire géométrique exacte d'un triangle se calcule en intégrant
    # le déterminant du Jacobien ('det') sur les points de Gauss.
    # Ici, nos poids de Gauss ('w') valent tous 1/6. On somme donc det * 1/6 sur les 3 points.
    aires_triangles = np.sum(det, axis=1) * (1.0 / 6.0)
    aire_combustible_cm2 = 0.0

    # Gmsh stocke les triangles en "blocs" contigus
    # Pour retrouver l'index global absolu d'un triangle, on doit calculer des décalages (offsets)
    triangle_offsets = {}
    current_offset = 0
    for block_id, cell_block in enumerate(mesh.cells):
        if cell_block.type == "triangle":
            triangle_offsets[block_id] = current_offset
            current_offset += len(cell_block.data)

    # On isole la zone Fuel (Combustible) car c'est la seule région qui dégage de la chaleur
    if "Fuel" in mesh.cell_sets:
        for block_id, elem_indices in enumerate(mesh.cell_sets["Fuel"]):
            # Si le bloc contient bien des triangles (et pas des lignes de bordure)
            if len(elem_indices) > 0 and mesh.cells[block_id].type == "triangle":
                # On ajoute le décalage pour avoir l'index global correct
                g_idx = elem_indices + triangle_offsets[block_id]
                # On additionne l'aire de tous les triangles de combustible
                aire_combustible_cm2 += np.sum(aires_triangles[g_idx])

    # Le maillage 2D est plat. On simule physiquement une "tranche" d'un vrai réacteur.
    # On impose une hauteur virtuelle de 1 mètre (100 cm)
    HAUTEUR_CM = 100.0

    # Volume total d'Uranium (en cm³) = Aire 2D * Hauteur
    volume_combustible_cm3 = aire_combustible_cm2 * HAUTEUR_CM

    # Hypothèse thermohydraulique : 1 cm³ de combustible nucléaire génère environ 100 Watts de chaleur.
    DENSITE_PUISSANCE = 100.0
    PUISSANCE_MAX_WATTS = volume_combustible_cm3 * DENSITE_PUISSANCE

    # Conversion de Watts vers MégaWatts (division par 1 million)
    PUISSANCE_MAX_MW = PUISSANCE_MAX_WATTS / 1e6

    logger.info(
        f"Analyse géométrique : Volume de Combustible = {volume_combustible_cm3:.0f} cm3"
    )
    logger.info(
        f"Puissance thermique maximale estimée du cœur : {PUISSANCE_MAX_MW:.2f} MW"
    )

    # --- CALIBRATION DU MODÈLE MATHÉMATIQUE ---
    # Pour amorcer l'équation différentielle, on place une "graine" artificielle de 10 neutrons
    # sur chaque noeud (représentant les fissions spontanées ou la source de démarrage)
    phi_0 = np.ones(nn) * 10.0

    # Somme de Riemann numérique discrète : représente l'inventaire neutronique virtuel de notre modèle 2D.
    puissance_brute_initiale = np.sum(phi_0)

    # On relie les mathématiques à la physique :
    # On décrète que cet état mathématique de départ correspond à 50% de la puissance physique
    # maximale que peut supporter le volume calculé précédemment.
    PUISSANCE_INITIALE_MW = PUISSANCE_MAX_MW * 0.50

    # Le FACTEUR_MW est le secret du solveur. C'est un scalaire de normalisation dynamique.
    # Puisque l'équation de diffusion est linéaire, multiplier le flux calculé par ce facteur
    # n'altère en rien la dynamique (stabilité, transitoires), mais permet un affichage industriel (en MW).
    FACTEUR_MW = PUISSANCE_INITIALE_MW / puissance_brute_initiale

    # Scénario imposé : Le pilote automatique doit augmenter la puissance du réacteur jusqu'à 150%
    POURCENTAGE_CIBLE = 150
    PUISSANCE_CIBLE_MW = PUISSANCE_INITIALE_MW * (POURCENTAGE_CIBLE / 100.0)
    logger.info(
        f"Scénario PID : Montée en puissance demandée (De {PUISSANCE_INITIALE_MW:.2f} MW à {PUISSANCE_CIBLE_MW:.2f} MW)"
    )

    # -- LE CONTROLEUR PD --
    # Variable persistante pour calculer la dérivée (vitesse d'évolution de l'erreur)
    erreur_precedente = None

    def controleur_PD(phi_actuel, phi_precedent, position_barres, dt):
        nonlocal erreur_precedente

        # On convertit la matrice de flux en puissance de chaleur en MW
        puissance_mw = np.sum(phi_actuel) * FACTEUR_MW

        # Sécurité : On bloque la puissance minimale à un chiffre très petit
        # pour empêcher le logarithme (np.log) de crasher en cas d'extinction totale du flux
        p_safe_mw = max(puissance_mw, 1e-10)

        # Erreur Logarithmique : Le flux nucléaire évolue exponentiellement
        # En utilisant un rapport logarithmique au lieu d'une soustraction classique,
        # on rend la commande quasi-linéaire et on évite de saturer les actionneurs d'un coup sec.
        erreur = np.log(p_safe_mw / PUISSANCE_CIBLE_MW)

        if erreur_precedente is None:
            erreur_precedente = erreur

        # Action Dérivée (Oméga) : L'inverse de la Période du réacteur.
        # C'est un radar d'anticipation. Si ce terme est grand, la puissance grimpe trop vite,
        # et le contrôleur freinera (enfoncera les barres) avant même d'avoir atteint la cible !
        omega = (erreur - erreur_precedente) / dt

        # Loi de commande PD
        Kp = (
            0.15  # Gain proportionnel : la "force" de rappel vers la cible en MégaWatts
        )
        Kd = kd_value  # Gain dérivé : la constante pour amortir le mouvement et éviter un emballement

        # Vitesse demandée (en fraction de course de barre par seconde)
        vitesse_demandee = (Kp * erreur) + (Kd * omega)

        # Delta de position de la barre de contrôle pour ce petit pas de temps 'dt'
        delta_pos = vitesse_demandee * dt

        # Mécanique physique des barres de contrôle :
        # Une vraie barre de Boral ne peut pas se téléporter. On limite sa vitesse maximale à 40% de la hauteur par seconde.
        # Pour être réaliste
        vitesse_max = 0.40
        delta_pos = np.clip(delta_pos, -vitesse_max * dt, vitesse_max * dt)

        # Nouvelle position globale (bornée entre 0.0 (Levée = Eau) et 1.0 (Insérée = Boral pur))
        nouvelle_pos = np.clip(position_barres + delta_pos, 0.0, 1.0)

        # Sauvegarde en mémoire pour l'itération dt suivante
        erreur_precedente = erreur
        return nouvelle_pos

    print("Assemblage des structures fixes...")
    logger.info("Début de l'assemblage des matrices fixes...")

    c_D, c_Sigma_a, c_nuSigma_f, c_inv_v = get_material_properties(
        mesh, conn, rod_insertion=1.0, user_mapping=user_mapping
    )

    M = assemble_mass_or_reaction(nn, len(conn), 3, len(w), conn, det, w, N, c_inv_v)
    K = assemble_stiffness(
        nn, len(conn), 3, len(w), conn, det, w, jacobians, gradN_ref, c_D
    )

    c_R = c_nuSigma_f - c_Sigma_a
    R_init = assemble_mass_or_reaction(nn, len(conn), 3, len(w), conn, det, w, N, c_R)
    logger.info("Assemblage des matrices M, K et R_init terminé.")

    print("Démarrage du pilotage dynamique...")
    logger.info("Démarrage de l'intégration temporelle dynamique...")

    noeuds_bords = get_dirichlet_nodes(mesh, ["OuterBoundary"])

    # On utilise notre algorithme pour éviter un transitoire initial violent (Crash ou Prompt-Critique)
    position_depart_ideale = pos_rodBar_init(
        mesh,
        conn,
        det,
        w,
        N,
        get_material_properties,
        K,
        nn,
        user_mapping,
        noeuds_bords,
    )

    integrateur = TimeIntegrator(M, K, R_init, noeuds_bords)
    phi_0 = np.ones(nn) * 10.0

    try:
        times, solutions, final_pos = integrateur.integrate(
            phi_0,
            t_span=(0.0, 15.0),
            n_steps=150,
            mesh=mesh,
            elem_tags=conn,
            det=det,
            w=w,
            N=N,
            get_props_func=get_material_properties,
            pilot_callback=controleur_PD,
            user_mapping=user_mapping,
            initial_rod_pos=position_depart_ideale,
        )
    except Exception as e:
        logger.error(
            f"Erreur critique lors de l'intégration temporelle : {e}", exc_info=True
        )
        raise

    if headless:
        logger.info(
            "Mode Headless détecté. Calcul de la puissance et sauvegarde CSV..."
        )
        puissance_history = [np.sum(sol) for sol in solutions]

        if save_csv:
            try:
                data = np.column_stack((times, puissance_history))
                np.savetxt(
                    save_csv, data, delimiter=",", header="Time,Power", comments=""
                )
                logger.info(f"Fichier de résultats CSV sauvegardé : {save_csv}")
            except Exception as e:
                logger.error(
                    f"Échec de l'écriture du fichier CSV {save_csv} : {e}",
                    exc_info=True,
                )

        return solutions, times, puissance_history

    if not headless:
        print("Génération du Dashboard interactif...")
        logger.info("Démarrage de l'animation Matplotlib...")
        from matplotlib.animation import FuncAnimation
        import matplotlib.gridspec as gridspec

        # Création d'une fenêtre large avec 2 zones (Gauche: Maillage, Droite: Graphique)
        fig = plt.figure(figsize=(14, 6))
        fig.patch.set_facecolor("#1e1e1e")  # Mode sombre "Salle de commande"
        gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1])

        ax_mesh = fig.add_subplot(gs[0])
        ax_curve = fig.add_subplot(gs[1])

        # --- PRÉ-CALCULS POUR L'ÉCHELLE FIXE ---
        # On trouve le flux maximum absolu de TOUTE la simulation pour figer la colorbar
        flux_global_max = np.max(solutions)
        flux_global_max = max(
            flux_global_max, 1e-5
        )  # Sécurité si le réacteur est éteint

        # Pré-calcul du tableau des puissances en MW pour le graphique 1D
        puissances_mw = [np.sum(sol) * FACTEUR_MW for sol in solutions]

        # -- CODE POUR LE GRAPHE DE GAUCHE MONTRANT LE REACTEUR --
        ax_mesh.set_aspect("equal")
        ax_mesh.axis("off")
        ax_mesh.set_title(
            "Cartographie du Flux Neutronique", color="white", fontsize=14
        )

        mesh_plot = ax_mesh.tripcolor(
            mesh.points[:, 0],
            mesh.points[:, 1],
            mesh.cells_dict["triangle"],
            solutions[0],
            shading="gouraud",
            cmap="magma",
            vmin=0,
            vmax=flux_global_max,
        )

        cbar = fig.colorbar(mesh_plot, ax=ax_mesh, shrink=0.8)
        cbar.set_label("Flux (n/cm²/s)", color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

        # -- CODE POUR LE GRAPHE DE DROITE SUR LE CONTROLEUR --
        ax_curve.set_facecolor("#2b2b2b")
        ax_curve.tick_params(colors="white")
        for spine in ax_curve.spines.values():
            spine.set_color("#555555")

        ax_curve.set_title(
            "Cinétique du Réacteur (Contrôleur PD)", color="white", fontsize=14
        )
        ax_curve.set_xlabel("Temps (s)", color="white")
        ax_curve.set_ylabel("Puissance Thermique (MW)", color="white")
        ax_curve.grid(color="#444444", linestyle="--", linewidth=0.5)

        puissance_max_graphique = max(max(puissances_mw), PUISSANCE_CIBLE_MW) * 1.20
        ax_curve.set_xlim(times[0], times[-1])
        ax_curve.set_ylim(0, puissance_max_graphique)

        ax_curve.axhline(
            PUISSANCE_CIBLE_MW,
            color="#00ff00",
            linestyle="--",
            linewidth=2,
            label="Consigne (Cible)",
        )

        times_graphique = np.linspace(times[0], times[-1], len(solutions))

        ax_curve.plot(
            times_graphique, puissances_mw, color="#555555", linewidth=1.5, zorder=1
        )

        # Éléments dynamiques
        ligne_temps = ax_curve.axvline(
            times_graphique[0], color="red", linewidth=1.5, alpha=0.8, zorder=2
        )
        (point_puissance,) = ax_curve.plot(
            [times_graphique[0]],
            [puissances_mw[0]],
            marker="o",
            color="red",
            markersize=6,
            zorder=3,
            label="P(t) Actuelle",
        )
        (trace_courbe,) = ax_curve.plot(
            [], [], color="#00d2ff", linewidth=2.5, zorder=2
        )

        ax_curve.legend(
            facecolor="#1e1e1e",
            edgecolor="white",
            labelcolor="white",
            loc="lower right",
        )

        # -- CODE POUR L'ANIMATION --
        hud_text = fig.suptitle(
            "Initialisation...", color="#00d2ff", fontsize=16, fontweight="bold"
        )

        def animate(i):
            # Mise à jour du maillage
            mesh_plot.set_array(solutions[i])

            # Le temps actuel correspond simplement à l'index i de notre axe synchronisé
            temps_actuel = times_graphique[i]

            # Mise à jour du Radar sur le graphique
            ligne_temps.set_xdata([temps_actuel, temps_actuel])
            point_puissance.set_data([temps_actuel], [puissances_mw[i]])

            # Fait grandir la courbe bleue au fur et à mesure avec les bonnes dimensions
            trace_courbe.set_data(times_graphique[: i + 1], puissances_mw[: i + 1])

            # Affichage numérique (HUD) global
            hud_text.set_text(
                f"Temps : {temps_actuel:.2f} s  |  Puissance : {puissances_mw[i]:.2f} MW"
            )

            return mesh_plot, ligne_temps, point_puissance, trace_courbe, hud_text

        # Moteur d'animation (interval=40 ms donne 25 images/seconde)
        ani = FuncAnimation(
            fig, animate, frames=len(solutions), interval=40, blit=False
        )

        plt.tight_layout()
        plt.subplots_adjust(top=0.88)  # Laisse de la place pour le bandeau supérieur

        # --- SAUVEGARDE DU GIF ---
        # 1. Création d'un dossier dédié (il se créera là où tu lances ton script)
        dossier_sortie = Path("animations_sauvegardes")
        dossier_sortie.mkdir(parents=True, exist_ok=True)

        # 2. Nom de fichier dynamique avec horodatage (AnnéeMoisJour_HeureMinuteSeconde)
        horodatage = datetime.now().strftime("%Y%m%d_%H%M%S")
        nom_fichier = dossier_sortie / f"simulation_reacteur_{horodatage}.gif"

        # 3. La sauvegarde (Attention : se fait AVANT le plt.show())
        logger.info(f"Création du GIF en cours...")
        logger.info(f"Sauvegarde de l'animation vers {nom_fichier}...")

        try:
            # On sauvegarde à 25 fps (correspond à ton intervalle de 40ms : 1000/40 = 25)
            ani.save(nom_fichier, writer="pillow", fps=25)
            logger.info(f"L'animation a bien été sauvegardée avec succès")
        except Exception as e:
            logger.info(f"Échec de la sauvegarde GIF : {e}")

        # Affiche la fenêtre à l'écran après avoir sauvegardé
        plt.show()

    logger.info("Fin de la génération de l'animation.")
    return solutions, times, puissances_mw
