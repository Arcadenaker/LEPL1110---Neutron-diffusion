import numpy as np
import meshio
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

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
    Construit le dictionnaire de traduction entre le Triangle de Référence Parfait
    et les Vrais Triangles déformés du maillage.
    """
    logger.debug("Extraction des données P1 FEM depuis le maillage...")
    triangles = mesh.cells_dict["triangle"]
    ne = len(triangles)
    points = mesh.points

    ngp = 3
    w = np.array([1 / 6, 1 / 6, 1 / 6])
    N = np.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]])

    gradN_ref = np.zeros((ngp, 3, 2))
    for g in range(ngp):
        gradN_ref[g, 0, :] = [-1.0, -1.0]
        gradN_ref[g, 1, :] = [1.0, 0.0]
        gradN_ref[g, 2, :] = [0.0, 1.0]

    jacobians = np.zeros((ne, ngp, 2, 2))
    dets = np.zeros((ne, ngp))

    for e in range(ne):
        p = points[triangles[e]]
        v1 = p[1] - p[0]
        v2 = p[2] - p[0]

        J = np.array([[v1[0], v2[0]], [v1[1], v2[1]]])
        det_J = abs(v1[0] * v2[1] - v1[1] * v2[0])

        for g in range(ngp):
            jacobians[e, g] = J
            dets[e, g] = det_J

    logger.debug(f"Extraction terminée pour {ne} éléments triangulaires.")
    return triangles, dets, w, N, jacobians, gradN_ref


def precalculer_position_critique(mesh, conn, det, w, N, get_props_func, K, nn, user_mapping, noeuds_bords):
    logger.info("Recherche du point d'équilibre initial des barres de contrôle")
    
    # 1. On isole les noeuds physiques
    mask = np.ones(nn, dtype=bool)
    mask[noeuds_bords] = False
    free_dofs = np.nonzero(mask)[0]
    
    K_FF = K[free_dofs, :][:, free_dofs]
    
    # --- NOUVEAU : Le vecteur de test (flux plat) ---
    phi_test = np.ones(len(free_dofs))
    
    pos_min = 0.0  
    pos_max = 1.0  
    pos_critique = 0.5
    
    # Vu que c'est gratuit en temps de calcul, on peut faire 10 étapes
    # pour avoir une précision chirurgicale (à 0.001 près !)
    for i in range(10): 
        pos_critique = (pos_min + pos_max) / 2.0
        
        _, c_Sigma_a, c_nuSigma_f, _ = get_props_func(
            mesh, conn, rod_insertion=pos_critique, user_mapping=user_mapping
        )
        c_R = c_nuSigma_f - c_Sigma_a
        
        # Assemblage hyper rapide (déjà optimisé sous Numba)
        R = assemble_mass_or_reaction(nn, len(conn), 3, len(w), conn, det, w, N, c_R)
        R_FF = R[free_dofs, :][:, free_dofs]
        
        A_FF = R_FF - K_FF
        
        # --- LA MAGIE MATHÉMATIQUE (0.001 seconde) ---
        # On calcule la dérivée instantanée du flux global
        variation_flux = A_FF.dot(phi_test)
        bilan_neutronique = np.sum(variation_flux)
        
        if bilan_neutronique > 0:
            # Réaction s'emballe, on enfonce les barres
            pos_min = pos_critique
        else:
            # Réaction s'étouffe, on lève les barres
            pos_max = pos_critique

    logger.info(f"Position d'équilibre trouvée : {pos_critique:.3f}")
    return pos_critique


def run_full_simulation(mesh_path, user_mapping=None, headless=False, save_csv=None):
    """Lit le maillage, résout l'équation et affiche le résultat."""
    logger.info(f"Démarrage de run_full_simulation sur : {mesh_path}")
    print(f"Chargement du maillage : {mesh_path}")

    try:
        mesh = meshio.read(mesh_path)
    except Exception as e:
        logger.error(
            f"Erreur lors de la lecture du maillage {mesh_path} : {e}", exc_info=True
        )
        raise

    nn = len(mesh.points)
    logger.debug(f"Maillage chargé. Nombre de nœuds : {nn}")

    # 1. Extraction des données géométriques et tags
    # On identifie les groupes physiques pour savoir où est le Fuel et les Barres
    # Note: On utilise extract_p1_fem_data défini plus haut dans ce fichier
    conn, det, w, N, jacobians, gradN_ref = extract_p1_fem_data(mesh)

    phi_0 = np.ones(nn) * 10.0
    puissance_initiale = np.sum(phi_0)
    logger.debug(
        f"Flux initial défini. Puissance initiale (intégrale) = {puissance_initiale}"
    )

    # Le réacteur est à l'arrêt, il n'y a que le bruit de fond (puissance_initiale).
    # On demande au PID de "tirer" les barres pour multiplier cette puissance par 1.5.
    POURCENTAGE_CIBLE = 150 

    # Calcul automatique de la cible absolue
    PUISSANCE_CIBLE = puissance_initiale * (POURCENTAGE_CIBLE / 100.0)
    logger.info(
        f"Scénario PID: Démarrage visé à {POURCENTAGE_CIBLE}% -> Puissance cible = {PUISSANCE_CIBLE}"
    )

    # Mémoire pour le calcul de la "Période" (Vitesse exponentielle)
    erreur_precedente = None

    def pilote_automatique_intelligent(phi_actuel, phi_precedent, position_barres, dt):
        nonlocal erreur_precedente

        puissance_t = np.sum(phi_actuel)
        p_safe = max(puissance_t, 1e-5)
        
        # 1. L'ERREUR LOGARITHMIQUE (La seule qui marche pour un réacteur !)
        # Si la puissance est trop HAUTE, l'erreur est POSITIVE (il faut insérer)
        # Si la puissance est trop BASSE, l'erreur est NÉGATIVE (il faut lever)
        erreur = np.log(p_safe / PUISSANCE_CIBLE)

        if erreur_precedente is None:
            erreur_precedente = erreur

        # 2. OMEGA (L'inverse de la Période du réacteur)
        # C'est notre Radar d'Anticipation. Il mesure l'accélération exponentielle.
        omega = (erreur - erreur_precedente) / dt

        # 3. LES GAINS (L'équilibre parfait)
        Kp = 0.15   # Le ressort : tire doucement vers la cible
        Kd = 0.50   # L'AMORTISSEUR EXTRÊME : Tient compte de l'inertie et freine très tôt

        # 4. CALCUL DU MOUVEMENT
        # Comme 0.0 = levé et 1.0 = inséré, une erreur positive (trop de puissance)
        # donne un delta_pos positif (on enfonce les barres). Le signe est naturel !
        vitesse_demandee = (Kp * erreur) + (Kd * omega)
        delta_pos = vitesse_demandee * dt

        # 5. SÉCURITÉ MÉCANIQUE (On autorise les barres à aller à 40% par seconde 
        # pour leur donner une chance de rattraper le flux)
        vitesse_max = 0.40
        delta_pos = np.clip(delta_pos, -vitesse_max * dt, vitesse_max * dt)

        nouvelle_pos = np.clip(position_barres + delta_pos, 0.0, 1.0)

        # LE RADAR DE PRÉCISION
        print(f"[PID] Puissance: {p_safe:.0f} | ErrLog: {erreur:+.2f} | Accélération (Omega): {omega:+.2f} | Barres: {position_barres:.3f} -> {nouvelle_pos:.3f}")

        erreur_precedente = erreur
        return nouvelle_pos

    # 3. Pré-assemblage des matrices fixes
    # M (Masse) et K (Diffusion/Fuites) ne changent jamais
    print("Assemblage des structures fixes...")
    logger.info("Début de l'assemblage des matrices fixes...")
    # On transmet ici le user_mapping à la fonction d'extraction
    c_D, c_Sigma_a, c_nuSigma_f, c_inv_v = get_material_properties(
        mesh, conn, rod_insertion=1.0, user_mapping=user_mapping
    )

    M = assemble_mass_or_reaction(nn, len(conn), 3, len(w), conn, det, w, N, c_inv_v)
    K = assemble_stiffness(
        nn, len(conn), 3, len(w), conn, det, w, jacobians, gradN_ref, c_D
    )

    # Matrice R initiale (pourra être mise à jour par l'intégrateur)
    c_R = c_nuSigma_f - c_Sigma_a
    R_init = assemble_mass_or_reaction(nn, len(conn), 3, len(w), conn, det, w, N, c_R)
    logger.info("Assemblage des matrices M, K et R_init terminé.")

    # 4. Lancement de la Simulation CINÉTIQUE
    print("Démarrage du pilotage dynamique...")
    logger.info("Démarrage de l'intégration temporelle dynamique...")

    noeuds_bords = get_dirichlet_nodes(mesh, ["OuterBoundary"])

    position_depart_ideale = precalculer_position_critique(
        mesh, conn, det, w, N, get_material_properties, K, nn, user_mapping, noeuds_bords
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
            pilot_callback=pilote_automatique_intelligent,
            user_mapping=user_mapping,
            
            # --- ON INJECTE NOTRE DÉCOUVERTE ICI ! ---
            initial_rod_pos=position_depart_ideale,
        )
    except Exception as e:
        logger.error(
            f"Erreur critique lors de l'intégration temporelle : {e}", exc_info=True
        )
        raise

    # --- 5. COMPORTEMENT HEADLESS (BATCH ET ÉTUDE PARAMÉTRIQUE) ---
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

        return times, puissance_history

    # 6. Visualisation du résultat final (Animation du transitoire)
    print("Génération de l'animation...")
    logger.info("Démarrage de la génération de l'animation Matplotlib...")
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.axis("off")  # On enlève les axes pour un rendu plus esthétique (mode sombre)
    fig.patch.set_facecolor("#1e1e1e")  # Fond gris foncé style VS Code

    # On utilise tripcolor avec shading='gouraud' pour un lissage parfait, c'est bien plus beau que tricontourf
    # On utilise la palette 'magma' ou 'plasma' qui ont de très beaux dégradés
    mesh_plot = ax.tripcolor(
        mesh.points[:, 0],
        mesh.points[:, 1],
        mesh.cells_dict["triangle"],
        solutions[0],
        shading="gouraud",
        cmap="magma",
    )

    # Configuration de la barre de couleur
    cbar = fig.colorbar(mesh_plot, ax=ax, shrink=0.8)
    cbar.set_label("Flux Neutronique", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")


    puissance_init = np.sum(solutions[0])
    title = ax.set_title(f"Temps : 0.00 s  |  Puissance : {puissance_init:,.0f}", color="white", fontsize=14)

    def animate(i):
        # 1. On met à jour les données de flux
        mesh_plot.set_array(solutions[i])

        # 2. Ajustement de l'échelle de couleurs
        vmax_current = np.max(solutions[i])
        if vmax_current < 1e-5:
            vmax_current = 1e-5
        mesh_plot.set_clim(vmin=0, vmax=vmax_current)

        # 3. Calcul de la puissance en temps réel pour cette image exacte
        puissance_actuelle = np.sum(solutions[i])

        # 4. Calcul du vrai temps
        # Astuce : Si tu as appliqué mon conseil de ne sauvegarder qu'une image sur 5 
        # pour aller plus vite, 'times[i]' serait désynchronisé. 
        # On calcule donc le temps réel proportionnellement à l'image affichée :
        progression = i / max(1, len(solutions) - 1)
        temps_actuel = times[0] + progression * (times[-1] - times[0])

        # 5. Mise à jour du texte
        title.set_text(f"Temps : {temps_actuel:.2f} s  |  Puissance : {puissance_actuelle:,.0f}")
        
        return mesh_plot, title

    # Lancement de l'animation (interval=50 ms entre chaque image)
    ani = FuncAnimation(fig, animate, frames=len(solutions), interval=50, blit=False)

    plt.tight_layout()
    plt.show()

    logger.info("Fin de l'exécution de run_full_simulation.")
    return solutions[-1]
