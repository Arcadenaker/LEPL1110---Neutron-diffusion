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
        # Elle contient les dérivées spatiales. Elle sert de transformation liénaire
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


def pos_rodBar_init(mesh, conn, det, w, N, get_props_func, K, nn, user_mapping, noeuds_bords):
    """
    Recherche algorithmique de l'insertion optimale initiale des barres de contrôle par Dichotomie
    
    Cette fonction permet d'aider le controleur PD et ne pas trop osciller pour atteindre la position
    de stationnarité
    """
    logger.info("Recherche du point d'équilibre initial des barres de contrôle")
    
    mask = np.ones(nn, dtype=bool)
    mask[noeuds_bords] = False
    free_dofs = np.nonzero(mask)[0]
    
    K_FF = K[free_dofs, :][:, free_dofs]
    
    # Théoriquement, la criticité absolue s'obtient en cherchant la plus grande 
    # valeur propre de la matrice (R - K). Cependant, résoudre eigsh() (qui cherche les valeurs propres) 
    # prend plusieurs secondes par itération
    # Ici, on utilise une approximation physique instantanée : on injecte un flux plat
    # et on analyse le signe de la dérivée temporelle (variation_flux). 
    # C'est une multiplication Matrice-Vecteur (O(n)), des milliers de fois plus rapide.
    phi_test = np.ones(len(free_dofs))
    
    pos_min = 0.0  
    pos_max = 1.0  
    pos_critique = 0.5
    
    # 10 itérations de dichotomie garantissent une précision de 1/2^10 = ~0.001 (0.1% de la course)
    for i in range(10): 
        pos_critique = (pos_min + pos_max) / 2.0
        
        _, c_Sigma_a, c_nuSigma_f, _ = get_props_func(
            mesh, conn, rod_insertion=pos_critique, user_mapping=user_mapping
        )
        c_R = c_nuSigma_f - c_Sigma_a
        
        R = assemble_mass_or_reaction(nn, len(conn), 3, len(w), conn, det, w, N, c_R)
        R_FF = R[free_dofs, :][:, free_dofs]
        
        # Matrice d'évolution A = Création (R) - Pertes (K)
        A_FF = R_FF - K_FF
        
        # Évaluation instantanée du bilan neutronique local
        variation_flux = A_FF.dot(phi_test)
        bilan_neutronique = np.sum(variation_flux)
        
        if bilan_neutronique > 0:
            # Sur-critique : on doit rentrer les barres de contrôle (augmenter pos_min)
            pos_min = pos_critique
        else:
            # Sous-critique : on doit retirer les barres de contrôle (diminuer pos_max)
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

    conn, det, w, N, jacobians, gradN_ref = extract_p1_fem_data(mesh)

    phi_0 = np.ones(nn) * 10.0
    puissance_initiale = np.sum(phi_0)
    logger.debug(f"Flux initial défini. Puissance initiale = {puissance_initiale}")

    POURCENTAGE_CIBLE = 150  # Puissance visée pour la stationnérité du réacteur
    PUISSANCE_CIBLE = puissance_initiale * (POURCENTAGE_CIBLE / 100.0)
    logger.info(f"Scénario PID: Démarrage visé à {POURCENTAGE_CIBLE}% -> Puissance cible = {PUISSANCE_CIBLE}")

    erreur_precedente = None # Variable mémoire du controleur
    def controleur_PD(phi_actuel, phi_precedent, position_barres, dt):
        """
        Régulateur Proportionnel-Dérivé (PD) adapté à la Cinétique des Réacteurs.
        """
        nonlocal erreur_precedente

        puissance_t = np.sum(phi_actuel)
        p_safe = max(puissance_t, 1e-5)
        
        # La dynamique d'un réacteur est régie par des équations différentielles exponentielles.
        # Une erreur linéaire classique (Cible - Actuel) produirait des valeurs démesurées 
        # lors des transitoires, provoquant la saturation instantanée des actionneurs.
        # L'erreur logarithmique permet de piloter la réponse exponentielle de façon quasi-linéaire.
        erreur = np.log(p_safe / PUISSANCE_CIBLE)

        if erreur_precedente is None:
            erreur_precedente = erreur

        # OMEGA (ω) : L'inverse de la Période du réacteur (T)
        # C'est la dérivée temporelle de l'erreur logarithmique.
        # Ce terme anticipe l'inertie neutronique : si ω est très élevé, la puissance 
        # grimpe trop vite, et ce terme forcera l'insertion des barres avant même d'atteindre la cible.
        omega = (erreur - erreur_precedente) / dt

        # Kp agit comme la raideur d'un ressort vers la cible.
        # Kd agit comme un amortisseur visqueux pour tuer les oscillations (effet yoyo).
        Kp = 0.15   
        Kd = 0.50   

        # Loi de commande finale
        vitesse_demandee = (Kp * erreur) + (Kd * omega)
        delta_pos = vitesse_demandee * dt

        # Saturation mécanique des actionneurs (Vitesse maximale physiquement possible)
        vitesse_max = 0.40
        delta_pos = np.clip(delta_pos, -vitesse_max * dt, vitesse_max * dt)

        nouvelle_pos = np.clip(position_barres + delta_pos, 0.0, 1.0)

        erreur_precedente = erreur
        return nouvelle_pos

    print("Assemblage des structures fixes...")
    logger.info("Début de l'assemblage des matrices fixes...")
    
    c_D, c_Sigma_a, c_nuSigma_f, c_inv_v = get_material_properties(
        mesh, conn, rod_insertion=1.0, user_mapping=user_mapping
    )

    M = assemble_mass_or_reaction(nn, len(conn), 3, len(w), conn, det, w, N, c_inv_v)
    K = assemble_stiffness(nn, len(conn), 3, len(w), conn, det, w, jacobians, gradN_ref, c_D)

    c_R = c_nuSigma_f - c_Sigma_a
    R_init = assemble_mass_or_reaction(nn, len(conn), 3, len(w), conn, det, w, N, c_R)
    logger.info("Assemblage des matrices M, K et R_init terminé.")

    print("Démarrage du pilotage dynamique...")
    logger.info("Démarrage de l'intégration temporelle dynamique...")

    noeuds_bords = get_dirichlet_nodes(mesh, ["OuterBoundary"])

    # On utilise notre algorithme pour éviter un transitoire initial violent (Crash ou Prompt-Critique)
    position_depart_ideale = pos_rodBar_init(
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
            pilot_callback=controleur_PD,
            user_mapping=user_mapping,
            initial_rod_pos=position_depart_ideale,
        )
    except Exception as e:
        logger.error(f"Erreur critique lors de l'intégration temporelle : {e}", exc_info=True)
        raise

    if headless:
        logger.info("Mode Headless détecté. Calcul de la puissance et sauvegarde CSV...")
        puissance_history = [np.sum(sol) for sol in solutions]

        if save_csv:
            try:
                data = np.column_stack((times, puissance_history))
                np.savetxt(save_csv, data, delimiter=",", header="Time,Power", comments="")
                logger.info(f"Fichier de résultats CSV sauvegardé : {save_csv}")
            except Exception as e:
                logger.error(f"Échec de l'écriture du fichier CSV {save_csv} : {e}", exc_info=True)

        return times, puissance_history

    print("Génération de l'animation...")
    logger.info("Démarrage de la génération de l'animation Matplotlib...")
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.axis("off")  
    fig.patch.set_facecolor("#1e1e1e")  

    mesh_plot = ax.tripcolor(
        mesh.points[:, 0],
        mesh.points[:, 1],
        mesh.cells_dict["triangle"],
        solutions[0],
        shading="gouraud",
        cmap="magma",
    )

    cbar = fig.colorbar(mesh_plot, ax=ax, shrink=0.8)
    cbar.set_label("Flux Neutronique", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    puissance_init = np.sum(solutions[0])
    title = ax.set_title(f"Temps : 0.00 s  |  Puissance : {puissance_init:,.0f}", color="white", fontsize=14)

    def animate(i):
        mesh_plot.set_array(solutions[i])

        # Recalibrage dynamique de l'échelle des couleurs (clim)
        # Indispensable car la magnitude du flux évolue de manière exponentielle au cours du temps.
        vmax_current = np.max(solutions[i])
        if vmax_current < 1e-5:
            vmax_current = 1e-5
        mesh_plot.set_clim(vmin=0, vmax=vmax_current)

        puissance_actuelle = np.sum(solutions[i])

        # Interpolation temporelle
        # Comme l'intégrateur a pu sauter la sauvegarde de certaines étapes (ex: i % 2 == 0)
        # pour optimiser la mémoire, on doit déduire le temps continu proportionnellement à l'indice.
        progression = i / max(1, len(solutions) - 1)
        temps_actuel = times[0] + progression * (times[-1] - times[0])

        title.set_text(f"Temps : {temps_actuel:.2f} s  |  Puissance : {puissance_actuelle:,.0f}")
        
        return mesh_plot, title

    ani = FuncAnimation(fig, animate, frames=len(solutions), interval=50, blit=False)

    plt.tight_layout()
    plt.show()

    logger.info("Fin de l'exécution de run_full_simulation.")
    return solutions[-1]