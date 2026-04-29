import numpy as np
import meshio
import matplotlib.pyplot as plt

from core.boundary_cond import get_dirichlet_nodes
from core.time_integration import TimeIntegrator
from core.assembly import assemble_mass_or_reaction, assemble_stiffness
from physics.materials import get_material_properties


def extract_p1_fem_data(mesh):
    """
    Construit le dictionnaire de traduction entre le Triangle de Référence Parfait
    et les Vrais Triangles déformés du maillage.
    """
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

    return triangles, dets, w, N, jacobians, gradN_ref


def run_full_simulation(mesh_path, user_mapping=None):
    """Lit le maillage, résout l'équation et affiche le résultat."""
    print(f"Chargement du maillage : {mesh_path}")
    mesh = meshio.read(mesh_path)
    nn = len(mesh.points)

    # 1. Extraction des données géométriques et tags
    # On identifie les groupes physiques pour savoir où est le Fuel et les Barres
    # Note: On utilise extract_p1_fem_data défini plus haut dans ce fichier
    conn, det, w, N, jacobians, gradN_ref = extract_p1_fem_data(mesh)

    # 2. Définition du Scénario et du Pilote
    PUISSANCE_CIBLE = 50000000000.0

    # Mémoires du PID
    erreur_precedente = 0.0
    erreur_integrale = 0.0

    def pilote_automatique_intelligent(phi_actuel, phi_precedent, position_barres):
        nonlocal erreur_precedente, erreur_integrale

        puissance_t = np.sum(phi_actuel)

        # Sécurité : On empêche la puissance de tomber au zéro mathématique absolu
        p_safe = max(puissance_t, 1e-5)

        # 1. LE SECRET : L'Erreur Logarithmique !
        # log(P_actuel / P_cible).
        # Si P_actuel < P_cible, l'erreur est négative -> Les barres vont se lever.
        erreur = np.log(p_safe / PUISSANCE_CIBLE)

        # 2. Les Gains du PID (Ajustés pour la dynamique logarithmique)
        Kp = 0.03  # Action immédiate
        Ki = (
            0.002  # Chercheur de point critique (très faible pour éviter l'emballement)
        )
        Kd = 0.15  # Amortisseur prédictif

        # 3. Calcul de la dérivée et de l'intégrale
        derivee_erreur = erreur - erreur_precedente

        # Anti-Windup : On ne cumule l'intégrale que si on est proche de la cible (à +/- un facteur e)
        # Ça empêche le pilote de devenir "fou" si le démarrage prend du temps.
        if abs(erreur) < 1.0:
            erreur_integrale += erreur

        erreur_precedente = erreur

        # 4. Calcul du mouvement
        delta_pos = Kp * erreur + Ki * erreur_integrale + Kd * derivee_erreur

        # 5. Sécurités physiques (Vitesse max des moteurs : 5% de la course par itération)
        delta_pos = np.clip(delta_pos, -0.05, 0.05)

        # 6. Application (0.0 = complètement levé, 1.0 = complètement inséré)
        nouvelle_pos = np.clip(position_barres + delta_pos, 0.0, 1.0)

        return nouvelle_pos

    # 3. Pré-assemblage des matrices fixes
    # M (Masse) et K (Diffusion/Fuites) ne changent jamais
    print("Assemblage des structures fixes...")
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

    # 4. Lancement de la Simulation CINÉTIQUE
    print("Démarrage du pilotage dynamique...")

    noeuds_bords = get_dirichlet_nodes(mesh, ["OuterBoundary"])
    integrateur = TimeIntegrator(M, K, R_init, noeuds_bords, theta=1.0)

    # Initialisation : flux nul partout sauf un peu de 'bruit' pour démarrer
    phi_0 = np.ones(nn) * 10.0

    # [LOGIC] On passe toutes les fonctions nécessaires à l'intégrateur
    # pour qu'il puisse recalculer la physique en boucle.
    times, solutions, final_pos = integrateur.integrate(
        phi_0,
        t_span=(0.0, 0.05),
        n_steps=100,
        mesh=mesh,
        elem_tags=conn,
        det=det,  # <-- NOUVEAU
        w=w,  # <-- NOUVEAU
        N=N,  # <-- NOUVEAU
        get_props_func=get_material_properties,
        pilot_callback=pilote_automatique_intelligent,
    )

    # 6. Visualisation du résultat final (Animation du transitoire)
    print("Génération de l'animation...")
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

    title = ax.set_title("Temps : 0.0 s", color="white", fontsize=14)

    def animate(i):
        # 1. On met à jour les données de flux pour l'image courante
        mesh_plot.set_array(solutions[i])

        # 2. TRÈS IMPORTANT : On ajuste dynamiquement l'échelle de couleurs.
        # Au fur et à mesure que les neutrons diffusent, le pic maximum diminue.
        # Si on ne fait pas ça, l'image deviendrait de plus en plus noire.
        vmax_current = np.max(solutions[i])
        if vmax_current < 1e-5:
            vmax_current = 1e-5  # Sécurité pour éviter la division par zéro
        mesh_plot.set_clim(vmin=0, vmax=vmax_current)

        # 3. Mise à jour du chrono
        title.set_text(f"Temps : {times[i]:.5f} s")
        return mesh_plot, title

    # Lancement de l'animation (interval=50 ms entre chaque image)
    ani = FuncAnimation(fig, animate, frames=len(solutions), interval=50, blit=False)

    plt.tight_layout()
    plt.show()

    return solutions[-1]
