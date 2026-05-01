import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import splu

from core.assembly import assemble_mass_or_reaction

# --- IMPORT DU LOGGER ---
from utils.logger import get_logger

logger = get_logger(__name__)
# ------------------------


class TimeIntegrator:
    def __init__(self, M, K, R, dirichlet_dofs, theta=0.5):
        """
        Initialise la résolution temporelle (Schéma Theta).

        L'idée est simple : on essaie de deviner le futur (l'étape suivante)
        à partir du présent.
        - Explicite (θ=0) : On trace une ligne droite depuis le présent. Rapide, mais le code explose si on avance trop vite.
        - Implicite (θ=1) : On regarde depuis le futur. Très stable, mais ça lisse trop les détails.
        - Crank-Nicolson (θ=0.5) : Le compromis parfait. On prend la moyenne des deux.
        """
        logger.debug(f"Initialisation du TimeIntegrator avec theta = {theta}")
        
        # On convertit tout au format CSR (compressed sparse row). 
        # C'est un format de matrice creuse pour que SciPy fasse ses calculs rapidement.
        self.M = M.tocsr()
        self.K = K.tocsr()
        self.R = R.tocsr()
        
        # Les dirichlet_dofs sont les bords du réacteur où le flux est forcé à zéro (les neutrons s'échappent)
        self.dirichlet_dofs = np.asarray(dirichlet_dofs, dtype=int)
        self.theta = theta

    def integrate(
        self,
        phi_0,
        t_span,
        n_steps,
        mesh,
        elem_tags,
        det,
        w,
        N,
        get_props_func,
        pilot_callback=None,
        user_mapping=None,

        # Par défaut, on démarre avec les barres à moitié insérées (0.5)
        # C'est une sécurité pour éviter de démarrer sur une configuration explosive (bon compromis)
        initial_rod_pos=0.5, 
    ):
        t_start, t_end = t_span
        
        # Le pas de temps (dt). Plus il est petit, plus la simulation est précise
        # (mais plus le processeur devra faire de calculs)
        dt = (t_end - t_start) / n_steps
        logger.info(
            f"Début de l'intégration temporelle : t=[{t_start}, {t_end}], dt={dt:.5f}, étapes={n_steps}"
        )

        times = np.linspace(t_start, t_end, n_steps + 1)
        nn = self.M.shape[0] # Nombre total de nœuds du maillage
        ne = len(elem_tags)

        # On crée un filtre pour identifier les degrés de liberté libres (free_dofs)
        # En résumé on ne calcule la physique qu'à l'intérieur du réacteur, 
        # on ignore la frontière extérieure puisqu'on sait déjà que le flux y est nul.
        mask = np.ones(nn, dtype=bool)
        mask[self.dirichlet_dofs] = False
        free_dofs = np.nonzero(mask)[0]

        solutions = [phi_0.copy()]
        phi_n = phi_0.copy()
        phi_prev = phi_0.copy()

        # Point de départ des barres de contrôle (controleur PD)
        current_rod_pos = initial_rod_pos 

        # -- Optimisation pour la vitesse --
        # Re-calculer les matrices de A à Z prend énormément de temps
        # On va garder les matrices en mémoire (cache) tant que les barres de contrôle 
        # n'ont pas bougé de façon significative.
        last_computed_pos = -1.0  # Mis à -1 pour forcer le calcul à la première boucle
        solve_lu = None  
        B_mat = None  

        # Limite la fréquence d'affichage des logs pour ne pas polluer la console
        step_log_interval = max(1, n_steps // 10)

        for i in range(1, n_steps + 1):
            if i % step_log_interval == 0:
                logger.debug(f"Progression de l'intégration : Étape {i}/{n_steps}") # Tout les X affiche un log

            # Le controleur analyse la situation et calcule la nouvelle position des barres
            if pilot_callback is not None:
                current_rod_pos = pilot_callback(phi_n, phi_prev, current_rod_pos, dt)

            # -- MISE À JOUR DE LA PHYSIQUE --
            # Si les barres ont bougé de plus de 1% (0.01), on est obligé de recalculer la physique.
            # Sinon, on gagne du temps et on réutilise les anciennes matrices
            if abs(current_rod_pos - last_computed_pos) > 0.01:
                logger.info(
                    f"Mouvement significatif des barres détecté (pos={current_rod_pos:.4f}). Re-calcul de la physique et factorisation LU..."
                )

                # On met à jour les matériaux (l'absorption change là où la barre s'est déplacée)
                _, c_Sigma_a, c_nuSigma_f, _ = get_props_func(
                    mesh,
                    elem_tags,
                    rod_insertion=current_rod_pos,
                    user_mapping=user_mapping,
                )
                c_R = c_nuSigma_f - c_Sigma_a

                # On ré-assemble uniquement la matrice de réaction (très rapide grâce à Numba)
                self.R = assemble_mass_or_reaction(
                    nn, ne, 3, len(w), elem_tags, det, w, N, c_R
                )

                # Formules mathématiques du Schéma Theta
                L = self.R - self.K
                A = (self.M - self.theta * dt * L).tocsc()
                B_mat = (self.M + (1.0 - self.theta) * dt * L).tocsr()

                # -- ZONE DE CALCUL INTENSIF --
                # splu calcule la factorisation Lower-Upper de la matrice (pr triangulaire)
                # C'est l'opération la plus lourde de tout le programme
                # Elle pré-mâche le travail de résolution d'équation pour les prochaines étapes.
                A_FF = A[free_dofs, :][:, free_dofs]
                solve_lu = splu(A_FF)

                # On sauvegarde la position dans notre système de cache
                last_computed_pos = current_rod_pos

            # -- RÉSOLUTION ÉCLAIR --
            # Grâce au cache et à solve_lu calculé,
            # trouver l'état du réacteur à l'instant suivant ne prend plus qu'une fraction de milliseconde.
            b_full = B_mat.dot(phi_n)
            rhs_reduced = b_full[free_dofs]

            phi_free_np1 = solve_lu.solve(rhs_reduced)

            # On reconstruit l'image complète du maillage en réintégrant les zéros sur les bords
            phi_np1 = np.zeros(nn)
            phi_np1[free_dofs] = phi_free_np1

            # Sécurité physique : il y a toujours un léger bruit de fond neutronique naturel.
            # On empêche la matrice de plonger mathématiquement en dessous de zéro.
            phi_np1 = np.maximum(phi_np1, 1e-10)

            # -- SAUVEGARDE OPTIMISÉE --
            # On ne sauvegarde qu'une frame sur 2 (et la toute dernière)
            # Ça divise par deux le travail d'animation de Matplotlib à la fin et économise la RAM.
            if i % 2 == 0 or i == n_steps:
                solutions.append(phi_np1.copy())

            # On avance dans le temps
            phi_prev = phi_n.copy()
            phi_n = phi_np1

        logger.info(
            f"Fin de l'intégration. Position finale des barres : {current_rod_pos:.4f}"
        )
        return times, solutions, current_rod_pos