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
        Initialise le schéma Theta (θ).

        On essaie d'estimer la pente d'une courbe pour deviner le futur.
        - Explicite (θ=0) : On trace la tangente au présent. Dangereux si on avance trop loin (dt grand).
        - Implicite (θ=1) : On trace la tangente depuis le futur. Très stable, mais amortit les détails.
        - Crank-Nicolson (θ=0.5) : On prend la moyenne des deux.
        """
        logger.debug(f"Initialisation du TimeIntegrator avec theta = {theta}")
        self.M = M.tocsr()
        self.K = K.tocsr()
        self.R = R.tocsr()
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
        # NOUVEAU : On ajoute un argument pour la position initiale des barres.
        # Par défaut, on le met à 0.85 (Barres insérées à 85%), c'est la sécurité absolue.
        initial_rod_pos=0.85, 
    ):
        t_start, t_end = t_span
        dt = (t_end - t_start) / n_steps
        logger.info(
            f"Début de l'intégration temporelle : t=[{t_start}, {t_end}], dt={dt:.5f}, étapes={n_steps}"
        )

        times = np.linspace(t_start, t_end, n_steps + 1)
        nn = self.M.shape[0]
        ne = len(elem_tags)

        mask = np.ones(nn, dtype=bool)
        mask[self.dirichlet_dofs] = False
        free_dofs = np.nonzero(mask)[0]

        solutions = [phi_0.copy()]
        phi_n = phi_0.copy()
        phi_prev = phi_0.copy()

        # On utilise la position sécurisée demandée en paramètre (de base 0.85) 
        # C'est le point de départ de notre PID.
        current_rod_pos = initial_rod_pos 

        # --- VARIABLES DE CACHE (LAZY COMPUTING) ---
        last_computed_pos = -1.0  # Mis à -1 pour forcer le calcul à la boucle 1
        solve_lu = None  
        B_mat = None  

        step_log_interval = max(1, n_steps // 10)

        for i in range(1, n_steps + 1):
            if i % step_log_interval == 0:
                logger.debug(f"Progression de l'intégration : Étape {i}/{n_steps}")

            if pilot_callback is not None:
                current_rod_pos = pilot_callback(phi_n, phi_prev, current_rod_pos)

            # Si la barre a bougé de plus de 0.1%, on recalcule la physique
            if abs(current_rod_pos - last_computed_pos) > 0.001:
                logger.info(
                    f"Mouvement significatif des barres détecté (pos={current_rod_pos:.4f}). Re-calcul de la physique et factorisation LU..."
                )

                _, c_Sigma_a, c_nuSigma_f, _ = get_props_func(
                    mesh,
                    elem_tags,
                    rod_insertion=current_rod_pos,
                    user_mapping=user_mapping,
                )
                c_R = c_nuSigma_f - c_Sigma_a

                # Assemblage multi-threadé ultra rapide
                self.R = assemble_mass_or_reaction(
                    nn, ne, 3, len(w), elem_tags, det, w, N, c_R
                )

                L = self.R - self.K
                A = (self.M - self.theta * dt * L).tocsc()
                B_mat = (self.M + (1.0 - self.theta) * dt * L).tocsr()

                # C'est l'étape la plus lourde de tout le programme.
                # On ne la lance QUE quand c'est indispensable.
                A_FF = A[free_dofs, :][:, free_dofs]
                solve_lu = splu(A_FF)

                # On met à jour la mémoire du cache
                last_computed_pos = current_rod_pos

            # --- RÉSOLUTION ÉCLAIR ---
            # On utilise le solveur LU et la matrice B qui sont en cache
            b_full = B_mat.dot(phi_n)
            rhs_reduced = b_full[free_dofs]

            # Résolution en une fraction de seconde grâce à splu précalculé
            phi_free_np1 = solve_lu.solve(rhs_reduced)

            # Reconstruction
            phi_np1 = np.zeros(nn)
            phi_np1[free_dofs] = phi_free_np1

            # Bruit de fond spontané (Masse critique)
            phi_np1 = np.maximum(phi_np1, 1e-10)

            solutions.append(phi_np1.copy())

            phi_prev = phi_n.copy()
            phi_n = phi_np1

        logger.info(
            f"Fin de l'intégration. Position finale des barres : {current_rod_pos:.4f}"
        )
        return times, solutions, current_rod_pos
