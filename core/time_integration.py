import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
import pypardiso

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
        initial_rod_pos=0.5, 
    ):
        t_start, t_end = t_span
        dt = (t_end - t_start) / n_steps
        
        logger.info(
            f"Début de l'intégration temporelle : t=[{t_start}, {t_end}], dt={dt:.5f}, étapes={n_steps}"
        )

        # On génère le vecteur de temps complet pour le calcul interne
        times_full = np.linspace(t_start, t_end, n_steps + 1)
        
        nn = self.M.shape[0]
        ne = len(elem_tags)

        # Filtrage des DOFs pour les conditions aux limites (Dirichlet)
        mask = np.ones(nn, dtype=bool)
        mask[self.dirichlet_dofs] = False
        free_dofs = np.nonzero(mask)[0]

        # --- SYNCHRONISATION DES SORTIES ---
        # On initialise les listes qui contiendront les résultats finaux
        solutions = [phi_0.copy()]
        saved_times = [times_full[0]] # On garde l'instant t=0 correspondant à phi_0

        phi_n = phi_0.copy()
        phi_prev = phi_0.copy()
        current_rod_pos = initial_rod_pos 

        # -- Système de Cache pour éviter les calculs inutiles --
        last_computed_pos = -1.0 
        solve_lu = None  
        B_mat = None  

        step_log_interval = max(1, n_steps // 10)

        for i in range(1, n_steps + 1):
            if i % step_log_interval == 0:
                logger.debug(f"Progression : Étape {i}/{n_steps}")

            # Calcul de la nouvelle position des barres via le pilote (PID/PD)
            if pilot_callback is not None:
                current_rod_pos = pilot_callback(phi_n, phi_prev, current_rod_pos, dt)

            # -- MISE À JOUR DE LA PHYSIQUE (CACHE LU) --
            # On ne recalcule et ne factorise la matrice que si le mouvement est significatif (> 1%)
            if abs(current_rod_pos - last_computed_pos) > 0.01:
                logger.info(f"Recalcul physique : Barres à {current_rod_pos:.4f}")

                # Mise à jour des propriétés matériaux selon la nouvelle position
                _, c_Sigma_a, c_nuSigma_f, _ = get_props_func(
                    mesh, elem_tags, rod_insertion=current_rod_pos, user_mapping=user_mapping
                )
                c_R = c_nuSigma_f - c_Sigma_a

                # Ré-assemblage rapide de la réaction
                self.R = assemble_mass_or_reaction(nn, ne, 3, len(w), elem_tags, det, w, N, c_R)

                # Construction des opérateurs du Schéma Theta (Crank-Nicolson si theta=0.5)
                L = self.R - self.K
                A = (self.M - self.theta * dt * L).tocsc()
                B_mat = (self.M + (1.0 - self.theta) * dt * L).tocsr()

                # Factorisation LU (Partie lourde)
                A_FF = A[free_dofs, :][:, free_dofs]
                solve_lu = pypardiso.factorized(A_FF)

                last_computed_pos = current_rod_pos

            # -- RÉSOLUTION DU SYSTÈME --
            # Calcul du second membre (RHS)
            b_full = B_mat.dot(phi_n)
            rhs_reduced = b_full[free_dofs]

            # Résolution rapide via la factorisation LU déjà prête
            phi_free_np1 = solve_lu(rhs_reduced)

            # Reconstruction du vecteur complet (avec les bords à zéro)
            phi_np1 = np.zeros(nn)
            phi_np1[free_dofs] = phi_free_np1
            phi_np1 = np.maximum(phi_np1, 1e-10) # Sécurité : Pas de flux négatif

            # --- SAUVEGARDE OPTIMISÉE (SÉLECTIONNÉE) ---
            # Correction cruciale : On ne sauvegarde le temps que si on sauvegarde la solution
            if i % 2 == 0 or i == n_steps:
                solutions.append(phi_np1.copy())
                saved_times.append(times_full[i])

            # Passage à l'étape suivante
            phi_prev = phi_n.copy()
            phi_n = phi_np1

        logger.info(f"Intégration terminée. Position finale : {current_rod_pos:.4f}")
        
        # On retourne un array numpy pour les temps pour être compatible avec Matplotlib
        return np.array(saved_times), solutions, current_rod_pos
