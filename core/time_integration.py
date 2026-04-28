import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import splu

from core.assembly import assemble_mass_or_reaction

class TimeIntegrator:
    def __init__(self, M, K, R, dirichlet_dofs, theta=0.5):
        """
        Initialise le schéma Theta (θ).
        
        On essaie d'estimer la pente d'une courbe pour deviner le futur.
        - Explicite (θ=0) : On trace la tangente au présent. Dangereux si on avance trop loin (dt grand).
        - Implicite (θ=1) : On trace la tangente depuis le futur. Très stable, mais amortit les détails.
        - Crank-Nicolson (θ=0.5) : On prend la moyenne des deux.
        """
        self.M = M.tocsr() 
        self.K = K.tocsr() 
        self.R = R.tocsr() 
        self.dirichlet_dofs = np.asarray(dirichlet_dofs, dtype=int)
        self.theta = theta
        
    def integrate(self, phi_0, t_span, n_steps, mesh, elem_tags, det, w, N, get_props_func, pilot_callback=None):
        t_start, t_end = t_span
        dt = (t_end - t_start) / n_steps
        times = np.linspace(t_start, t_end, n_steps + 1)
        nn = self.M.shape[0]
        
        # [LOGIC] Pré-calcul du masque de Dirichlet (ne change pas)
        mask = np.ones(nn, dtype=bool)
        mask[self.dirichlet_dofs] = False
        free_dofs = np.nonzero(mask)[0]

        solutions = [phi_0.copy()]
        phi_n = phi_0.copy()
        
        # [LOGIC] On initialise la mémoire du passé avec l'état initial
        phi_prev = phi_0.copy()

        # On initialise l'état des barres (ex: 0.0 =  pas insérées)
        current_rod_pos = 0.0 

        for i in range(1, n_steps + 1):
            # --- ÉTAPE DYNAMIQUE 1 : PILOTAGE ---
            # [LOGIC] On demande au "pilote" de bouger les barres selon le flux actuel
            if pilot_callback is not None:
                current_rod_pos = pilot_callback(phi_n, phi_prev, current_rod_pos)

            # --- ÉTAPE DYNAMIQUE 2 : MISE À JOUR PHYSIQUE ---
            # [MATH] On recalcule les propriétés (Sigma_a) avec la nouvelle position
            _, c_Sigma_a, c_nuSigma_f, _ = get_props_func(mesh, elem_tags, rod_insertion=current_rod_pos)
            c_R = c_nuSigma_f - c_Sigma_a
            
            # --- ÉTAPE DYNAMIQUE 3 : RE-ASSEMBLAGE ---
            ne = len(elem_tags)
            self.R = assemble_mass_or_reaction(nn, ne, 3, len(w), elem_tags, det, w, N, c_R)
            
            # --- ÉTAPE DYNAMIQUE 4 : RÉSOLUTION ---
            # [MATH] L'opérateur L = R - K change, donc A et B changent aussi !
            L = self.R - self.K
            A = (self.M - self.theta * dt * L).tocsc() 
            B = (self.M + (1.0 - self.theta) * dt * L).tocsr()
            
            # [LOGIC] CRITIQUE : On doit RE-FACTORISER la matrice à chaque pas de temps
            # C'est l'étape coûteuse, mais nécessaire pour le réalisme.
            A_FF = A[free_dofs, :][:, free_dofs]
            solve_lu = splu(A_FF)
            
            # Calcul du second membre (RHS)
            b_full = B.dot(phi_n)
            # (On suppose dir_vals = 0 pour simplifier)
            rhs_reduced = b_full[free_dofs]
            
            # Résolution du système pour ce pas de temps précis
            phi_free_np1 = solve_lu.solve(rhs_reduced)
            
            # Reconstruction et stockage
            phi_np1 = np.zeros(nn)
            phi_np1[free_dofs] = phi_free_np1
            
            # --- NOUVEAU : Plancher physique (Bruit de fond neutronique) ---
            phi_np1 = np.maximum(phi_np1, 1e-10)
            
            solutions.append(phi_np1.copy())
            
            phi_prev = phi_n.copy() 
            phi_n = phi_np1
            
        return times, solutions, current_rod_pos