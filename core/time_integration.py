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
        ne = len(elem_tags)
        
        mask = np.ones(nn, dtype=bool)
        mask[self.dirichlet_dofs] = False
        free_dofs = np.nonzero(mask)[0]

        solutions = [phi_0.copy()]
        phi_n = phi_0.copy()
        phi_prev = phi_0.copy()

        # Initialisation de la position de départ (levées)
        current_rod_pos = 0.0 
        
        # --- VARIABLES DE CACHE (LAZY COMPUTING) ---
        last_computed_pos = -1.0  # Mis à -1 pour forcer le calcul à la boucle 1
        solve_lu = None           # Stockera l'objet factorisé
        B_mat = None              # Stockera la matrice B
        
        for i in range(1, n_steps + 1):
            
            if pilot_callback is not None:
                current_rod_pos = pilot_callback(phi_n, phi_prev, current_rod_pos)

            # Si la barre a bougé de plus de 0.1%, on recalcule la physique
            if abs(current_rod_pos - last_computed_pos) > 0.001:
                
                _, c_Sigma_a, c_nuSigma_f, _ = get_props_func(mesh, elem_tags, rod_insertion=current_rod_pos)
                c_R = c_nuSigma_f - c_Sigma_a
                
                # Assemblage multi-threadé ultra rapide
                self.R = assemble_mass_or_reaction(nn, ne, 3, len(w), elem_tags, det, w, N, c_R)
                
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
            
        return times, solutions, current_rod_pos