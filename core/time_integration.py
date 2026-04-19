# core/time_integration.py
"""
Module d'intégration temporelle pour l'équation de diffusion neutronique.

Équation résolue :
    (1/v) ∂φ/∂t = ∇·(D∇φ) + (νΣf - Σa)φ

Forme semi-discrète :
    M φ_t + K φ = R φ
où :
    - M : matrice de masse (terme (1/v) ∂φ/∂t)
    - K : matrice de diffusion (terme -∇·(D∇φ))
    - R : matrice de réaction (terme (νΣf - Σa)φ)
"""

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import lil_matrix, csr_matrix


class TimeIntegrator:
    """
    Classe pour l'intégration temporelle du problème de diffusion neutronique.
    """
    
    def __init__(self, M, K, R, dirichlet_dofs, theta=0.5):
        """
        Initialise l'intégrateur temporel.
        
        Parameters
        ----------
        M : sparse matrix
            Matrice de masse (terme instationnaire 1/v)
        K : sparse matrix
            Matrice de diffusion (Laplacien)
        R : sparse matrix
            Matrice de réaction (production - absorption)
        dirichlet_dofs : array-like
            Indices des DDLs avec conditions de Dirichlet
        theta : float, optional (default=0.5)
            Paramètre du θ-schéma :
            - θ = 0   : Euler explicite (instable pour diffusion)
            - θ = 0.5 : Crank-Nicolson (précision ordre 2)
            - θ = 1   : Euler implicite (stable, ordre 1)
        """
        self.M = M.tocsr()
        self.K = K.tocsr()
        self.R = R.tocsr()
        self.dirichlet_dofs = np.asarray(dirichlet_dofs, dtype=int)
        self.theta = theta
        
        # Matrice du système effectif : M - dt*(R - K)
        # car : M φ_t = -K φ + R φ = -(K - R) φ
        self.system_matrix = None   # pour optimiser le code, si pas de temps est le même, on sauvegarde A
        self.dt_current = None   # pour optimiser le code, si pas de temps est le même
        
    def step(self, phi_n, dt, dirichlet_values=None):
        """
        Effectue un pas de temps θ-schéma.
        
        Schéma résolu :
        (M + θ·dt·(K - R)) φ^{n+1} = (M - (1-θ)·dt·(K - R)) φ^n
        
        Parameters
        ----------
        phi_n : ndarray
            Solution au temps t^n
        dt : float
            Pas de temps Δt
        dirichlet_values : ndarray, optional
            Valeurs de Dirichlet au temps t^{n+1} (par défaut : 0)
            
        Returns
        -------
        phi_np1 : ndarray
            Solution au temps t^{n+1}
        """
        if dirichlet_values is None:
            dirichlet_values = np.zeros(len(self.dirichlet_dofs))
        
        # Construction des matrices du θ-schéma
        # Opérateur effectif : L = K - R (diffusion - réaction)
        L = self.K - self.R
        
        # Matrice LHS : M + θ·dt·L
        A = self.M + self.theta * dt * L
        
        # Matrice RHS : M - (1-θ)·dt·L
        B = self.M - (1.0 - self.theta) * dt * L
        
        # Second membre
        rhs = B.dot(phi_n)
        
        # Application des conditions de Dirichlet
        phi_np1 = self._solve_with_dirichlet(A, rhs, dirichlet_values)
        
        return phi_np1
    
    def _solve_with_dirichlet(self, A, rhs, dirichlet_values):
        """
        Résout le système avec conditions de Dirichlet par réduction.
        
        Parameters
        ----------
        A : sparse matrix
            Matrice du système
        rhs : ndarray
            Second membre
        dirichlet_values : ndarray
            Valeurs imposées sur les DDLs de Dirichlet
            
        Returns
        -------
        U_full : ndarray
            Solution complète
        """
        n = len(rhs)
        
        # Identification des DDLs libres et imposés
        mask = np.ones(n, dtype=bool)
        mask[self.dirichlet_dofs] = False
        free_dofs = np.nonzero(mask)[0]
        
        # Extraction du système réduit
        A_FF = A[free_dofs, :][:, free_dofs]
        A_FD = A[free_dofs, :][:, self.dirichlet_dofs]
        
        # Modification du second membre
        rhs_F = rhs[free_dofs]
        rhs_reduced = rhs_F - A_FD.dot(dirichlet_values)
        
        # Résolution
        U_free = spsolve(A_FF.tocsr(), rhs_reduced)
        
        # Reconstruction du vecteur complet
        U_full = np.zeros(n, dtype=float)
        U_full[free_dofs] = U_free
        U_full[self.dirichlet_dofs] = dirichlet_values
        
        return U_full
    
    def integrate(self, phi_0, t_span, n_steps, dirichlet_values=None, 
                  callback=None):
        """
        Intégration temporelle complète de t_0 à t_final.
        
        Parameters
        ----------
        phi_0 : ndarray
            Condition initiale φ(x, t=0)
        t_span : tuple (t_start, t_end)
            Intervalle de temps
        n_steps : int
            Nombre de pas de temps
        dirichlet_values : ndarray or callable, optional
            Valeurs de Dirichlet (constantes ou fonction de t)
        callback : callable, optional
            Fonction appelée à chaque pas : callback(t, phi)
            Utile pour calculer des observables ou sauvegarder
            
        Returns
        -------
        times : ndarray
            Tableau des temps [t_0, t_1, ..., t_final]
        solutions : list of ndarray
            Liste des solutions à chaque pas de temps
        """
        t_start, t_end = t_span
        dt = (t_end - t_start) / n_steps
        times = np.linspace(t_start, t_end, n_steps + 1)
        
        solutions = [phi_0.copy()]
        phi_n = phi_0.copy()
        
        for i, t in enumerate(times[1:], 1):
            # Valeurs de Dirichlet au temps t^{n+1}
            if callable(dirichlet_values):
                dir_vals = dirichlet_values(t)
            elif dirichlet_values is not None:
                dir_vals = dirichlet_values
            else:
                dir_vals = np.zeros(len(self.dirichlet_dofs))
            
            # Pas de temps
            phi_np1 = self.step(phi_n, dt, dir_vals)
            
            # Callback utilisateur (pour observables)
            if callback is not None:
                callback(t, phi_np1)
            
            solutions.append(phi_np1.copy())
            phi_n = phi_np1
        
        return times, solutions


def compute_time_step(M, K, R, cfl_factor=0.5):
    """
    Estime un pas de temps stable pour le schéma explicite.
    
    Pour le θ-schéma avec θ ≥ 0.5, il n'y a pas de restriction de stabilité,
    mais pour θ < 0.5 ou pour des raisons de précision, il est utile d'estimer
    un pas de temps raisonnable basé sur la condition CFL.
    
    Parameters
    ----------
    M : sparse matrix
        Matrice de masse
    K : sparse matrix
        Matrice de diffusion
    R : sparse matrix
        Matrice de réaction
    cfl_factor : float, optional
        Facteur de sécurité (typiquement entre 0.1 et 1.0)
        
    Returns
    -------
    dt_max : float
        Pas de temps maximal recommandé
        
    Notes
    -----
    Le pas de temps est estimé comme :
        dt_max ≈ cfl_factor / λ_max
    où λ_max est la plus grande valeur propre de M^{-1}(K - R)
    
    En pratique, on utilise une approximation diagonale rapide.
    """
    # Approximation diagonale de M^{-1}
    M_diag = M.diagonal()
    M_diag_inv = 1.0 / (M_diag + 1e-14)
    
    # Approximation diagonale de K - R
    L_diag = K.diagonal() - R.diagonal()
    
    # Estimation grossière de λ_max
    lambda_max_approx = np.max(M_diag_inv * L_diag)
    
    if lambda_max_approx > 0:
        dt_max = cfl_factor / lambda_max_approx
    else:
        # Système sur-critique pur (pas de diffusion dominante)
        dt_max = cfl_factor * 0.01  # Valeur par défaut conservatrice
    
    return dt_max