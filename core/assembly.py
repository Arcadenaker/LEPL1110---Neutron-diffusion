import numpy as np
from scipy.sparse import coo_matrix
from numba import njit

@njit
def _fast_assemble_mass_reaction_core(ne, nloc, ngp, conn, det, w, N, coeffs):
    """
    [MATH] Intégrale de forme : ∫ c * N_i * N_j dx
    """
    size = ne * nloc * nloc * ngp
    data = np.zeros(size)
    rows = np.zeros(size, dtype=np.int32)
    cols = np.zeros(size, dtype=np.int32)
    
    idx = 0
    for e in range(ne):
        c_e = coeffs[e]
        nodes = conn[e]
        
        for g in range(ngp):
            # [LOGIC] w[g] * det[e, g] représente la "vraie" fraction d'aire de l'élément 
            # associée à ce point de Gauss.
            facteur = w[g] * det[e, g] * c_e
            
            for a in range(nloc):
                for b in range(nloc):
                    data[idx] = facteur * N[g, a] * N[g, b]
                    rows[idx] = nodes[a]
                    cols[idx] = nodes[b]
                    idx += 1
                    
    return data[:idx], rows[:idx], cols[:idx]

@njit
def _fast_assemble_stiffness_core(ne, nloc, ngp, conn, det, w, jacobians, gradN_ref, coeffs_D):
    """
    [MATH] Intégrale de Diffusion : ∫ D * (∇N_i · ∇N_j) dx
    """
    size = ne * nloc * nloc * ngp
    data = np.zeros(size)
    rows = np.zeros(size, dtype=np.int32)
    cols = np.zeros(size, dtype=np.int32)

    # On alloue la mémoire une seule fois, en dehors de toutes les boucles.
    # Au lieu d'allouer à chaque fois ces matrices dans chaque boucle, on écrase simplement les données (+ fast)
    grad_real_x = np.zeros(nloc)
    grad_real_y = np.zeros(nloc)
    
    # [LOGIC] Pré-allocation de scalaires virtuels pour remplacer les tableaux
    idx = 0
    for e in range(ne):
        D_e = coeffs_D[e]
        nodes = conn[e]
        
        for g in range(ngp):
            # On extrait les composantes du Jacobien 2x2
            J00 = jacobians[e, g, 0, 0]
            J01 = jacobians[e, g, 0, 1]
            J10 = jacobians[e, g, 1, 0]
            J11 = jacobians[e, g, 1, 1]
            
            # [MATH] Déterminant : (ad - bc)
            detJ = J00 * J11 - J01 * J10
            
            # [MATH] L'inverse transposée calculée scalairement.
            # Intuition : Si le triangle est étiré en X, la pente (gradient) est écrasée en X.
            # L'inverse transposée fait exactement cette correction géométrique.
            invJT_00 =  J11 / detJ
            invJT_01 = -J10 / detJ
            invJT_10 = -J01 / detJ
            invJT_11 =  J00 / detJ
            
            facteur = w[g] * det[e, g] * D_e
            
            for a in range(nloc):
                gr_x = gradN_ref[g, a, 0]
                gr_y = gradN_ref[g, a, 1]

                # Application de la transformation : ∇_reel = (J^-1)^T * ∇_ref
                grad_real_x[a] = invJT_00 * gr_x + invJT_01 * gr_y
                grad_real_y[a] = invJT_10 * gr_x + invJT_11 * gr_y
            
            for a in range(nloc):
                for b in range(nloc):
                    # [MATH] Produit scalaire des gradients réels
                    dot_product = grad_real_x[a] * grad_real_x[b] + grad_real_y[a] * grad_real_y[b]
                    
                    data[idx] = facteur * dot_product
                    rows[idx] = nodes[a]
                    cols[idx] = nodes[b]
                    idx += 1
                    
    return data[:idx], rows[:idx], cols[:idx]

def assemble_mass_or_reaction(nn, ne, nloc, ngp, conn, det, w, N, coeffs):
    d, r, c = _fast_assemble_mass_reaction_core(ne, nloc, ngp, conn, det, w, N, coeffs)
    return coo_matrix((d, (r, c)), shape=(nn, nn)).tocsr()

def assemble_stiffness(nn, ne, nloc, ngp, conn, det, w, jacobians, gradN_ref, coeffs_D):
    d, r, c = _fast_assemble_stiffness_core(ne, nloc, ngp, conn, det, w, jacobians, gradN_ref, coeffs_D)
    return coo_matrix((d, (r, c)), shape=(nn, nn)).tocsr()