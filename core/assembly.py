import numpy as np
from scipy.sparse import coo_matrix
from numba import njit, prange

@njit(parallel=True)
def _fast_assemble_mass_reaction_core(ne, nloc, ngp, conn, det, w, N, coeffs):
    size = ne * nloc * nloc * ngp
    data = np.zeros(size)
    rows = np.zeros(size, dtype=np.int32)
    cols = np.zeros(size, dtype=np.int32)
    
    # On utilise prange pour répartir les éléments sur tous les cœurs
    for e in prange(ne):
        c_e = coeffs[e]
        nodes = conn[e]
        
        for g in range(ngp):
            facteur = w[g] * det[e, g] * c_e
            
            for a in range(nloc):
                for b in range(nloc):
                    # Calcul de l'index absolu, 100% thread-safe
                    idx = e * (ngp * nloc * nloc) + g * (nloc * nloc) + a * nloc + b
                    
                    data[idx] = facteur * N[g, a] * N[g, b]
                    rows[idx] = nodes[a]
                    cols[idx] = nodes[b]
                    
    # Plus besoin de slicer [:idx] car on remplit exactement la taille 'size'
    return data, rows, cols

@njit(parallel=True)
def _fast_assemble_stiffness_core(ne, nloc, ngp, conn, det, w, jacobians, gradN_ref, coeffs_D):
    size = ne * nloc * nloc * ngp
    data = np.zeros(size)
    rows = np.zeros(size, dtype=np.int32)
    cols = np.zeros(size, dtype=np.int32)

    for e in prange(ne):
        # SÉCURITÉ THREADS : On alloue ces brouillons DANS la boucle prange.
        # Ainsi, chaque cœur du processeur a son propre grad_real_x !
        grad_real_x = np.zeros(nloc)
        grad_real_y = np.zeros(nloc)
        
        D_e = coeffs_D[e]
        nodes = conn[e]
        
        for g in range(ngp):
            J00 = jacobians[e, g, 0, 0]
            J01 = jacobians[e, g, 0, 1]
            J10 = jacobians[e, g, 1, 0]
            J11 = jacobians[e, g, 1, 1]
            
            detJ = J00 * J11 - J01 * J10
            
            invJT_00 =  J11 / detJ
            invJT_01 = -J10 / detJ
            invJT_10 = -J01 / detJ
            invJT_11 =  J00 / detJ
            
            facteur = w[g] * det[e, g] * D_e
            
            for a in range(nloc):
                gr_x = gradN_ref[g, a, 0]
                gr_y = gradN_ref[g, a, 1]
                grad_real_x[a] = invJT_00 * gr_x + invJT_01 * gr_y
                grad_real_y[a] = invJT_10 * gr_x + invJT_11 * gr_y
            
            for a in range(nloc):
                for b in range(nloc):
                    dot_product = grad_real_x[a] * grad_real_x[b] + grad_real_y[a] * grad_real_y[b]
                    
                    # Calcul de l'index absolu
                    idx = e * (ngp * nloc * nloc) + g * (nloc * nloc) + a * nloc + b
                    
                    data[idx] = facteur * dot_product
                    rows[idx] = nodes[a]
                    cols[idx] = nodes[b]
                    
    return data, rows, cols

def assemble_mass_or_reaction(nn, ne, nloc, ngp, conn, det, w, N, coeffs):
    d, r, c = _fast_assemble_mass_reaction_core(ne, nloc, ngp, conn, det, w, N, coeffs)
    return coo_matrix((d, (r, c)), shape=(nn, nn)).tocsr()

def assemble_stiffness(nn, ne, nloc, ngp, conn, det, w, jacobians, gradN_ref, coeffs_D):
    d, r, c = _fast_assemble_stiffness_core(ne, nloc, ngp, conn, det, w, jacobians, gradN_ref, coeffs_D)
    return coo_matrix((d, (r, c)), shape=(nn, nn)).tocsr()