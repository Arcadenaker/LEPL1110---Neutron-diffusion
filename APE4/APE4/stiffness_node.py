# stiffness.py
import numpy as np
from scipy.sparse import lil_matrix

def assemble_stiffness_and_rhs_node_wise(elemTags, conn, jac, det, xphys, w, N, gN, kappa_fun, rhs_fun, tag_to_dof):
    """
    Assemble global stiffness matrix and load vector for:
        -d/dx (kappa(x) du/dx) = f(x)

    K_ij = ∫ kappa * grad(N_i)·grad(N_j) dx
    F_i  = ∫ f * N_i dx

    Notes:
    - gmsh gives gN in reference coordinates; we map with inv(J).
    - For 1D line embedded in 3D, gmsh provides a 3x3 Jacobian; we keep the same approach.

    Returns
    -------
    K : lil_matrix (nn x nn)
    F : ndarray (nn,)
    """
    ne = len(elemTags)
    ngp = len(w)
    nloc = int(len(conn) // ne)
    nn = int(np.max(tag_to_dof) + 1)

    det = np.asarray(det, dtype=np.float64).reshape(ne, ngp)
    xphys = np.asarray(xphys, dtype=np.float64).reshape(ne, ngp, 3)
    jac = np.asarray(jac, dtype=np.float64).reshape(ne, ngp, 3, 3)
    conn = np.asarray(conn, dtype=np.int64).reshape(ne, nloc)
    N = np.asarray(N, dtype=np.float64).reshape(ngp, nloc)
    gN = np.asarray(gN, dtype=np.float64).reshape(ngp, nloc, 3)

    K = lil_matrix((nn, nn), dtype=np.float64)
    F = np.zeros(nn, dtype=np.float64)

    # Outer loop over every global DOF
    for n in range(nn):
        for e in range(ne):
            # 1. Map Gmsh tags to global DOF indices for this element
            element_tags = conn[e, :]
            dof_indices = tag_to_dof[element_tags]

            # 2. Check if the current DOF 'n' is part of this element
            if n in dof_indices:
                # Find the local index 'a' within the element [0, nloc-1]
                a = np.where(dof_indices == n)[0][0]

                for g in range(ngp):
                    xg = xphys[e, g]
                    wg = w[g]
                    detg = det[e, g]
                    invjacg = np.linalg.inv(jac[e, g])

                    kappa_g = float(kappa_fun(xg))
                    f_g = float(rhs_fun(xg))

                    # Assemble Load Vector F_n
                    F[n] += wg * f_g * N[g, a] * detg

                    # Assemble Stiffness Matrix Row K[n, :]
                    gradNa = invjacg @ gN[g, a]
                    for b in range(nloc):
                        Ib = int(dof_indices[b]) # Mapped column index
                        gradNb = invjacg @ gN[g, b]
                        
                        K[n, Ib] += wg * kappa_g * float(np.dot(gradNa, gradNb)) * detg
    
    return K, F