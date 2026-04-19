import numpy as np
from scipy.sparse import lil_matrix

# Fonctions qui font le lien entre  GMSH et le solver et qui 
# calcule le lien entre les différents noeuds et élément

def assemble_reaction(elemTags, conn, det, w, N, coeffs_reaction):
    """
    This function takes the element information as parameters:
        - elemTags: the elements in question.
        - conn: the nodes that make up these elements.
        
        "In the finite element method, we don't compute the solution continuously everywhere; we interpolate it from the vertices (the nodes). 
        To do this, we use shape functions (N). They act like a spatial 'cross-fade' system: if you are exactly on a node, its shape function is 1 (100% influence) and the others are 0. If you are in the middle of the triangle, each node pulls the solution towards itself with a certain percentage."
        - det: scaling factor that tells us whether the perfect reference triangle has stretched/shrunk to become the real mesh triangle.
        - w: Gauss points. These points represent an area of the triangle. They indicate what percentage of the triangle's total area this point represents. The sum of the weights equals the total area.
        
        - N: represents the value of these famous shape functions (the "influence percentages" of each node) evaluated at the Gauss points on the reference triangle.
        - coeffs_reaction: the net balance of creation/disappearance in the medium (e.g., in a reactor, this is neutron production minus absorption: nu*Sigma_f - Sigma_a).

    Assembles the global reaction matrix:
        R_ij = sum_e ∫_e (nu*Sigma_f - Sigma_a) * N_i * N_j dx
    """
    ne = len(elemTags)
    ngp = len(w)
    nloc = int(len(conn) // ne)
    nn = int(np.max(conn))

    det = np.asarray(det, dtype=np.float64).reshape(ne, ngp)
    conn = np.asarray(conn, dtype=np.int64).reshape(ne, nloc)
    N = np.asarray(N, dtype=np.float64).reshape(ngp, nloc)
    coeffs_reaction = np.asarray(coeffs_reaction, dtype=np.float64)

    R = lil_matrix((nn, nn), dtype=np.float64)

    # Boucle sur les éléments (triangles)
    for e in range(ne):
        nodes = conn[e, :] - 1  # Passage en base 0 pour Python
        coeff_e = coeffs_reaction[e] # Extraction du coeff (nu*Sig_f - Sig_a) pour CE triangle
        
        # Boucle sur les points de la Quadrature de Gauss
        for g in range(ngp):
            wg = w[g]
            detg = det[e, g]
            
            # Constante globale pour ce point d'intégration : Poids * Jacobien * Materiau
            facteur = wg * detg * coeff_e 
            
            # Double boucle sur les nœuds locaux (a et b) de l'élément (ex: 0 à 5 pour P2)
            for a in range(nloc):
                Ia = int(nodes[a])
                Na = N[g, a]
                
                for b in range(nloc):
                    Ib = int(nodes[b])
                    Nb = N[g, b]
                    
                    # Math : R_ab += facteur * N_a * N_b
                    R[Ia, Ib] += facteur * Na * Nb

    return R




def assemble_stiffness(elemTags, conn, det, w, jacobians, gradN_ref, coeffs_D):
    """
    This function takes the element information as parameters:
        - elemTags: the elements in question.
        - conn: the nodes that make up these elements.
        
        "The brilliant trick of finite elements is to say: 'I refuse to do math on the real Gmsh triangles.'
        Instead, mathematicians invented a perfect and unique Reference Triangle (for example, a right triangle with vertices (0,0), (1,0) and (0,1)).
        We do all the heavy mathematical calculations once and for all on this perfect template. Then, we 'deform' this template to match each real triangle in the mesh."
        - det: scaling factor that tells us whether the triangle has stretched/shrunk.
        - w: Gauss points. These points represent an area of the triangle. They indicate what percentage of the triangle's total area this point represents. The sum of the weights equals the total area.
        
        - gradN_ref: represents the slopes (derivatives) of the shape functions on the perfect triangle.
        - coeffs_D: the diffusion capacity of the medium (moderator/fuel/etc.).

    Assembles the global stiffness (diffusion) matrix:
        K_ij = sum_e ∫_e D * (∇N_i · ∇N_j) dx
    """
    ne = len(elemTags)
    ngp = len(w)
    nloc = int(len(conn) // ne)
    nn = int(np.max(conn))

    # Reshape des données basiques
    det = np.asarray(det, dtype=np.float64).reshape(ne, ngp)
    conn = np.asarray(conn, dtype=np.int64).reshape(ne, nloc)
    coeffs_D = np.asarray(coeffs_D, dtype=np.float64)
    
    # -------------------------------------------------------------
    # RESHAPE SPÉCIFIQUE POUR LA RIGIDITÉ
    # jacobians : matrice 3x3 aplatie pour chaque élément et chaque point de Gauss
    jacobians = np.asarray(jacobians, dtype=np.float64).reshape(ne, ngp, 3, 3)
    
    # gradN_ref : vecteur (dNu, dNv, dNw) pour chaque point et chaque nœud
    gradN_ref = np.asarray(gradN_ref, dtype=np.float64).reshape(ngp, nloc, 3)
    # -------------------------------------------------------------

    K = lil_matrix((nn, nn), dtype=np.float64)

    # Boucle sur les éléments
    for e in range(ne):
        nodes = conn[e, :] - 1
        D_e = coeffs_D[e] # Coefficient de diffusion de cet élément
        
        # Boucle sur les points de Gauss
        for g in range(ngp):
            wg = w[g]
            detg = det[e, g]
            facteur = wg * detg * D_e
            
            # --- LA MAGIE DU JACOBIEN ---
            # On extrait la sous-matrice 2x2 du Jacobien (car on est en 2D : x,y et u,v)
            J_2x2 = jacobians[e, g, 0:2, 0:2] 
            
            # On calcule l'inverse transposée du Jacobien : (J^-1)^T
            inv_J_T = np.linalg.inv(J_2x2).T
            
            # Pré-calcul des vrais gradients (dN/dx, dN/dy) pour tous les nœuds locaux
            gradN_real = np.zeros((nloc, 2))
            for a in range(nloc):
                # Gradient de référence 2D : [dN_a/du, dN_a/dv]
                grad_ref_a = gradN_ref[g, a, 0:2] 
                
                # Math : ∇N_reel = (J^-1)^T * ∇N_ref
                gradN_real[a, :] = inv_J_T.dot(grad_ref_a)
            # -----------------------------
            
            # Double boucle d'assemblage
            for a in range(nloc):
                Ia = int(nodes[a])
                grad_a = gradN_real[a, :] # Vecteur [dN_a/dx, dN_a/dy]
                
                for b in range(nloc):
                    Ib = int(nodes[b])
                    grad_b = gradN_real[b, :] # Vecteur [dN_b/dx, dN_b/dy]
                    
                    # Math : Produit scalaire ∇N_a · ∇N_b = (dN_a/dx * dN_b/dx) + (dN_a/dy * dN_b/dy)
                    produit_scalaire = np.dot(grad_a, grad_b)
                    
                    # Ajout à la matrice K globale
                    K[Ia, Ib] += facteur * produit_scalaire

    return K