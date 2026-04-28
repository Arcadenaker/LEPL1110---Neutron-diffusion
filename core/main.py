import numpy as np
import meshio
import time

from boundary_cond import get_dirichlet_nodes
from time_integration import TimeIntegrator
from assembly import assemble_mass_or_reaction, assemble_stiffness

def extract_p1_fem_data(mesh):
    """
    [LOGIC] Cette fonction construit le dictionnaire de traduction entre
    notre "Triangle de Référence Parfait" (mathématique) et les "Vrais Triangles" 
    déformés de notre maillage (géométrique).
    """
    triangles = mesh.cells_dict["triangle"]
    ne = len(triangles)
    points = mesh.points
    
    # 3 Points de Gauss pour P1
    ngp = 3
    w = np.array([1/6, 1/6, 1/6]) 
    
    # Fonctions de forme N = [1-u-v, u, v] aux points de Gauss
    N = np.array([
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5]
    ])
    
    # [MATH] Gradients des fonctions de forme sur le triangle de référence.
    # dN1/du = -1, dN2/du = 1, dN3/du = 0.
    # Ces pentes sont constantes partout car c'est un plan incliné (P1).
    gradN_ref = np.zeros((ngp, 3, 2)) # Uniquement X et Y
    for g in range(ngp):
        gradN_ref[g, 0, :] = [-1., -1.]
        gradN_ref[g, 1, :] = [ 1.,  0.]
        gradN_ref[g, 2, :] = [ 0.,  1.]

    # On réduit la dimension à 2x2 pour le Jacobien pour économiser la RAM
    jacobians = np.zeros((ne, ngp, 2, 2))
    dets = np.zeros((ne, ngp))

    for e in range(ne):
        p = points[triangles[e]]
        
        # Vecteurs directeurs X et Y uniquement
        v1 = p[1] - p[0]
        v2 = p[2] - p[0]
        
        # La matrice Jacobienne.
        # Intuition : Elle représente comment l'espace est étiré.
        # Si le déterminant est grand, notre vrai triangle est géant par rapport au modèle.
        J = np.array([
            [v1[0], v2[0]], 
            [v1[1], v2[1]]
        ])
        
        # Déterminant classique 2x2 : ad - bc (L'aire du parallélogramme)
        det_J = abs(v1[0]*v2[1] - v1[1]*v2[0])
        
        for g in range(ngp):
            jacobians[e, g] = J
            dets[e, g] = det_J

    return triangles, dets, w, N, jacobians, gradN_ref


def main():
    print("=== Solveur Neutronique Éléments Finis ===")
    
    mesh = meshio.read("reactor_mesh.msh")
    nn = len(mesh.points)
    
    print("Extraction des propriétés géométriques...")
    conn, det, w, N, jacobians, gradN_ref = extract_p1_fem_data(mesh)
    ne = len(conn)
    ngp = len(w) 
    nloc = 3     
    
    dirichlet_dofs = get_dirichlet_nodes(mesh, [100])
    dirichlet_vals = np.zeros(len(dirichlet_dofs))

    c_D = np.full(ne, 1.5)   
    c_R = np.full(ne, 0.02)  
    c_M = np.full(ne, 1e-5)  

    print(f"Assemblage des matrices ({ne} éléments)...")
    t0 = time.time()
    
    M = assemble_mass_or_reaction(nn, ne, nloc, ngp, conn, det, w, N, c_M)
    R = assemble_mass_or_reaction(nn, ne, nloc, ngp, conn, det, w, N, c_R)
    K = assemble_stiffness(nn, ne, nloc, ngp, conn, det, w, jacobians, gradN_ref, c_D)
    
    print(f"Assemblage terminé en {time.time()-t0:.4f} secondes.")

    phi_0 = np.zeros(nn)
    phi_0[nn//2] = 100.0 

    integrateur = TimeIntegrator(M, K, R, dirichlet_dofs, theta=0.5)
    
    print("Intégration temporelle en cours...")
    times, solutions = integrateur.integrate(phi_0, t_span=(0.0, 1.0), n_steps=50, dirichlet_values=dirichlet_vals)

    print("=== Simulation terminée avec succès ===")

if __name__ == "__main__":
    main()