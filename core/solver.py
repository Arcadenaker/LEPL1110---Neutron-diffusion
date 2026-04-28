import numpy as np
import meshio
import matplotlib.pyplot as plt

from core.boundary_cond import get_dirichlet_nodes
from core.time_integration import TimeIntegrator
from core.assembly import assemble_mass_or_reaction, assemble_stiffness
from physics.materials import get_material_properties

def extract_p1_fem_data(mesh):
    """
    Construit le dictionnaire de traduction entre le Triangle de Référence Parfait 
    et les Vrais Triangles déformés du maillage.
    """
    triangles = mesh.cells_dict["triangle"]
    ne = len(triangles)
    points = mesh.points
    
    ngp = 3
    w = np.array([1/6, 1/6, 1/6]) 
    N = np.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]])
    
    gradN_ref = np.zeros((ngp, 3, 2))
    for g in range(ngp):
        gradN_ref[g, 0, :] = [-1., -1.]
        gradN_ref[g, 1, :] = [ 1.,  0.]
        gradN_ref[g, 2, :] = [ 0.,  1.]

    jacobians = np.zeros((ne, ngp, 2, 2))
    dets = np.zeros((ne, ngp))

    for e in range(ne):
        p = points[triangles[e]]
        v1 = p[1] - p[0]
        v2 = p[2] - p[0]
        
        J = np.array([[v1[0], v2[0]], [v1[1], v2[1]]])
        det_J = abs(v1[0]*v2[1] - v1[1]*v2[0])
        
        for g in range(ngp):
            jacobians[e, g] = J
            dets[e, g] = det_J

    return triangles, dets, w, N, jacobians, gradN_ref


def run_full_simulation(mesh_path):
    """Lit le maillage, résout l'équation et affiche le résultat."""
    print(f"Chargement du maillage : {mesh_path}")
    mesh = meshio.read(mesh_path)
    nn = len(mesh.points)
    
    # 1. Extraction des données géométriques
    conn, det, w, N, jacobians, gradN_ref = extract_p1_fem_data(mesh)
    ne = len(conn)
    ngp = len(w)
    nloc = 3
    
    # 2. Physique réelle via materials.py (avec 'conn' passé en argument)
    print("Application des propriétés physiques...")
    c_D, c_Sigma_a, c_nuSigma_f, c_inv_v = get_material_properties(mesh, conn)
    c_R = c_nuSigma_f - c_Sigma_a

    # 3. Assemblage des matrices
    print(f"Assemblage des matrices ({ne} éléments)...")
    M = assemble_mass_or_reaction(nn, ne, nloc, ngp, conn, det, w, N, c_inv_v)
    R = assemble_mass_or_reaction(nn, ne, nloc, ngp, conn, det, w, N, c_R)
    K = assemble_stiffness(nn, ne, nloc, ngp, conn, det, w, jacobians, gradN_ref, c_D)

    # 4. Conditions aux limites (Tag 1000 pour le bord extérieur)
    dirichlet_dofs = get_dirichlet_nodes(mesh, [1000])
    
    # 5. Intégration Temporelle
    print("Démarrage de la simulation temporelle...")
    integrateur = TimeIntegrator(M, K, R, dirichlet_dofs, theta=0.5)
    
    phi_0 = np.zeros(nn)
    phi_0[nn//2] = 100.0 # Impulsion centrale
    
    times, solutions = integrateur.integrate(phi_0, t_span=(0.0, 1.0), n_steps=50)
    
    # 6. Visualisation du résultat final
    print("Génération de l'affichage...")
    plt.figure(figsize=(8, 6))
    plt.tricontourf(mesh.points[:,0], mesh.points[:,1], solutions[-1], levels=50, cmap='inferno')
    plt.colorbar(label="Flux Neutronique")
    plt.title("Carte de chaleur : Flux Neutronique Final")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    
    return solutions[-1]