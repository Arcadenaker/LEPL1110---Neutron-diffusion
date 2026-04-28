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
    dirichlet_dofs = get_dirichlet_nodes(mesh, ["OuterBoundary"])
    
    # 5. Intégration Temporelle
    print("Démarrage de la simulation temporelle...")
    integrateur = TimeIntegrator(M, K, R, dirichlet_dofs, theta=1.0)
    
    # --- NOUVELLE INITIALISATION PHYSIQUE ---
    # On va chercher tous les nœuds (points) qui composent le combustible
    fuel_nodes = []
    if "Fuel" in mesh.cell_sets:
        for block_id, elem_indices in enumerate(mesh.cell_sets["Fuel"]):
            cell_block = mesh.cells[block_id]
            if cell_block.type == 'triangle':
                # On récupère tous les sommets des triangles de ce bloc
                nodes = cell_block.data[elem_indices]
                fuel_nodes.extend(nodes.flatten())
                
    fuel_nodes = np.unique(fuel_nodes) # On supprime les doublons
    
    # Création du flux initial : 0 partout...
    phi_0 = np.zeros(nn)
    
    # ... sauf dans le combustible où on allume "les braises" !
    if len(fuel_nodes) > 0:
        phi_0[fuel_nodes] = 100.0
    else:
        # Fallback de sécurité si aucun fuel n'est trouvé
        phi_0[np.argmin(mesh.points[:, 0]**2 + mesh.points[:, 1]**2)] = 100.0
    # ----------------------------------------

    times, solutions = integrateur.integrate(phi_0, t_span=(0.0, 0.002), n_steps=200)
    
    # 6. Visualisation du résultat final (Animation du transitoire)
    print("Génération de l'animation...")
    from matplotlib.animation import FuncAnimation
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.axis('off') # On enlève les axes pour un rendu plus esthétique (mode sombre)
    fig.patch.set_facecolor('#1e1e1e') # Fond gris foncé style VS Code
    
    # On utilise tripcolor avec shading='gouraud' pour un lissage parfait, c'est bien plus beau que tricontourf
    # On utilise la palette 'magma' ou 'plasma' qui ont de très beaux dégradés
    mesh_plot = ax.tripcolor(mesh.points[:,0], mesh.points[:,1], mesh.cells_dict["triangle"], 
                             solutions[0], shading='gouraud', cmap='magma')
    
    # Configuration de la barre de couleur
    cbar = fig.colorbar(mesh_plot, ax=ax, shrink=0.8)
    cbar.set_label("Flux Neutronique", color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    title = ax.set_title("Temps : 0.0 s", color='white', fontsize=14)

    def animate(i):
        # 1. On met à jour les données de flux pour l'image courante
        mesh_plot.set_array(solutions[i])
        
        # 2. TRÈS IMPORTANT : On ajuste dynamiquement l'échelle de couleurs.
        # Au fur et à mesure que les neutrons diffusent, le pic maximum diminue.
        # Si on ne fait pas ça, l'image deviendrait de plus en plus noire.
        vmax_current = np.max(solutions[i])
        if vmax_current < 1e-5: 
            vmax_current = 1e-5 # Sécurité pour éviter la division par zéro
        mesh_plot.set_clim(vmin=0, vmax=vmax_current)
        
        # 3. Mise à jour du chrono
        title.set_text(f"Temps : {times[i]:.5f} s")
        return mesh_plot, title

    # Lancement de l'animation (interval=50 ms entre chaque image)
    ani = FuncAnimation(fig, animate, frames=len(solutions), interval=50, blit=False)
    
    plt.tight_layout()
    plt.show()
    
    return solutions[-1]