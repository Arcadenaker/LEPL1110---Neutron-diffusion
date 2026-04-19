import numpy as np
import meshio

# Importation de vos modules
from assembly import assemble_stiffness, assemble_mass, assemble_reaction
from time_integration import TimeIntegrator
from boundary_cond import get_dirichlet_nodes

def main():
    print("=== Démarrage du solveur neutronique FEM ===")

    # =========================================================
    # 1. CHARGEMENT DU MAILLAGE (Gmsh)
    # =========================================================
    print("1. Lecture du fichier Gmsh...")
    fichier_maillage = "reactor_mesh.msh"
    mesh = meshio.read(fichier_maillage)
    
    nn = len(mesh.points) # Nombre total de nœuds
    print(f"   -> {nn} nœuds trouvés dans le maillage.")

    # (Ici, vous devriez extraire elemTags, conn, det, w, jacobians, etc.
    # à partir de 'mesh' pour vos fonctions d'assemblage. 
    # Pour l'exemple, on suppose que vous avez une fonction utilitaire pour ça)
    # elemTags, conn, det, w, jacobians, gradN_ref = extract_fem_data(mesh)

    # =========================================================
    # 2. IDENTIFICATION DES CONDITIONS AUX LIMITES
    # =========================================================
    print("2. Application des conditions de bord...")
    # Supposons que dans Gmsh, le bord du réacteur ait le Physical Group numéro 100
    tags_bords = [100] 
    
    # Appel de votre nouveau fichier boundary_cond.py
    dirichlet_dofs = get_dirichlet_nodes(mesh, boundary_physical_tags=tags_bords)
    print(f"   -> {len(dirichlet_dofs)} nœuds identifiés sur la frontière.")
    
    # On impose un flux de 0.0 sur les bords (vide autour du réacteur)
    dirichlet_values = np.zeros(len(dirichlet_dofs))

    # =========================================================
    # 3. ASSEMBLAGE DES MATRICES (core/assembly.py)
    # =========================================================
    print("3. Assemblage des matrices globales...")
    # (On utilise des variables fictives ici pour que le code compile conceptuellement)
    # K = assemble_stiffness(...)
    # M = assemble_mass(...)
    # R = assemble_reaction(...)
    
    # =========================================================
    # 4. INTÉGRATION TEMPORELLE (core/time_integration.py)
    # =========================================================
    print("4. Initialisation de la dynamique spatio-temporelle...")
    
    # Condition initiale : tout à zéro, sauf un pic au centre (nœud 0 par ex)
    phi_0 = np.zeros(nn)
    phi_0[0] = 100.0 

    # On passe nos DDLs de Dirichlet à l'intégrateur
    integrateur = TimeIntegrator(
        M=M, 
        K=K, 
        R=R, 
        dirichlet_dofs=dirichlet_dofs, # <-- C'est ici que ça se connecte !
        theta=0.5
    )

    print("5. Résolution...")
    times, solutions = integrateur.integrate(
        phi_0=phi_0,
        t_span=(0.0, 10.0), # Simulation de 10 secondes
        n_steps=200,
        dirichlet_values=dirichlet_values
    )

    print("=== Simulation terminée avec succès ===")
    
    # (Post-traitement : écriture de 'solutions[-1]' dans un fichier VTK pour Paraview par exemple)

if __name__ == "__main__":
    main()