import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# On ajoute la racine du projet au PATH pour que Python trouve 'core' et 'physics'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from physics.geometry import ReactorGeometry, ReactorMeshGenerator
from core.solver import solve_dynamic_diffusion # IMPORT CORRIGÉ

def run_parametric_study():
    rayons_test = np.linspace(8.0, 24.0, 5)

    donnees_x_assemblages = []
    donnees_y_flux = []

    for r_noyau in rayons_test:
        geom = ReactorGeometry(R_n=r_noyau, R_hex=2.0)
        centers, tags, _ = geom.get_tagged_assemblies(n_cr_rings=1, cr_density=0.5)
        n_assemblages = len(centers)

        if n_assemblages == 0:
            continue

        params = {
            "R_hex": 2.0,
            "R_reflec": r_noyau + 5.0,
            "cr_rings": 1,
            "cr_density": 0.5,
            "pins_fuel": 4,
            "pins_cr": 3,
        }

        # Création des dossiers si inexistants
        os.makedirs("output/meshes", exist_ok=True)
        mesh_path = f"output/meshes/parametric_{n_assemblages}_assemblages.msh"
        
        generator = ReactorMeshGenerator(geom, params)
        generator.generate(mesh_path)

        # APPEL DU SOLVEUR DÉCOMMENTÉ
        print(f"Calcul en cours pour R={r_noyau:.1f} ({n_assemblages} assemblages)...")
        max_flux = solve_dynamic_diffusion(mesh_path)

        donnees_x_assemblages.append(n_assemblages)
        donnees_y_flux.append(max_flux) # AJOUT CORRIGÉ

    # GRAPHIQUE DÉCOMMENTÉ
    plt.figure()
    plt.plot(donnees_x_assemblages, donnees_y_flux, marker='o', color='red', linewidth=2)
    plt.xlabel("Nombre total d'assemblages")
    plt.ylabel("Flux Neutronique Maximum")
    plt.title("Étude Paramétrique : Flux vs Taille du Cœur")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_parametric_study()