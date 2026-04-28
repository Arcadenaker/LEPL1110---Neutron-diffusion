import numpy as np
import matplotlib.pyplot as plt
from physics.geometry import ReactorGeometry, ReactorMeshGenerator
# import de ton futur solveur :
# from core.solvers import solve_dynamic_diffusion


def run_parametric_study():
    # 1. Définir l'espace paramétrique (variation du rayon du noyau)
    rayons_test = np.linspace(8.0, 24.0, 5)

    donnees_x_assemblages = []
    donnees_y_flux = []

    for r_noyau in rayons_test:
        # --- A. GÉNÉRATION ---
        geom = ReactorGeometry(R_n=r_noyau, R_hex=2.0)
        centers, tags, _ = geom.get_tagged_assemblies(n_cr_rings=1, cr_density=0.5)
        n_assemblages = len(centers)

        # Ignorer les géométries vides ou non pertinentes
        if n_assemblages == 0:
            continue

        params = {
            "R_hex": 2.0,
            "R_reflec": r_noyau + 5.0,  # Adapter le réflecteur dynamiquement
            "cr_rings": 1,
            "cr_density": 0.5,
            "pins_fuel": 4,
            "pins_cr": 3,
        }

        mesh_path = f"output/meshes/parametric_{n_assemblages}_assemblages.msh"
        generator = ReactorMeshGenerator(geom, params)
        generator.generate(mesh_path)

        # --- B. RÉSOLUTION ÉLÉMENTS FINIS ---
        # Exécution silencieuse de ton solveur sur ce maillage
        # resultats = solve_dynamic_diffusion(mesh_path)

        # --- C. EXTRACTION ---
        donnees_x_assemblages.append(n_assemblages)
        # donnees_y_flux.append(resultats.get_max_flux())

    # 2. Tracer le graphique final
    plt.figure()
    # plt.plot(donnees_x_assemblages, donnees_y_flux, marker='o')
    plt.xlabel("Nombre total d'assemblages")
    plt.ylabel("Observable (ex: Flux Max ou k_eff)")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    run_parametric_study()
