import numpy as np
import meshio
import os

from physics.geometry import ReactorGeometry, ReactorMeshGenerator
from core.solver import extract_p1_fem_data, pos_rodBar_init
from core.assembly import assemble_stiffness
from physics.materials import get_material_properties
from core.boundary_cond import get_dirichlet_nodes


def run_discrete_brute_force(params, progress_callback=None):
    """
    Étude statique par force brute sur des configurations discrètes (Anneaux x Densité).
    Densités strictes : 0.5 (1/2) et 1.0 (Full).
    """
    max_rings = params.get("max_rings", 3)
    densities_to_test = [0.5, 1.0]

    # Génération des combinaisons (ex: 1 anneau 50%, 1 anneau 100%, 2 anneaux 50%...)
    combinations = []
    for r in range(1, max_rings + 1):
        for d in densities_to_test:
            combinations.append({"rings": r, "density": d})

    fuels = ["Fuel_Uranium", "Fuel_MOX"]
    results = {fuel: [] for fuel in fuels}
    labels = []  # Pour l'axe X du graphique

    out_dir = "output/meshes/study_A"
    os.makedirs(out_dir, exist_ok=True)

    total_steps = len(combinations)

    for i, config in enumerate(combinations):
        r_cr = config["rings"]
        d_cr = config["density"]

        lbl = f"{r_cr} Couronne{'s' if r_cr > 1 else ''}\n({int(d_cr * 100)}%)"
        labels.append(lbl)

        if progress_callback:
            progress_callback(
                i, total_steps, f"Test {r_cr} anneau(x) à {int(d_cr * 100)}%"
            )

        # --- ÉTAPE 1 : Géométrie avec les paramètres originaux ---
        geom = ReactorGeometry(R_n=params["R_noyau"], R_hex=params["R_hex"])

        # Vérification rapide pour éviter de mailler si la couronne sort du cœur
        _, tags, _ = geom.get_tagged_assemblies(n_cr_rings=r_cr, cr_density=d_cr)
        if np.sum(tags == "CR") == 0:
            # Si la combinaison ne génère aucune barre (coeur trop petit), on zappe
            for fuel in fuels:
                results[fuel].append(1.0)  # On force 1.0 (échec)
            continue

        m_params = {
            "R_hex": params["R_hex"],
            "R_reflec": params["R_noyau"] + params["epaisseur_reflec"],
            "cr_rings": r_cr,
            "cr_density": d_cr,
            "pins_fuel": params["pins_fuel"],
            "pins_cr": params["pins_cr"],
        }

        mesh_path = os.path.join(out_dir, f"static_R{r_cr}_D{d_cr:.1f}.msh")
        generator = ReactorMeshGenerator(geom, m_params)
        generator.generate(mesh_path)

        # --- ÉTAPE 2 : Extraction FEM ---
        mesh = meshio.read(mesh_path)
        nn = len(mesh.points)
        conn, det, w, N, jacs, gradN = extract_p1_fem_data(mesh)
        nodes_bc = get_dirichlet_nodes(mesh, ["OuterBoundary"])

        # --- ÉTAPE 3 : Évaluation des Matériaux ---
        for fuel in fuels:
            user_map = {
                "Fuel": fuel,
                "Moderator": params.get("mat_mod", "Water_Moderator"),
                "Reflector": params.get("mat_ref", "Graphite_Moderator"),
                "ControlRods": params.get("mat_cr", "Boral_ControlRod"),
            }

            c_D, _, _, _ = get_material_properties(mesh, conn, 1.0, user_map)
            K = assemble_stiffness(
                nn, len(conn), 3, len(w), conn, det, w, jacs, gradN, c_D
            )

            z_eq = pos_rodBar_init(
                mesh,
                conn,
                det,
                w,
                N,
                get_material_properties,
                K,
                nn,
                user_map,
                nodes_bc,
            )
            results[fuel].append(z_eq)

    if progress_callback:
        progress_callback(total_steps, total_steps, "Force brute terminée")

    return labels, results
