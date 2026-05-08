import os
import numpy as np
import meshio

from physics.geometry import ReactorGeometry, ReactorMeshGenerator
from core.solver import extract_p1_fem_data
from core.assembly import assemble_mass_or_reaction, assemble_stiffness
from physics.materials import get_material_properties
from core.boundary_cond import get_dirichlet_nodes
from core.time_integration import TimeIntegrator
from utils.logger import get_logger

logger = get_logger(__name__)


def run_validation_test(params, progress_callback=None):
    """
    Exécute le test de validation V&V (Verification & Validation).
    Vérifie la conservation de la population, la positivité et la borne de croissance.
    """
    R_noyau = params.get("R_noyau", 16.0)
    R_hex = params.get("R_hex", 2.0)
    thick = params.get("epaisseur_reflec", 4.0)

    if progress_callback:
        progress_callback(10, 100, "Génération du maillage de test...")

    # 1. Géométrie et Maillage
    geom = ReactorGeometry(R_n=R_noyau, R_hex=R_hex)
    m_params = {
        "R_hex": R_hex,
        "R_reflec": R_noyau + thick,
        "cr_rings": params.get("cr_rings", 0),
        "cr_density": params.get("cr_density", 0.0),
        "pins_fuel": params.get("pins_fuel", 4),
        "pins_cr": params.get("pins_cr", 3),
    }

    out_dir = os.path.join("output", "meshes", "study_case3")
    os.makedirs(out_dir, exist_ok=True)
    mesh_path = os.path.join(out_dir, "temp_validation_v3.msh")

    try:
        generator = ReactorMeshGenerator(geom, m_params)
        generator.generate(mesh_path)

        # 2. Extraction des données FEM
        mesh = meshio.read(mesh_path)
        nn = len(mesh.points)
        conn, det, w, N, jacs, gradN = extract_p1_fem_data(mesh)
        nodes_bc = get_dirichlet_nodes(mesh, ["OuterBoundary"])

        user_map = {
            "Fuel": params.get("mat_fuel", "Fuel_Uranium"),
            "Moderator": params.get("mat_mod", "Water_Moderator"),
            "Reflector": params.get("mat_ref", "Graphite_Moderator"),
            "ControlRods": params.get("mat_cr", "Boral_ControlRod"),
        }

        # 3. Assemblage des matrices physiques
        D, Sa, nSf, inv_v = get_material_properties(
            mesh, conn, rod_insertion=0.5, user_mapping=user_map
        )

        M = assemble_mass_or_reaction(nn, len(conn), 3, len(w), conn, det, w, N, inv_v)
        K = assemble_stiffness(nn, len(conn), 3, len(w), conn, det, w, jacs, gradN, D)
        R_mat = assemble_mass_or_reaction(
            nn, len(conn), 3, len(w), conn, det, w, N, nSf - Sa
        )

        # 4. Intégration Temporelle (Schéma Theta)
        phi_0 = np.ones(nn) * 10.0
        phi_0[nodes_bc] = 0.0  # Respect Dirichlet initialement

        integrateur = TimeIntegrator(M, K, R_mat, nodes_bc, theta=0.5)

        if progress_callback:
            progress_callback(40, 100, "Résolution temporelle...")

        times, solutions, _ = integrateur.integrate(
            phi_0,
            t_span=(0.0, 5.0),
            n_steps=100,
            mesh=mesh,
            elem_tags=conn,
            det=det,
            w=w,
            N=N,
            get_props_func=get_material_properties,
            user_mapping=user_map,
        )

        # 5. Post-traitement et Calcul des indicateurs
        if progress_callback:
            progress_callback(80, 100, "Calcul des intégrales et bornes...")

        # Synchronisation de l'axe temporel (l'intégrateur saute 1 frame sur 2)
        saved_times = np.linspace(times[0], times[-1], len(solutions))

        areas = np.sum(det * w, axis=1)  # Aire de chaque triangle

        pop_totale = []
        num_max = []
        num_min = []

        # Constante de croissance théorique C = max(v * (nuSf - Sa))
        v_local = 1.0 / inv_v
        C_eff = np.max(v_local * (nSf - Sa))
        phi0_inf = np.max(phi_0)
        theo_bound = phi0_inf * np.exp(C_eff * saved_times)

        for sol in solutions:
            # Population : intégrale spatiale P1
            phi_e = np.mean(sol[conn], axis=1)
            pop_totale.append(np.sum(areas * phi_e))

            # Extremums pour les principes du maximum
            num_max.append(np.max(sol))
            num_min.append(np.min(sol))

        return saved_times, num_max, theo_bound, num_min, pop_totale

    finally:
        if os.path.exists(mesh_path):
            os.remove(mesh_path)
