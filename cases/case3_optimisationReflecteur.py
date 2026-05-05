import numpy as np
import os
import sys

# On s'assure que Python trouve les dossiers à la racine du projet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from physics.geometry import ReactorGeometry, ReactorMeshGenerator
from core.solver import run_full_simulation

def run_reflector_optimization(params, progress_callback=None):
    """
    Étude d'optimisation du réflecteur.
    Génère les maillages et calcule le facteur de forme (Fq) pour différentes épaisseurs.
    """
    r_noyau = params["R_noyau"]
    max_ep = params.get("max_epaisseur", 16.0)
    
    # Génération de 5 points de test (de 0 cm à l'épaisseur max)
    epaisseurs_test = np.linspace(0.0, max_ep, 5)
    donnees_fq = []
    
    out_dir = "output/meshes/study_3"
    os.makedirs(out_dir, exist_ok=True)
    
    total_steps = len(epaisseurs_test)
    
    # Mapping des matériaux depuis l'interface
    user_map = {
        "Fuel": params.get("mat_fuel", "Fuel_Uranium"),
        "Moderator": params.get("mat_mod", "Water_Moderator"),
        "Reflector": params.get("mat_ref", "Graphite_Moderator"),
        "ControlRods": params.get("mat_cr", "Boral_ControlRod")
    }

    for i, epaisseur in enumerate(epaisseurs_test):
        if progress_callback:
            progress_callback(i, total_steps, f"Test épaisseur = {epaisseur:.1f} cm")

        geom = ReactorGeometry(R_n=r_noyau, R_hex=params["R_hex"])
        
        m_params = {
            "R_hex": params["R_hex"],
            "R_reflec": r_noyau + epaisseur,
            "cr_rings": params.get("cr_rings", 1),
            "cr_density": params.get("cr_density", 0.5),
            "pins_fuel": params.get("pins_fuel", 4),
            "pins_cr": params.get("pins_cr", 3),
        }
        
        mesh_path = os.path.join(out_dir, f"opti_reflec_{epaisseur:.1f}cm.msh")
        generator = ReactorMeshGenerator(geom, m_params)
        generator.generate(mesh_path)

        # Calcul (Headless mode)
        try:
            solutions, _, _ = run_full_simulation(mesh_path, user_mapping=user_map, headless=True)
            solution_finale = solutions[-1]
            
            flux_moyen = np.mean(solution_finale)
            flux_max = np.max(solution_finale)
            fq = flux_max / flux_moyen if flux_moyen > 1e-10 else 1.0
        except Exception as e:
            # En cas d'échec du solveur pour une configuration extrême
            print(f"Erreur solveur (ep={epaisseur}) : {e}")
            fq = float('inf') 
            
        donnees_fq.append(fq)

    if progress_callback:
        progress_callback(total_steps, total_steps, "Optimisation terminée")
        
    return epaisseurs_test, donnees_fq