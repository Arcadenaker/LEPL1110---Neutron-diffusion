import os
import numpy as np

from physics.geometry import ReactorGeometry, ReactorMeshGenerator
from core.solver import run_full_simulation

# --- IMPORT DU LOGGER ---
from utils.logger import get_logger

logger = get_logger(__name__)
# ------------------------

def run_overshoot_analysis(params, progress_callback=None):
    """
    Étude paramétrique de l'Overshoot.
    Renvoie une grille 2D de dictionnaires décrivant l'état physique du réacteur
    et l'overshoot calculé si la stabilisation est réussie.
    """
    logger.info("Début de l'analyse matricielle d'overshoot (Cas 2)")
    
    r_noyau = params['R_noyau']
    max_thick = params.get('max_thickness', 12.0)
    
    # 1. Définition de l'espace de recherche (Grilles)
    thicknesses = np.arange(2.0, max_thick + 1.0, 2.0)
    target_rings = np.arange(1, 5)
    
    # Structure de données explicite remplaçant la score_matrix
    results_grid = []
    
    total_steps = len(thicknesses) * len(target_rings)
    current_step = 0
    
    out_dir = "output/meshes/study_B"
    os.makedirs(out_dir, exist_ok=True)
    
    user_map = {
        "Fuel": params.get("mat_fuel", "Fuel_Uranium"),
        "Moderator": params.get("mat_mod", "Water_Moderator"),
        "Reflector": params.get("mat_ref", "Graphite_Moderator"),
        "ControlRods": params.get("mat_cr", "Boral_ControlRod"),
    }
    
    for thick in thicknesses:
        row_results = []
        for rings in target_rings:
            current_step += 1
            if progress_callback:
                msg = f"Réflecteur : {thick}cm | CR : {rings} couche(s) ({current_step}/{total_steps})"
                progress_callback(current_step, total_steps, msg)
                
            # Initialisation de la structure pour cette combinaison
            cell_data = {
                "status": "INCONNU",
                "overshoot_mw": None
            }
            
            # -- ÉTAPE A : GÉOMÉTRIE ET MAILLAGE --
            geom = ReactorGeometry(R_n=r_noyau, R_hex=params["R_hex"])
            cr_density = params.get('cr_density', 1.0)
            
            # Vérification de la constructibilité
            _, tags, _ = geom.get_tagged_assemblies(n_cr_rings=rings, cr_density=cr_density)
            if np.sum(tags == "CR") == 0:
                cell_data["status"] = "ERREUR_GEOM"
                row_results.append(cell_data)
                continue
                
            m_params = {
                "R_hex": params["R_hex"],
                "R_reflec": r_noyau + thick,
                "cr_rings": rings,
                "cr_density": cr_density,
                "pins_fuel": params.get("pins_fuel", 4),
                "pins_cr": params.get("pins_cr", 3),
            }
            
            mesh_path = os.path.join(out_dir, f"heatmap_T{thick}_R{rings}.msh")
            generator = ReactorMeshGenerator(geom, m_params)
            
            try:
                generator.generate(mesh_path)
            except Exception as e:
                logger.error(f"Échec de la génération du maillage T{thick}-R{rings}: {e}")
                cell_data["status"] = "ERREUR_SOLVEUR"
                row_results.append(cell_data)
                continue
                
            # -- ÉTAPE B : SIMULATION DU TRANSOIRE --
            try:
                _, times, p_history = run_full_simulation(
                    mesh_path, 
                    user_mapping=user_map, 
                    headless=True,
                    kd_value=0.5
                )
                
                puissance_initiale = p_history[0]
                target_power = puissance_initiale * 1.50
                final_power = p_history[-1]
                max_power = np.max(p_history)
                
                # -- ÉTAPE C : DIAGNOSTIC STRICT --
                if final_power < target_power * 0.95:
                    cell_data["status"] = "SOUS_CRITIQUE"
                    
                elif final_power > target_power * 1.05 or max_power > target_power * 2.5:
                    cell_data["status"] = "SUR_CRITIQUE"
                    
                else:
                    cell_data["status"] = "STABLE"
                    cell_data["overshoot_mw"] = max(0.0, max_power - target_power)
                    
            except Exception as e:
                logger.error(f"Divergence / Échec solveur T{thick}-R{rings}: {e}")
                # Un crash numérique est classifié comme une excursion de puissance incontrôlable
                cell_data["status"] = "SUR_CRITIQUE"
            
            row_results.append(cell_data)
        
        results_grid.append(row_results)
                
    if progress_callback:
        progress_callback(total_steps, total_steps, "Analyse terminée.")
        
    return thicknesses, target_rings, results_grid
