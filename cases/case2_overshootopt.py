# -*- coding: utf-8 -*-
import os
import numpy as np

from physics.geometry import ReactorGeometry, ReactorMeshGenerator
from core.solver import run_full_simulation
from utils.logger import get_logger

logger = get_logger(__name__)

def run_overshoot_analysis(params, progress_callback=None):
    """
    Analyse spatiale de l'overshoot du contrôleur PD.
    Cartographie l'influence de l'épaisseur du réflecteur face à la position radiale 
    d'un unique anneau de contrôle (du centre vers la périphérie).
    """
    R_noyau = params.get("R_noyau", 12.0)
    R_hex = params.get("R_hex", 2.0)
    
    # --- 1. IDENTIFICATION DE LA TOPOLOGIE ---
    geom_base = ReactorGeometry(R_n=R_noyau, R_hex=R_hex)
    centers_base = geom_base.hex_centers()
    
    if len(centers_base) == 0:
        raise ValueError("Le noyau est trop petit pour contenir des assemblages.")
        
    B = np.round(centers_base[:, 1] / (1.5 * R_hex))
    A = np.round(centers_base[:, 0] / (R_hex * np.sqrt(3)) - B / 2)
    d_hex = np.maximum.reduce([np.abs(A), np.abs(B), np.abs(A + B)]).astype(int)
    
    D_max = np.max(d_hex) if d_hex.size > 0 else 0
    if D_max < 1:
        raise ValueError("Le cœur ne contient pas suffisamment d'anneaux pour mener l'étude.")

    # --- 2. DÉFINITION DES AXES D'ÉTUDE ---
    max_thick = params.get("max_thickness", 12.0)
    thicknesses = np.arange(2.0, max_thick + 0.1, 2.0)  # Épaisseurs : 2.0, 4.0, 6.0...
    target_rings = np.arange(1, D_max + 1)              # Anneaux : de 1 (centre) à D_max (bord)
    
    overshoot_matrix = np.zeros((len(thicknesses), len(target_rings)))
    
    total_steps = len(thicknesses) * len(target_rings)
    current_step = 0

    out_dir = os.path.join("output", "meshes", "study_case2")
    os.makedirs(out_dir, exist_ok=True)

    user_mapping = {
        "Fuel": params.get("mat_fuel", "Fuel_Uranium"),
        "Moderator": params.get("mat_mod", "Water_Moderator"),
        "Reflector": params.get("mat_ref", "Graphite_Moderator"),
        "ControlRods": params.get("mat_cr", "Boral_ControlRod")
    }

    # --- 3. ITÉRATION SUR LA GRILLE 2D ---
    for i, thick in enumerate(thicknesses):
        for j, ring_idx in enumerate(target_rings):
            current_step += 1
            msg = f"Réflecteur: {thick}cm | Anneau CR: {ring_idx} | [{current_step}/{total_steps}]"
            if progress_callback:
                progress_callback(current_step, total_steps, msg)

            geom = ReactorGeometry(R_n=R_noyau, R_hex=R_hex)
            
            # --- SURCHARGE LOCALE DE LA GÉOMÉTRIE ---
            # On force la valeur de `ring_idx` dans les paramètres par défaut pour éviter les effets de closure
            def custom_tagged_assemblies(n_cr_rings=1, cr_density=1.0, r_idx=ring_idx):
                centers = geom.hex_centers()
                B_c = np.round(centers[:, 1] / (1.5 * R_hex))
                A_c = np.round(centers[:, 0] / (R_hex * np.sqrt(3)) - B_c / 2)
                d = np.maximum.reduce([np.abs(A_c), np.abs(B_c), np.abs(A_c + B_c)]).astype(int)
                
                tags = np.full(len(centers), "FUEL", dtype=object)
                
                # Assigne 100% de barres 'CR' UNIQUEMENT sur l'anneau ciblé
                mask_cr = (d == r_idx)
                tags[mask_cr] = "CR"
                
                return centers, tags, d
                
            geom.get_tagged_assemblies = custom_tagged_assemblies
            
            m_params = {
                "R_hex": R_hex,
                "R_reflec": R_noyau + thick,
                "cr_rings": 1,
                "cr_density": 1.0,
                "pins_fuel": params.get("pins_fuel", 4),
                "pins_cr": params.get("pins_cr", 3),
            }

            mesh_path = os.path.join(out_dir, f"mesh_T{thick}_R{ring_idx}.msh")
            generator = ReactorMeshGenerator(geom, m_params)

            try:
                # Génération du maillage et exécution Headless
                generator.generate(mesh_path)
                solutions, times, puissance_history = run_full_simulation(
                    mesh_path,
                    user_mapping=user_mapping,
                    headless=True
                )
                
                # --- 4. CALCUL DE L'OVERSHOOT ---
                p_cible = np.mean(puissance_history[-10:])
                p_max = np.max(puissance_history)
                
                if p_cible > 0:
                    overshoot = ((p_max - p_cible) / p_cible) * 100.0
                    overshoot_matrix[i, j] = max(0.0, overshoot)
                else:
                    overshoot_matrix[i, j] = np.nan
                    
            except Exception as e:
                logger.error(f"Échec {msg} : {e}", exc_info=True)
                overshoot_matrix[i, j] = np.nan
                
            finally:
                # Nettoyage des fichiers temporaires
                if os.path.exists(mesh_path):
                    try:
                        os.remove(mesh_path)
                    except OSError:
                        pass

    return thicknesses, target_rings, overshoot_matrix
