import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# On s'assure que Python trouve les dossiers 'core' et 'physics' à la racine du projet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from physics.geometry import ReactorGeometry, ReactorMeshGenerator
from core.solver import run_full_simulation

def optimize_reflector():
    print("=====================================================")
    print("--- DÉMARRAGE DE L'ÉTUDE : OPTIMISATION RÉFLECTEUR ---")
    print("=====================================================")
    
    # Rayon du cœur fixe (12 cm)
    r_noyau = 12.0 
    
    # On teste 5 épaisseurs stratégiques (0 cm = pas de réflecteur, jusqu'à 16 cm)
    epaisseurs_test = np.array([0.0, 4.0, 8.0, 12.0, 16.0])
    
    donnees_fq = []

    # Création du dossier de maillage s'il n'existe pas
    os.makedirs("output/meshes", exist_ok=True)

    for epaisseur in epaisseurs_test:
        print(f"\n[+] ÉTAPE : Réflecteur = {epaisseur} cm")
        print("    -> Génération du maillage...")
        
        geom = ReactorGeometry(R_n=r_noyau, R_hex=2.0)
        centers, tags, _ = geom.get_tagged_assemblies(n_cr_rings=1, cr_density=0.5)

        params = {
            "R_hex": 2.0,
            "R_reflec": r_noyau + epaisseur,  # L'épaisseur varie ici !
            "cr_rings": 1,
            "cr_density": 0.5,
            "pins_fuel": 4,
            "pins_cr": 3,
        }

        mesh_path = f"output/meshes/opti_reflec_{epaisseur}cm.msh"
        generator = ReactorMeshGenerator(geom, params)
        generator.generate(mesh_path)

        print("    -> Calcul de la diffusion en cours (Headless mode)...")
        # On récupère les solutions, on ignore le temps et la puissance (_)
        solutions, _, _ = run_full_simulation(mesh_path, headless=True)
        solution_finale = solutions[-1]
        
        # --- CALCUL DU FACTEUR DE FORME (Fq) ---
        # Fq = Flux Max / Flux Moyen. Plus c'est bas (proche de 1), mieux c'est !
        flux_moyen = np.mean(solution_finale)
        flux_max = np.max(solution_finale)
        
        # Sécurité division par zéro
        fq = flux_max / flux_moyen if flux_moyen > 1e-10 else 1.0
        
        donnees_fq.append(fq)
        print(f"    => Facteur de forme calculé : Fq = {fq:.2f}")

    # --- AFFICHAGE DU GRAPHIQUE OPTIMUM ---
    print("\n[+] Génération du graphique final...")
    plt.figure(figsize=(8, 5))
    plt.plot(epaisseurs_test, donnees_fq, marker='s', color='#2ca02c', linewidth=2, markersize=8)
    
    # On met en évidence l'optimum visuel (saturation de l'effet d'albédo)
    plt.axvline(8.0, color='red', linestyle='--', alpha=0.5, label="Optimum d'ingénierie estimé (~8 cm)")
    
    plt.xlabel("Épaisseur du réflecteur en graphite (cm)", fontsize=12)
    plt.ylabel("Facteur de Forme $F_q$ (Max/Moyen)", fontsize=12)
    plt.title("Optimisation du Réflecteur : Aplatissement du Flux", fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    
    # Sauvegarde de la figure
    os.makedirs("output/graphs", exist_ok=True)
    plt.savefig("output/graphs/optimisation_reflecteur.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    optimize_reflector()