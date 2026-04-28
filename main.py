import os
from utils.gui import ReactorGUI
from core.solver import run_full_simulation

def main():
    print("=== LEPL1110 : SIMULATEUR NEUTRONIQUE ===")
    print("1. Concevoir le cœur (GUI + Maillage)")
    print("2. Résoudre le dernier maillage généré")
    print("3. Quitter")
    
    choix = input("\nVotre choix : ")
    
    if choix == "1":
        app = ReactorGUI()
        app.mainloop()
    elif choix == "2":
        # On cherche le fichier par défaut généré par la GUI
        mesh_path = "output/meshes/reactor_core.msh"
        if os.path.exists(mesh_path):
            run_full_simulation(mesh_path)
        else:
            print(f"Erreur : Aucun maillage trouvé à l'emplacement {mesh_path}")
            print("Veuillez d'abord générer un maillage via l'option 1.")
    else:
        print("Fin du programme.")

if __name__ == "__main__":
    main()