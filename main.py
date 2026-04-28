import sys
import os

# S'assure que Python trouve les dossiers 'core', 'physics' et 'utils'
# même si le script est lancé depuis un autre répertoire.
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils.gui import ReactorGUI


def main():
    print("=== LEPL1110 : SIMULATEUR NEUTRONIQUE ===")
    print("Démarrage de l'environnement graphique...")

    # Instanciation et lancement de l'interface principale
    app = ReactorGUI()
    app.mainloop()

    print("Fermeture du simulateur.")


if __name__ == "__main__":
    main()
