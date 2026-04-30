import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)

# -- LA CLASSE MATÉRIAU --

# Cette valeur est une approximation qui estime la durée de vie moyenne (en secondes)
# d'une génération de neutrons, imposée par la lenteur des neutrons retardés.
L_EFF_GLOBAL = 0.08135 

class Material:
    """
    Classe qui permet de stocker proprement les matériaux et calculer la vitesse
    des neutrons.
    """
    def __init__(self, D, Sigma_a, nuSigma_f):
        self.D = D                  # Capacité des neutrons à bouger
        self.Sigma_a = Sigma_a      # Capacité à absorber les neutrons
        self.nuSigma_f = nuSigma_f  # Capacité à produire de nouveaux neutrons

    @property
    def v(self):
        """
        Fonction qui permet de calculer la vitesse des neutrons

        Nous considérons l_eff​ comme une constante de temps globale du réacteur car 
        la dynamique du flux est solidaire sur l'ensemble du cœur. Dans notre modèle FEM, 
        nous changeons donc la vitesse v dans chaque matériau afin que le produit v⋅Σa​ 
        soit identique partout, ce qui permet que l'inertie des neutrons retardés est 
        respectée de manière homogène dans toute la simulation.
        """
        # Si le matériau absorbe les neutrons (comme l'eau légère ou les barres)
        if self.Sigma_a > 0.005:
            # l_eff = 1/(v_eff * \sigma_a) <=> v_eff = 1/(l_eff * \sigma_a)
            return round(1.0 / (L_EFF_GLOBAL * self.Sigma_a), 2)
        
        # Si le matériau n'absorbe presque rien (Eau lourde, Graphite), 
        # la division par zéro ferait exploser la matrice. On plafonne la vitesse.
        return 1000


# -- LA BDD des matériaux

# On crée nos matériaux en utilisant Material
MATERIAL_DB = {
    # Uranium enrichi (Produit plus qu'il n'absorbe)
    "Fuel_Uranium":         Material(D=1.5,  Sigma_a=0.05,   nuSigma_f=0.16),
    
    # Eau légère (Absorbe pas mal de neutrons)
    "Water_Moderator":      Material(D=0.16, Sigma_a=0.0197, nuSigma_f=0.0),
    
    # Eau lourde / Graphite (Transparents aux neutrons, n'absorbent presque rien)
    "HeavyWater_Moderator": Material(D=0.87, Sigma_a=0.000093, nuSigma_f=0.0),
    "Graphite_Moderator":   Material(D=0.84, Sigma_a=0.00024,  nuSigma_f=0.0),
    
    # Barre de contrôle (Un vrai trou noir à neutrons)
    "Boral_ControlRod":     Material(D=0.20, Sigma_a=0.5,    nuSigma_f=0.0)
}


# -- FONCTION QU'ON UTILISE POUR LE MAILLAGE --

def get_material_properties(mesh, elem_tags, rod_insertion=1.0, user_mapping=None):
    """
    Cette fonction se promène dans tous les triangles du maillage.
    Elle regarde la couleur du triangle (Fuel, Modérateur...) et lui colle 
    les bonnes propriétés physiques issues de notre base de données.
    """
    logger.debug(f"Calcul des propriétés matériaux. rod_insertion={rod_insertion}")
    ne = len(elem_tags)
    
    # On prépare des grands tableaux vides contenant un zéro pour chaque triangle
    D_vec = np.zeros(ne)
    Sigma_a_vec = np.zeros(ne)
    nuSigma_f_vec = np.zeros(ne)
    inv_v_vec = np.zeros(ne)

    # Petite étape technique pour Meshio : gérer le décalage des index des triangles
    triangle_offsets = {}
    current_offset = 0
    for block_id, cell_block in enumerate(mesh.cells):
        if cell_block.type == "triangle":
            triangle_offsets[block_id] = current_offset
            current_offset += len(cell_block.data)

    # Si l'interface graphique (GUI) n'a rien envoyé, on met la configuration par défaut
    if user_mapping is None:
        logger.warning("Aucun user_mapping fourni, utilisation des matériaux par défaut.")
        user_mapping = {
            "Fuel": "Fuel_Uranium",
            "Moderator": "Water_Moderator",
            "Reflector": "Graphite_Moderator",
            "ControlRods": "Boral_ControlRod",
        }

    # --- ÉTAPE A : Les zones fixes (Combustible, Modérateur, Réflecteur) ---
    for mesh_name in ["Fuel", "Moderator", "Reflector"]:
        if mesh_name in mesh.cell_sets:
            # On va chercher l'objet Material correspondant dans la base de données
            db_name = user_mapping[mesh_name]
            props = MATERIAL_DB[db_name]

            # On parcourt tous les triangles appartenant à cette zone
            for block_id, elem_indices in enumerate(mesh.cell_sets[mesh_name]):
                if len(elem_indices) > 0 and mesh.cells[block_id].type == "triangle":
                    g_idx = elem_indices + triangle_offsets[block_id]
                    
                    # On remplit nos tableaux.
                    # Attention la syntaxe a changé : props.D au lieu de props["D"]
                    D_vec[g_idx] = props.D
                    Sigma_a_vec[g_idx] = props.Sigma_a
                    nuSigma_f_vec[g_idx] = props.nuSigma_f
                    
                    # C'est ici que l'objet calcule secrètement sa vitesse avec props.v !
                    inv_v_vec[g_idx] = 1.0 / props.v

    # --- ÉTAPE B : La zone dynamique (Les Barres de Contrôle) ---
    if "ControlRods" in mesh.cell_sets:
        
        # On récupère les objets de la barre de contrôle ET de l'eau
        cr_name = user_mapping.get("ControlRods", "Boral_ControlRod")
        mod_name = user_mapping.get("Moderator", "Water_Moderator")
        
        cr_full = MATERIAL_DB[cr_name]   # Quand la barre est 100% insérée
        cr_empty = MATERIAL_DB[mod_name] # Quand la barre est 100% levée (c'est de l'eau)

        # La magie de l'interpolation : si la barre est à moitié levée (0.5),
        # le triangle devient un mélange parfait entre du Boral et de l'eau.
        eff_Sigma_a = (rod_insertion * cr_full.Sigma_a) + ((1.0 - rod_insertion) * cr_empty.Sigma_a)
        eff_D = (rod_insertion * cr_full.D) + ((1.0 - rod_insertion) * cr_empty.D)

        for block_id, elem_indices in enumerate(mesh.cell_sets["ControlRods"]):
            if len(elem_indices) > 0 and mesh.cells[block_id].type == "triangle":
                g_idx = elem_indices + triangle_offsets[block_id]
                
                D_vec[g_idx] = eff_D
                Sigma_a_vec[g_idx] = eff_Sigma_a
                nuSigma_f_vec[g_idx] = 0.0 # Une barre ne produit jamais de neutrons
                inv_v_vec[g_idx] = 1.0 / cr_full.v

    logger.debug("Propriétés des matériaux appliquées avec succès sur tout le maillage.")
    return D_vec, Sigma_a_vec, nuSigma_f_vec, inv_v_vec