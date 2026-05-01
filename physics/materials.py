import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)

# Cette valeur est le "temps de génération effectif" (l_eff).
# Dans la réalité, un réacteur nucléaire sans "neutrons retardés" (ceux émis 
# quelques secondes après la fission) exploserait en une milliseconde. 
# Ce l_eff artificiellement élevé (0.08s) "simule" mathématiquement l'effet stabilisateur 
# des neutrons retardés pour nous permettre de piloter la simulation avec un controleur PD.
L_EFF_GLOBAL = 0.08135 

class Material:
    """
    Classe de stockage des propriétés neutroniques à un groupe d'énergie.
    """
    def __init__(self, D, Sigma_a, nuSigma_f):
        self.D = D                  # Coefficient de Diffusion (cm) : Capacité à s'échapper.
        self.Sigma_a = Sigma_a      # Section Efficace Macroscopique d'Absorption (cm^-1).
        self.nuSigma_f = nuSigma_f  # Section Efficace de Production (Fission) (cm^-1).

    @property
    def v(self):
        """
        Calcul dynamique de la vitesse (v) des neutrons dans ce milieu

        Dans l'équation de diffusion dépendante du temps, le terme d'inertie est (1/v) * (d\Phi/dt).
        Or, pour que notre l_eff global s'applique à tout le réacteur, la théorie impose que :
        l_eff = 1 / (v * Sigma_a).
        
        Nous devons donc faire une approximation : au lieu de donner une vitesse constante 
        aux neutrons (ce qui est physiquement vrai), on ajuste la vitesse (v) de chaque 
        matériau pour que le produit (v * Sigma_a) donne toujours notre L_EFF_GLOBAL.
        Cela garantit que l'inertie de la matrice de Masse (M) reste homogène dans tout le cœur.
        """
        # Si le matériau absorbe significativement les neutrons (Uranium, Eau légère, Boral)
        if self.Sigma_a > 0.005:
            # On isole v de la formule : l_eff = 1/(v * \sigma_a)
            return round(1.0 / (L_EFF_GLOBAL * self.Sigma_a), 2)
        
        # Si le milieu est presque "transparent" (Graphite, Eau lourde),
        # Sigma_a tend vers 0. La division ferait tendre v vers l'infini
        # Mathématiquement, cela rendrait notre matrice de Masse singulière et le solveur LU planterait
        # On plafonne donc la vitesse virtuelle à une valeur arbitrairement élevée
        return 1500


# -- LA BASE DE DONNÉES NUCLÉAIRE --

# Référence principale pour les valeurs : "Introduction to Nuclear Engineering", John R. Lamarsh
# Les valeurs pures sont données pour un spectre thermique de Maxwell-Boltzmann 
# à 20°C (énergie moyenne des neutrons E = 0.0253 eV, vitesse v = 2200 m/s)

MATERIAL_DB = {
    # --- LES MODÉRATEURS ET RÉFLECTEURS ---

    # Eau légère (H2O) : 
    # La valeur de Lamarsh à 2200 m/s est Sigma_a = 0.0222 cm^-1.
    # Corrigée par le facteur (sqrt(pi)/2) pour le spectre thermique réel à 20°C, on obtient ~0.0196.
    "Water_Moderator":        Material(D=0.16, Sigma_a=0.0196, nuSigma_f=0.0),
    
    # Eau légère borée (Pilotage chimique des REP) :
    # L'ajout d'acide borique dissout dans l'eau augmente l'absorption globale du fluide.
    "BoratedWater_Moderator": Material(D=0.16, Sigma_a=0.0260, nuSigma_f=0.0),
    
    # Eau lourde (D2O) :
    # Lamarsh donne 0.000033 pour du D2O pur à 100%. 
    # En réalité, la présence de 0.2% de H2O remonte l'absorption autour de 0.00009.
    "HeavyWater_Moderator":   Material(D=0.87, Sigma_a=0.000093, nuSigma_f=0.0),
    
    # Graphite nucléaire (C) :
    # Valeur standard pour un réacteur type RBMK ou UNGG (dépend de la pureté de fabrication).
    "Graphite_Moderator":     Material(D=0.84, Sigma_a=0.00032,  nuSigma_f=0.0),

    # Béryllium (Be) :
    # Réflecteur d'élite utilisé dans les réacteurs de recherche (ex: RJH au CEA).
    # Renvoie massivement les neutrons avec une absorption très faible.
    "Beryllium_Reflector":    Material(D=0.50, Sigma_a=0.0011,   nuSigma_f=0.0),


    # --- LES ZONES HOMOGÉNÉISÉES ---
    # Note pour la modélisation : Ces valeurs ne sont pas celles des isotopes purs de Lamarsh
    # Ce sont des "Cellules Cœur" moyennées. Un triangle "Fuel" dans notre maillage 
    # représente en fait un mix de 30% d'Uranium, 10% de gaine (Zirconium) et 60% d'eau.
    
    # Combustible standard UOX (Uranium enrichi)
    # C'est le seul milieu sur-critique naturel de notre base (nuSigma_f > Sigma_a).
    "Fuel_Uranium":           Material(D=1.50, Sigma_a=0.050,  nuSigma_f=0.160),
    
    # Combustible MOX (Mix Uranium / Plutonium 239)
    # Utilisé dans le parc français. Il absorbe plus de neutrons thermiques que l'UOX, 
    # mais produit plus de neutrons par fission (plus nerveux à piloter).
    "Fuel_MOX":               Material(D=1.20, Sigma_a=0.080,  nuSigma_f=0.200),
    

    # --- LES ACTIONNEURS ---
    
    # Barres de contrôle (Boral / Carbure de Bore)
    # Le Bore 10 pur a une absorption réelle colossale (Sigma_a > 100 cm^-1)
    "Boral_ControlRod":       Material(D=0.20, Sigma_a=0.500,  nuSigma_f=0.0)
}


# -- FONCTION DE MAPPING (MAILLAGE <-> PHYSIQUE) --

def get_material_properties(mesh, elem_tags, rod_insertion=1.0, user_mapping=None):
    """
    Cette fonction agit comme un traducteur : elle parcourt chaque triangle du maillage Gmsh 
    et lui assigne ses 4 coefficients mathématiques (D, Sigma_a, nuSigma_f, inv_v) 
    en fonction de son tag physique (Fuel, Moderator, etc.).
    """
    logger.debug(f"Calcul des propriétés matériaux. rod_insertion={rod_insertion}")
    ne = len(elem_tags)
    
    # Allocation contiguë en mémoire (np.zeros). C'est indispensable pour 
    # passer ces vecteurs proprement aux fonctions C++ de Numba plus tard.
    D_vec = np.zeros(ne)
    Sigma_a_vec = np.zeros(ne)
    nuSigma_f_vec = np.zeros(ne)
    inv_v_vec = np.zeros(ne)

    # Petite étape technique propre à Meshio : 
    # Gmsh découpe le fichier de maillage en "blocs" géométriques. 
    # On calcule les décalages (offsets) pour retrouver l'ID global d'un triangle à partir de son ID de bloc.
    triangle_offsets = {}
    current_offset = 0
    for block_id, cell_block in enumerate(mesh.cells):
        if cell_block.type == "triangle":
            triangle_offsets[block_id] = current_offset
            current_offset += len(cell_block.data)

    # Fallback : Si l'interface graphique (GUI) plante ou n'envoie rien, 
    # on force une configuration de réacteur standard (style REP).
    if user_mapping is None:
        logger.warning("Aucun user_mapping fourni, utilisation des matériaux par défaut.")
        user_mapping = {
            "Fuel": "Fuel_Uranium",
            "Moderator": "Water_Moderator",
            "Reflector": "Graphite_Moderator",
            "ControlRods": "Boral_ControlRod",
        }

    # --- ÉTAPE A : Remplissage des zones statiques ---
    # On boucle sur les composants qui ne bougent pas pendant la simulation.
    for mesh_name in ["Fuel", "Moderator", "Reflector"]:
        if mesh_name in mesh.cell_sets:
            # Traduction du nom GMSH vers l'objet Material de notre BDD
            db_name = user_mapping[mesh_name]
            props = MATERIAL_DB[db_name]

            for block_id, elem_indices in enumerate(mesh.cell_sets[mesh_name]):
                if len(elem_indices) > 0 and mesh.cells[block_id].type == "triangle":
                    g_idx = elem_indices + triangle_offsets[block_id]
                    
                    # On propage la physique aux triangles concernés
                    D_vec[g_idx] = props.D
                    Sigma_a_vec[g_idx] = props.Sigma_a
                    nuSigma_f_vec[g_idx] = props.nuSigma_f
                    
                    # L'inertie (la matrice de Masse) nécessite l'inverse de la vitesse
                    inv_v_vec[g_idx] = 1.0 / props.v

    # --- ÉTAPE B : Homogénéisation des Barres de Contrôle ---
    # Le maillage 2D (les triangles) est fixe. On ne peut pas "bouger" physiquement les barres 
    # à chaque étape de temps (il faudrait remailler)
    # L'astuce mathématique consiste à faire varier la densité macroscopique du triangle
    if "ControlRods" in mesh.cell_sets:
        
        # Une barre levée laisse place au modérateur liquide (ou vide)
        cr_name = user_mapping.get("ControlRods", "Boral_ControlRod")
        mod_name = user_mapping.get("Moderator", "Water_Moderator")
        
        cr_full = MATERIAL_DB[cr_name]   # État inséré (Absorption Max)
        cr_empty = MATERIAL_DB[mod_name] # État levé (Eau)

        # Interpolation linéaire basée sur le pourcentage d'insertion (0.0 à 1.0)
        # Ex : À rod_insertion = 0.5, le triangle agit mathématiquement comme 
        # un fluide composé à 50% d'Eau et à 50% de Boral.
        eff_Sigma_a = (rod_insertion * cr_full.Sigma_a) + ((1.0 - rod_insertion) * cr_empty.Sigma_a)
        eff_D = (rod_insertion * cr_full.D) + ((1.0 - rod_insertion) * cr_empty.D)

        for block_id, elem_indices in enumerate(mesh.cell_sets["ControlRods"]):
            if len(elem_indices) > 0 and mesh.cells[block_id].type == "triangle":
                g_idx = elem_indices + triangle_offsets[block_id]
                
                D_vec[g_idx] = eff_D
                Sigma_a_vec[g_idx] = eff_Sigma_a
                
                # Sécurité : Une barre ne produit jamais de neutrons, 
                # même si l'interface GUI l'associe au combustible par erreur.
                nuSigma_f_vec[g_idx] = 0.0 
                
                # Pour la vitesse, on force l'inertie du Boral plein pour éviter 
                # des instabilités numériques du solveur lors de l'extraction.
                inv_v_vec[g_idx] = 1.0 / cr_full.v

    logger.debug("Propriétés des matériaux appliquées avec succès sur tout le maillage.")
    return D_vec, Sigma_a_vec, nuSigma_f_vec, inv_v_vec