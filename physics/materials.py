import numpy as np

# Base de données des matériaux nucléaires pour le modèle de diffusion à un groupe d'énergie.
# Les valeurs macroscopiques sont moyennées pour un spectre thermique standard à 20°C.
MATERIAL_DB = {
    "Fuel_Uranium": {
        # Valeurs effectives pour un assemblage UO2 faiblement enrichi homogénéisé.
        # Le coefficient D est élevé (spectre durci) et l'absorption prend en compte 
        # l'auto-protection spatiale de la pastille.
        "D": 1.5,           # cm
        "Sigma_a": 0.05,    # cm^-1 
        "nuSigma_f": 0.06,  # cm^-1 (Génère un k_inf = 1.20)
        "v": 220000.0       # cm/s (vitesse neutrons thermiques)
    },
    "Water_Moderator": {
        # Eau légère (H2O). La très forte section de diffusion de l'hydrogène 
        # donne un coefficient D très faible.[2]
        "D": 0.16,          # cm 
        "Sigma_a": 0.0197,  # cm^-1 (Valeur tabulée exacte, souvent arrondie à 0.01 dans les modèles simples) [2]
        "nuSigma_f": 0.0,   
        "v": 220000.0
    },
    "HeavyWater_Moderator": {
        # Eau lourde (D2O). Excellent modérateur avec une transparence neutronique exceptionnelle.[2]
        "D": 0.87,          # cm 
        "Sigma_a": 0.000093,# cm^-1 [2]
        "nuSigma_f": 0.0,
        "v": 220000.0
    },
    "Graphite_Moderator": {
        # Graphite de qualité nucléaire (densité 1.60 g/cm^3).[2]
        "D": 0.84,          # cm
        "Sigma_a": 0.00024, # cm^-1 [2]
        "nuSigma_f": 0.0,
        "v": 220000.0
    },
    "Beryllium_Moderator": {
        # Béryllium métallique (densité 1.85 g/cm^3). Très bon pouvoir ralentisseur sous un faible volume.[2]
        "D": 0.50,          # cm
        "Sigma_a": 0.00104, # cm^-1 [2]
        "nuSigma_f": 0.0,
        "v": 220000.0
    },
    "Boral_ControlRod": {
        # Composite d'aluminium et de particules de carbure de bore (B4C).[3]
        # La valeur Sigma_a de 0.5 cm^-1 modélise efficacement l'effet "puits noir" 
        # tout en évitant les instabilités numériques dans un maillage grossier.
        "D": 0.20,          # cm (Dominé par la matrice d'aluminium)
        "Sigma_a": 0.5,     # cm^-1 
        "nuSigma_f": 0.0,
        "v": 220000.0
    }
}

def get_material_properties(mesh, elem_tags):
    """
    Crée les vecteurs de propriétés pour chaque élément à partir des noms de matériaux
    définis dans le maillage (mesh) et fait le lien avec la MATERIAL_DB.
    """
    ne = len(elem_tags)
    D_vec = np.zeros(ne)
    Sigma_a_vec = np.zeros(ne)
    nuSigma_f_vec = np.zeros(ne)
    inv_v_vec = np.zeros(ne)

    # On itère sur notre base de données physique pour peupler les vecteurs mathématiques
    for mat_name, props in MATERIAL_DB.items():
        if mat_name in mesh.cell_sets:
            # On récupère les indices des éléments appartenant à ce groupe physique
            indices = mesh.cell_sets[mat_name]["triangle"]
            
            D_vec[indices] = props["D"]
            Sigma_a_vec[indices] = props["Sigma_a"]
            nuSigma_f_vec[indices] = props["nuSigma_f"]
            
            # On stocke l'inverse de la vitesse
            inv_v_vec[indices] = 1.0 / props["v"]
            
    return D_vec, Sigma_a_vec, nuSigma_f_vec, inv_v_vec