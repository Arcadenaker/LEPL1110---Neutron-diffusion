import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)

# Base de données des matériaux nucléaires pour le modèle de diffusion à un groupe d'énergie.
# Les valeurs macroscopiques sont moyennées pour un spectre thermique standard à 20°C.
MATERIAL_DB = {
    "Fuel_Uranium": {
        # Valeurs effectives pour un assemblage UO2 faiblement enrichi homogénéisé.
        # Le coefficient D est élevé (spectre durci) et l'absorption prend en compte
        # l'auto-protection spatiale de la pastille.
        "D": 1.5,  # cm
        "Sigma_a": 0.05,  # cm^-1
        "nuSigma_f": 0.06,  # cm^-1 (Génère un k_inf = 1.20)
        "v": 550.0,  # cm/s (vitesse neutrons thermiques)
    },
    "Water_Moderator": {
        # Eau légère (H2O). La très forte section de diffusion de l'hydrogène
        # donne un coefficient D très faible.[2]
        "D": 0.16,  # cm
        "Sigma_a": 0.0197,  # cm^-1 (Valeur tabulée exacte, souvent arrondie à 0.01 dans les modèles simples) [2]
        "nuSigma_f": 0.0,
        "v": 220000.0,
    },
    "HeavyWater_Moderator": {
        # Eau lourde (D2O). Excellent modérateur avec une transparence neutronique exceptionnelle.
        "D": 0.87,  # cm
        "Sigma_a": 0.000093,  # cm^-1
        "nuSigma_f": 0.0,
        "v": 550.0,
    },
    "Graphite_Moderator": {
        # Graphite de qualité nucléaire (densité 1.60 g/cm^3).
        "D": 0.84,  # cm
        "Sigma_a": 0.00024,  # cm^-1
        "nuSigma_f": 0.0,
        "v": 550.0,
    },
    "Beryllium_Moderator": {
        # Béryllium métallique (densité 1.85 g/cm^3). Très bon pouvoir ralentisseur sous un faible volume.
        "D": 0.50,  # cm
        "Sigma_a": 0.00104,  # cm^-1
        "nuSigma_f": 0.0,
        "v": 550.0,
    },
    "Boral_ControlRod": {
        # Composite d'aluminium et de particules de carbure de bore (B4C).
        # La valeur Sigma_a de 0.5 cm^-1 modélise efficacement l'effet "puits noir"
        # tout en évitant les instabilités numériques dans un maillage grossier.
        "D": 0.20,  # cm (Dominé par la matrice d'aluminium)
        "Sigma_a": 0.5,  # cm^-1
        "nuSigma_f": 0.0,
        "v": 550.0,
    },
}


# Ajoute l'argument user_mapping=None à la définition de la fonction
def get_material_properties(mesh, elem_tags, rod_insertion=1.0, user_mapping=None):
    """
    Crée les vecteurs de propriétés pour chaque élément à partir des noms de matériaux.
    """
    logger.debug(f"Calcul des propriétés matériaux. rod_insertion={rod_insertion}")
    ne = len(elem_tags)
    D_vec, Sigma_a_vec, nuSigma_f_vec, inv_v_vec = (
        np.zeros(ne),
        np.zeros(ne),
        np.zeros(ne),
        np.zeros(ne),
    )

    # 1. Calcul des offsets (indispensable pour meshio)
    triangle_offsets = {}
    current_offset = 0
    for block_id, cell_block in enumerate(mesh.cells):
        if cell_block.type == "triangle":
            triangle_offsets[block_id] = current_offset
            current_offset += len(cell_block.data)

    # --- NOUVEAU CODE : Gestion du dictionnaire dynamique ---
    # Si aucun mapping n'est fourni, on met les valeurs par défaut
    if user_mapping is None:
        logger.warning(
            "Aucun user_mapping fourni, utilisation des matériaux par défaut."
        )
        user_mapping = {
            "Fuel": "Fuel_Uranium",
            "Moderator": "Water_Moderator",
            "Reflector": "Graphite_Moderator",
            "ControlRods": "Boral_ControlRod",
        }
    else:
        logger.debug(f"Utilisation du user_mapping: {user_mapping}")

    # 2. Matériaux standards (Fixes)
    # On boucle sur les clés fixes du maillage, et on cherche leur équivalent dans user_mapping
    for mesh_name in ["Fuel", "Moderator", "Reflector"]:
        if mesh_name in mesh.cell_sets:
            try:
                db_name = user_mapping[mesh_name]
                props = MATERIAL_DB[db_name]
            except KeyError as e:
                logger.error(
                    f"Erreur: Matériau introuvable dans MATERIAL_DB ou user_mapping manquant : {e}",
                    exc_info=True,
                )
                raise

            for block_id, elem_indices in enumerate(mesh.cell_sets[mesh_name]):
                if len(elem_indices) > 0 and mesh.cells[block_id].type == "triangle":
                    g_idx = elem_indices + triangle_offsets[block_id]
                    D_vec[g_idx] = props["D"]
                    Sigma_a_vec[g_idx] = props["Sigma_a"]
                    nuSigma_f_vec[g_idx] = props["nuSigma_f"]
                    inv_v_vec[g_idx] = 1.0 / props["v"]

    # 3. Logique dynamique pour les Barres de Contrôle
    if "ControlRods" in mesh.cell_sets:
        # On utilise le mapping pour trouver la barre, ET pour trouver par quoi elle est remplacée (le modérateur)
        cr_name = user_mapping.get("ControlRods", "Boral_ControlRod")
        mod_name = user_mapping.get("Moderator", "Water_Moderator")

        try:
            cr_full = MATERIAL_DB[cr_name]
            cr_empty = MATERIAL_DB[mod_name]
        except KeyError as e:
            logger.error(
                f"Erreur: Matériau CR ou Modérateur introuvable dans MATERIAL_DB : {e}",
                exc_info=True,
            )
            raise

        # Interpolation linéaire
        eff_Sigma_a = (
            rod_insertion * cr_full["Sigma_a"]
            + (1.0 - rod_insertion) * cr_empty["Sigma_a"]
        )
        eff_D = rod_insertion * cr_full["D"] + (1.0 - rod_insertion) * cr_empty["D"]

        for block_id, elem_indices in enumerate(mesh.cell_sets["ControlRods"]):
            if len(elem_indices) > 0 and mesh.cells[block_id].type == "triangle":
                g_idx = elem_indices + triangle_offsets[block_id]
                D_vec[g_idx] = eff_D
                Sigma_a_vec[g_idx] = eff_Sigma_a
                nuSigma_f_vec[g_idx] = 0.0
                inv_v_vec[g_idx] = 1.0 / cr_full["v"]

    logger.debug(
        "Propriétés des matériaux appliquées avec succès sur tout le maillage."
    )
    return D_vec, Sigma_a_vec, nuSigma_f_vec, inv_v_vec
