import numpy as np
from scipy.sparse import coo_matrix
from numba import njit, prange

# --- IMPORT DU LOGGER ---
from utils.logger import get_logger

logger = get_logger(__name__)
# ------------------------

# -- ASSEMBLAGE RAPIDE : MATRICE DE MASSE ET DE RÉACTION --
# Le décorateur @njit(parallel=True) vient de la librairie Numba
# Il va transformer ce code Python (normalement lent) en code rapide (C++)
# et l'exécuter sur TOUS les cœurs du processeur en même temps. 
# Sans ça, la simulation serait plus longue
@njit(parallel=True)
def _fast_assemble_mass_reaction_core(ne, nloc, ngp, conn, det, w, N, coeffs):
    # On pré-calcule la taille exacte des tableaux nécessaires.
    # Si on demandait à Python d'agrandir un tableau petit à petit, 
    # ce serait très lent. Ici, on réserve toute la mémoire d'un coup
    size = ne * nloc * nloc * ngp
    data = np.zeros(size)
    rows = np.zeros(size, dtype=np.int32)
    cols = np.zeros(size, dtype=np.int32)

    # La boucle prange est la version multi-cœurs de range
    # Chaque cœur du processeur va prendre un paquet de triangles (éléments) à traiter
    for e in prange(ne):
        c_e = coeffs[e]   # Propriété physique du triangle actuel (ex: absorption)
        nodes = conn[e]   # Les 3 sommets (nœuds) qui forment ce triangle

        # Pour calculer l'intégrale sur le triangle, on utilise les points de Gauss (ngp)
        # C'est une technique mathématique pour calculer des aires difficiles
        for g in range(ngp):
            # Le facteur regroupe le poids du point de Gauss (w), 
            # la déformation du triangle (det), et la physique locale (c_e).
            facteur = w[g] * det[e, g] * c_e

            # On croise chaque sommet du triangle avec les autres (matrice 3x3 locale)
            for a in range(nloc):
                for b in range(nloc):
                    # Calcul de l'index absolu dans notre grand tableau 1D
                    # C'est cette formule qui garantit que deux cœurs de processeur 
                    # ne doivent écrire au même endroit en même temps
                    idx = e * (ngp * nloc * nloc) + g * (nloc * nloc) + a * nloc + b

                    # On stocke la valeur calculée et ses coordonnées (ligne, colonne)
                    data[idx] = facteur * N[g, a] * N[g, b]
                    rows[idx] = nodes[a]
                    cols[idx] = nodes[b]

    return data, rows, cols

# -- ASSEMBLAGE RAPIDE : MATRICE DE RIGIDITÉ (DIFFUSION) --
# Cette fonction est beaucoup plus complexe mathématiquement car elle gère la 
# fuite des neutrons d'un triangle à l'autre (le gradient).
@njit(parallel=True)
def _fast_assemble_stiffness_core(
    ne, nloc, ngp, conn, det, w, jacobians, gradN_ref, coeffs_D
):
    size = ne * nloc * nloc * ngp
    data = np.zeros(size)
    rows = np.zeros(size, dtype=np.int32)
    cols = np.zeros(size, dtype=np.int32)

    for e in prange(ne):
        # SÉCURITÉ THREADS : On alloue ces petits tableaux "brouillons" 
        # à l'INTÉRIEUR de la boucle. Si on les mettait au début du fichier,
        # tous les cœurs essaieraient d'écrire dessus en même temps et ce serait le chaos
        grad_real_x = np.zeros(nloc)
        grad_real_y = np.zeros(nloc)

        D_e = coeffs_D[e] # Coefficient de diffusion du triangle
        nodes = conn[e]

        for g in range(ngp):
            # Récupération de la matrice Jacobienne (comment le triangle réel est déformé
            # par rapport à un triangle parfait de référence)
            J00 = jacobians[e, g, 0, 0]
            J01 = jacobians[e, g, 0, 1]
            J10 = jacobians[e, g, 1, 0]
            J11 = jacobians[e, g, 1, 1]

            # Déterminant de la matrice Jacobienne (l'aire de déformation)
            detJ = J00 * J11 - J01 * J10

            # Calcul à la main de l'inverse de la transposée du Jacobien.
            # On fait ça pour éviter d'appeler np.linalg.inv qui serait trop lent ici
            invJT_00 = J11 / detJ
            invJT_01 = -J10 / detJ
            invJT_10 = -J01 / detJ
            invJT_11 = J00 / detJ

            facteur = w[g] * det[e, g] * D_e

            # Étape cruciale : on transforme le gradient du triangle parfait
            # pour qu'il s'applique à notre vrai triangle tordu.
            for a in range(nloc):
                gr_x = gradN_ref[g, a, 0]
                gr_y = gradN_ref[g, a, 1]
                grad_real_x[a] = invJT_00 * gr_x + invJT_01 * gr_y
                grad_real_y[a] = invJT_10 * gr_x + invJT_11 * gr_y

            # On calcule l'interaction entre chaque sommet avec le dot product
            for a in range(nloc):
                for b in range(nloc):
                    dot_product = (
                        grad_real_x[a] * grad_real_x[b]
                        + grad_real_y[a] * grad_real_y[b]
                    )

                    # Index absolu sécurisé pour le multithreading
                    idx = e * (ngp * nloc * nloc) + g * (nloc * nloc) + a * nloc + b

                    data[idx] = facteur * dot_product
                    rows[idx] = nodes[a]
                    cols[idx] = nodes[b]

    return data, rows, cols

# -- FONCTIONS D'INTERFACE (LES CHEFS D'ORCHESTRE) --
# Ces fonctions servent de pont entre le reste du code Python et les fonctions 
# boostées par Numba en haut. Elles transforment les tableaux bruts
# en matrices creuses manipulables par SciPy
def assemble_mass_or_reaction(nn, ne, nloc, ngp, conn, det, w, N, coeffs):
    logger.debug(f"Début de l'assemblage (Masse/Réaction) pour {ne} éléments...")
    
    # 1. On lance le calcul sur tous les cœurs
    d, r, c = _fast_assemble_mass_reaction_core(ne, nloc, ngp, conn, det, w, N, coeffs)

    # 2. Construction de la matrice creuse
    # Dans un réacteur, un triangle ne touche que ses voisins. 
    # La matrice finale est donc pleine de zéros (99% de vide).
    # "coo_matrix" stocke uniquement les valeurs non-nulles pour économiser 
    # des Gigaoctets de RAM. "tocsr()" la formate pour que SciPy la résolve plus vite
    matrice_csr = coo_matrix((d, (r, c)), shape=(nn, nn)).tocsr()
    
    logger.debug(
        f"Assemblage terminé. Matrice creuse construite: taille {matrice_csr.shape} avec {matrice_csr.nnz} éléments non-nuls."
    )

    return matrice_csr


def assemble_stiffness(nn, ne, nloc, ngp, conn, det, w, jacobians, gradN_ref, coeffs_D):
    logger.debug(f"Début de l'assemblage de Rigidité (Diffusion) pour {ne} éléments...")
    
    # Même principe : calcul intensif envoyé à Numba
    d, r, c = _fast_assemble_stiffness_core(
        ne, nloc, ngp, conn, det, w, jacobians, gradN_ref, coeffs_D
    )

    # Conversion en matrice creuse optimisée
    matrice_csr = coo_matrix((d, (r, c)), shape=(nn, nn)).tocsr()
    
    logger.debug(
        f"Assemblage de Rigidité terminé. Matrice creuse construite: taille {matrice_csr.shape} avec {matrice_csr.nnz} éléments non-nuls."
    )

    return matrice_csr