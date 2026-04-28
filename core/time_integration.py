import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import splu

class TimeIntegrator:
    def __init__(self, M, K, R, dirichlet_dofs, theta=0.5):
        """
        Initialise le schéma Theta (θ).
        
        On essaie d'estimer la pente d'une courbe pour deviner le futur.
        - Explicite (θ=0) : On trace la tangente au présent. Dangereux si on avance trop loin (dt grand).
        - Implicite (θ=1) : On trace la tangente depuis le futur. Très stable, mais amortit les détails.
        - Crank-Nicolson (θ=0.5) : On prend la moyenne des deux.
        """
        self.M = M.tocsr() 
        self.K = K.tocsr() 
        self.R = R.tocsr() 
        self.dirichlet_dofs = np.asarray(dirichlet_dofs, dtype=int)
        self.theta = theta
        
    def integrate(self, phi_0, t_span, n_steps, dirichlet_values=None):
        t_start, t_end = t_span
        dt = (t_end - t_start) / n_steps
        times = np.linspace(t_start, t_end, n_steps + 1)
        n = self.M.shape[0]
        
        # L'équation neutronique est M * d(phi)/dt = R*phi - K*phi
        # K est la fuite (négatif sur le bilan), R est la source/absorption.
        # L'équation physique dit que la variation de neutrons = (Sources -> R - Pertes -> K)
        # Donc l'opérateur linéaire global est L = R - K
        L = self.R - self.K
        
        # Formulation implicite/explicite combinée :
        # (M - θ*dt*L) * phi_futur = (M + (1-θ)*dt*L) * phi_present
        # A * phi_futur = B * phi_present
        A = (self.M - self.theta * dt * L).tocsc() 
        B = (self.M + (1.0 - self.theta) * dt * L).tocsr()
        
        # Génère un tableau de booléens de longueur n 
        # (le nombre total de nœuds du maillage) initialisé partout à True
        # On part du principe qu'au début, tous les nœuds sont "inconnus" et doivent être calculés par le solveur.
        mask = np.ones(n, dtype=bool)

        # Désactivation des nœuds où la valeur est déjà connue (bords)
        # Les nœuds de Dirichlet ne sont pas des inconnues,
        # leur valeur est imposée (souvent à zéro pour simuler un bord de réacteur où les neutrons s'échappent). 
        # Les calculer via le système linéaire serait redondant et rendrait la matrice singulière (non inversible).
        mask[self.dirichlet_dofs] = False

        # Extraction des indices restants
        # La fonction np.nonzero() renvoie les indices où le masque est encore à True
        # On obtient ainsi la liste des Degrés de Liberté (DDL) libres,
        # ce sont les seuls nœuds que le processeur va réellement "résoudre".
        free_dofs = np.nonzero(mask)[0]
        
        # On sélectionne les lignes des nœuds libres, puis les colonnes des nœuds libres.
        # Cela crée une sous-matrice "Carrée" qui contient uniquement les interactions 
        # entre les points dont on ne connaît pas encore la valeur.
        A_FF = A[free_dofs, :][:, free_dofs]      

        # On sélectionne les lignes des nœuds libres, mais cette fois les colonnes de Dirichlet.
        # Cela crée une matrice "Rectangle" qui stocke comment chaque point du bord 
        # influence ses voisins directs à l'intérieur du réacteur.
        A_FD = A[free_dofs, :][:, self.dirichlet_dofs] 
        
        # On utilise l'algorithme "SuperLU" pour décomposer la matrice A_FF en deux matrices 
        # triangulaires (L et U). C'est une étape de préparation lourde qui rend 
        # toutes les solutions futures presque instantanées.
        solve_lu = splu(A_FF)
        
        # On crée une liste nommée 'solutions' et on y met une copie indépendante du flux initial (phi_0).
        # Pourquoi : On veut garder une trace de l'état de départ pour pouvoir tracer l'évolution complète à la fin, 
        # sans que les calculs futurs ne viennent modifier cette valeur initiale stockée.
        solutions = [phi_0.copy()]

        # On initialise une variable de travail 'phi_n' qui contient elle aussi une copie du flux initial.
        # Pourquoi : Cette variable sert de "mémoire tampon" pour le calcul. Elle représente le flux au temps "présent". 
        # À chaque tour de boucle, on l'utilisera pour calculer le futur, puis on l'écrasera avec ce futur pour avancer.
        phi_n = phi_0.copy()
        
        # On vérifie si l'utilisateur a omis de fournir des valeurs spécifiques pour les bords (None).
        # Pourquoi : On veut que le code soit robuste ; si aucune valeur n'est donnée, on part du principe 
        # que le flux est nul aux frontières (cas classique du réacteur entouré de vide).
        if dirichlet_values is None:
            # On crée un tableau de zéros dont la taille correspond exactement au nombre de nœuds de bord.
            # Pourquoi : Pour imposer mathématiquement une condition de "bord noir" (flux = 0) sur chaque nœud de Dirichlet.
            dir_vals = np.zeros(len(self.dirichlet_dofs))
        
        # Si des valeurs ont été fournies, on entre dans le cas où le bord a une intensité spécifique.
        else:
            # On convertit les valeurs fournies en un tableau NumPy pour assurer la compatibilité avec les opérations matricielles.
            # Pourquoi : Cela permet d'accepter une simple liste Python en entrée tout en garantissant la rapidité des calculs NumPy par la suite.
            dir_vals = np.asarray(dirichlet_values)

        # On calcule le produit de la matrice d'interaction bord/intérieur (A_FD) par les valeurs imposées (dir_vals).
        # Pourquoi : Dans l'équation A*phi = b, les termes connus (bords) sont déplacés à droite du signe égal.
        # Ce calcul transforme les valeurs de bord en une "force" ou une "source" qui modifie le bilan des nœuds voisins.
        boundary_force = A_FD.dot(dir_vals)

        # On lance une boucle qui va itérer à chaque pas de temps
        for i in range(1, n_steps + 1):
            
            # On calcule le produit de la matrice 'B' par le flux actuel 'phi_n'.
            # On calcule l'état des neutrons juste avant de passer à l'étape suivante pour savoir d'où l'on part.
            b_full = B.dot(phi_n)
            
            # On extrait les valeurs pour les nœuds libres et on soustrait la 'boundary_force' pré-calculée.
            # On "nettoie" le second membre pour ne garder que les inconnues réelles, tout en 
            # intégrant l'effet physique des bords (Dirichlet) sur ces nœuds.
            rhs_reduced = b_full[free_dofs] - boundary_force
            
            # On résout le système linéaire réduit en utilisant la factorisation LU
            # C'est le calcul de l'état futur. Grâce à LU, au lieu de refaire toute une division matricielle coûteuse
            phi_free_np1 = solve_lu.solve(rhs_reduced)
            
            # On crée un nouveau tableau de zéros de la taille totale du maillage (n).
            # Pourquoi : On a besoin d'un tableau vide pour reconstruire la solution complète sur tout le réacteur
            phi_np1 = np.zeros(n)
            
            # On injecte les valeurs calculées aux emplacements des nœuds libres dans le tableau complet.
            # Pourquoi : Pour replacer chaque résultat de flux à sa position géographique exacte dans le maillage.
            phi_np1[free_dofs] = phi_free_np1
            
            # On ré-injecte les valeurs de bord fixées (dir_vals) aux emplacements de Dirichlet.
            # pour que la solution finale respecte parfaitement les conditions imposées
            phi_np1[self.dirichlet_dofs] = dir_vals
            
            # On ajoute une copie de ce flux complet à la liste des solutions.
            # Pourquoi : Pour archiver cet instant T de la simulation et pouvoir analyser l'évolution du film 
            # neutronique une fois la boucle terminée.
            solutions.append(phi_np1.copy())
            
            # On écrase l'ancien flux 'phi_n' par le nouveau 'phi_np1'.
            # C'est le passage au pas de temps suivant
            phi_n = phi_np1
            
        return times, solutions