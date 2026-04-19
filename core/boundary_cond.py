import numpy as np

def get_dirichlet_nodes(mesh, boundary_physical_tags):
    """
    Extrait les indices des nœuds situés sur les bords spécifiés.

    Parameters
    ----------
    mesh : meshio.Mesh
        L'objet maillage chargé via meshio.
    boundary_physical_tags : list of int
        Liste des tags (Physical Groups) définis dans Gmsh qui correspondent 
        aux frontières de Dirichlet (ex: [100, 101]).

    Returns
    -------
    dirichlet_dofs : ndarray
        Tableau 1D contenant les indices uniques des nœuds du bord.
    """
    boundary_nodes = []

    # Dans meshio, les groupes physiques sont souvent stockés dans cell_sets
    # ou cell_data selon la version du format .msh.
    # Ici, on itère sur les dictionnaires de cell_sets créés par Gmsh.
    
    for tag in boundary_physical_tags:
        tag_name = f"gmsh:physical:{tag}" # Nom standard de meshio pour les tags Gmsh
        
        if tag_name in mesh.cell_sets:
            # On récupère les indices des éléments (lignes en 2D, triangles en 3D) sur ce bord
            # mesh.cell_sets[tag_name] renvoie une liste de blocs. 
            for block_type, elem_indices in mesh.cell_sets[tag_name].items():
                
                # On trouve les nœuds qui composent ces éléments de bord
                # mesh.cells_dict contient la connectivité par type d'élément (ex: "line")
                if block_type in mesh.cells_dict:
                    nodes_of_elements = mesh.cells_dict[block_type][elem_indices]
                    
                    # On aplatit la liste et on l'ajoute à notre liste globale
                    boundary_nodes.extend(nodes_of_elements.flatten())

    # On supprime les doublons (car un nœud peut appartenir à deux lignes adjacentes)
    dirichlet_dofs = np.unique(boundary_nodes)
    
    return dirichlet_dofs