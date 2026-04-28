import numpy as np

def get_dirichlet_nodes(mesh, boundary_physical_tags):
    """
    Dans un réacteur, on a besoin de savoir où les neutrons
    s'échappent (-> sur les bords). GMSH le prend en compte avec des tags physiques.
    """
    boundary_nodes = []
    
    for tag in boundary_physical_tags:
        tag_name = f"gmsh:physical:{tag}" 
        
        if tag_name in mesh.cell_sets:
            for block_id, elem_indices in enumerate(mesh.cell_sets[tag_name]):
                if len(elem_indices) > 0:
                    cell_block = mesh.cells[block_id]
                    # On s'assure qu'on ne prend que les lignes 1D (bords en 2D) ou sommets
                    if cell_block.type in ['line', 'vertex']:
                        nodes = cell_block.data[elem_indices]
                        boundary_nodes.extend(nodes.flatten())

    return np.unique(boundary_nodes)