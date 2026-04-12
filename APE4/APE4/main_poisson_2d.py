import argparse
import numpy as np
import matplotlib.pyplot as plt

from gmsh_utils import (
    border_dofs_from_tags, build_2d_rectangle_mesh, gmsh_init, gmsh_finalize,
    prepare_quadrature_and_basis, get_jacobians,
    mesh1, mesh2
)
from stiffness_node import assemble_stiffness_and_rhs_node_wise
from dirichlet import solve_dirichlet
from errors import compute_L2_H1_errors

from plot_utils import plot_mesh_2d, plot_fe_solution_2d
import gmsh


def main(L, H, h, order):

    Mx = 2.0
    My = 2.0
    def u_exact(x): return np.sin(Mx*np.pi * x[0]) * np.sin(My*np.pi * x[1])
    def kappa(x): return 1.0
    def f(x): return (Mx*Mx*np.pi*np.pi + My*My*np.pi*np.pi)*np.sin(Mx*np.pi*x[0]) * np.sin(My*np.pi*x[1])
    def grad_exact(x): return np.array([Mx*np.pi*np.cos(Mx*np.pi*x[0]) * np.sin(My*np.pi*x[1]), My*np.pi*np.sin(Mx*np.pi*x[0]) * np.cos(My*np.pi*x[1]), 0.0])

    def size_field(x, y): return h
    
    gmsh_init("poisson_2d")
    
    elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags = build_2d_rectangle_mesh(L=L, H=H, size_field=size_field, order=order)
    #elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags = mesh1(L=1, H=1, order=order, smin=0.01, smax=0.1, dmin=0, dmax=0.3)
    #elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags = mesh2(L=2, H=1, order=order, smin=0.01, smax=0.015, dmin=0, dmax=0.3)

    plot_mesh_2d(elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags)

    # define a mapping between the nodeTags returned from gmsh and the dofs of the system
    # ------------------------------------------
    unique_dofs_tags = np.unique(elemNodeTags)
    num_dofs = len(unique_dofs_tags)
    max_tag = int(np.max(nodeTags))
    tag_to_dof = np.full(max_tag + 1, -1, dtype=int)
    for i, tag in enumerate(unique_dofs_tags):
        tag_to_dof[int(tag)] = i
    # ------------------------------------------


    xi, w, N, gN = prepare_quadrature_and_basis(elemType, order)
    jac, det, coords = get_jacobians(elemType, xi)

    K_lil, F = assemble_stiffness_and_rhs_node_wise(elemTags, elemNodeTags, jac, det, coords, w, N, gN, kappa, f, tag_to_dof)
    
    K = K_lil.tocsr()

    border_tags = np.concatenate(bnds_tags)

    # to apply dirichlet condition with the analytical expr, we need to evaluate nodes at right coords -> use mapping
    num_dofs = len(F) 
    dof_coords = np.zeros((num_dofs, 3))
    all_coords = nodeCoords.reshape(-1, 3)
    for i, tag in enumerate(nodeTags):
        idx = tag_to_dof[int(tag)]
        if idx != -1:
            dof_coords[idx] = all_coords[i]

    # get the indices of the boundary nodes mapped to the system we solve
    border_dofs = border_dofs_from_tags(border_tags, tag_to_dof)
    # set the dirichlet values : here we evaluate the analytical function for the example
    dir_vals = [u_exact(dof_coords[d]) for d in border_dofs]

    # Solve 
    U = solve_dirichlet(K, F, border_dofs, dir_vals)

    L2, H1_semi, H1 = compute_L2_H1_errors(elemType, elemTags, elemNodeTags, U, xi, w, N, gN, jac, det, coords, u_exact, grad_exact=grad_exact, tag_to_dof=tag_to_dof)
    print("Errors: L2 = {:.6e}, H1 semi = {:.6e}, H1 = {:.6e}".format(L2, H1_semi, H1))
    gmsh_finalize()

    plot_fe_solution_2d(elemNodeTags, nodeCoords, nodeTags, U, tag_to_dof, show_mesh=False, ax=None)
    plt.show()
    
    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Poisson 2D with Gmsh FE")
    parser.add_argument("-order", type=int, default=1)
    parser.add_argument("-L", type=float, default=1.0)
    parser.add_argument("-H", type=float, default=1.0)
    parser.add_argument("-hc", type=float, default=0.1)
    args = parser.parse_args()
    order = args.order
    L = args.L
    H = args.H
    h = args.hc
    
    main(L, H, h, order)
