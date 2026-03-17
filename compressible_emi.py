import logging
import time
import shutil
import numpy as np
from mpi4py import MPI
import dolfinx
import dolfinx.fem.petsc
import ufl


def subplus(x):
    return ufl.conditional(ufl.ge(x, 0.0), x, 0.0)


def heaviside(x):
    return ufl.conditional(ufl.ge(x, 0.0), 1.0, 0.0)


def compressibility(C, kappa=1e3):
    J = pow(ufl.det(C), 1 / 2)
    return kappa * (J * ufl.ln(J) - J + 1)


def psi_extracellular(C, a_e=1.52, b_e=16.31):
    """Isotropic strain energy for the extracellular matrix."""
    I1 = ufl.tr(C)
    return (a_e / (2.0 * b_e)) * (ufl.exp(b_e * (I1 - 3)) - 1.0)


def psi_intracellular(C, f0, a_i=5.70, b_i=11.67, a_if=19.83, b_if=24.72):
    """Transversely isotropic Holzapfel-Ogden strain energy for the cell."""
    I1 = ufl.tr(C)
    I4f = ufl.inner(C * f0, f0)
    return (a_i / (2.0 * b_i)) * (ufl.exp(b_i * (I1 - 3)) - 1.0) + (
        a_if / (2.0 * b_if)
    ) * heaviside(I4f - 1) * (ufl.exp(b_if * subplus(I4f - 1) ** 2) - 1.0)


def main():
    plot_matrix = True
    save_matrix = True
    save_solution = True

    logging.basicConfig(level=logging.INFO)
    dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
    comm = MPI.COMM_WORLD

    # Create a Box Mesh (1.0 x 0.2 x 0.2)
    mesh = dolfinx.mesh.create_box(
        comm, [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.2, 0.2])], [5, 3, 3]
    )

    # Define the Intracellular Domain (Inner Box)
    def is_intracellular(x):
        return (
            (x[0] >= 0.1)
            & (x[0] <= 0.9)
            & (x[1] >= 0.05)
            & (x[1] <= 0.15)
            & (x[2] >= 0.05)
            & (x[2] <= 0.15)
        )

    # Tag the subdomains (0: Extracellular, 1: Intracellular)
    tdim = mesh.topology.dim
    map_c = mesh.topology.index_map(tdim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cell_indices = np.arange(num_cells, dtype=np.int32)
    cell_markers = np.zeros(num_cells, dtype=np.int32)

    intracellular_cells = dolfinx.mesh.locate_entities(mesh, tdim, is_intracellular)
    cell_markers[intracellular_cells] = 1

    cell_tags = dolfinx.mesh.meshtags(mesh, tdim, cell_indices, cell_markers)

    with dolfinx.io.XDMFFile(mesh.comm, "cell_tags_emi_comp.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(cell_tags, mesh.geometry)

    # Create custom integration measure using the tags
    dx_custom = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)

    # Function space for the displacement
    P2 = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (3,)))

    # The displacement
    u = dolfinx.fem.Function(P2, name="u")
    u_test = ufl.TestFunction(P2)

    # Compute the deformation gradient
    F = ufl.grad(u) + ufl.Identity(3)
    C = ufl.variable(F.T * F)
    J = ufl.det(F)
    Cdev = J ** (-2.0 / 3.0) * C

    # Fiber direction (longitudinal)
    f0 = dolfinx.fem.Constant(mesh, [1.0, 0.0, 0.0])

    # Strain energies (deviatoric + volumetric parts)
    psi_e_val = psi_extracellular(Cdev) + compressibility(C)
    psi_i_val = psi_intracellular(Cdev, f0=f0) + compressibility(C)

    # Passive Stress Components
    S_e = 2.0 * ufl.diff(psi_e_val, C)
    S_i_p = 2.0 * ufl.diff(psi_i_val, C)

    # Active tension (Only applies to intracellular domain)
    Ta = dolfinx.fem.Constant(mesh, 0.0)
    S_a = Ta * ufl.outer(f0, f0)
    S_i = S_i_p + S_a

    # Weak form: Integrate extracellular energy over dx(0) and intracellular over dx(1)
    var_C = ufl.derivative(C, u, u_test)
    Ru = ufl.inner(S_e, 0.5 * var_C) * dx_custom(0) + ufl.inner(
        S_i, 0.5 * var_C
    ) * dx_custom(1)

    # Boundary setup
    def left(x):
        return np.isclose(x[0], 0.0)

    def right(x):
        return np.isclose(x[0], 1.0)

    left_marker, right_marker = 1, 2
    fdim = mesh.topology.dim - 1

    left_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, left)
    right_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, right)

    marked_facets = np.hstack([left_facets, right_facets])
    marked_values = np.hstack(
        [
            np.full_like(left_facets, left_marker),
            np.full_like(right_facets, right_marker),
        ]
    )
    sorted_facets = np.argsort(marked_facets)
    facet_tag = dolfinx.mesh.meshtags(
        mesh, fdim, marked_facets[sorted_facets], marked_values[sorted_facets]
    )

    # Dirichlet BCs for FF (Fiber-Fiber) stretching experiment as described in the paper
    u_left = np.array((0.0, 0.0, 0.0), dtype=dolfinx.default_scalar_type)
    left_dofs = dolfinx.fem.locate_dofs_topological(
        P2, facet_tag.dim, facet_tag.find(left_marker)
    )

    stretch_value = 0.15  # 15% stretch in fiber direction
    u_right = np.array((stretch_value, 0.0, 0.0), dtype=dolfinx.default_scalar_type)
    right_dofs = dolfinx.fem.locate_dofs_topological(
        P2, facet_tag.dim, facet_tag.find(right_marker)
    )

    bcs = [
        dolfinx.fem.dirichletbc(u_left, left_dofs, P2),
        dolfinx.fem.dirichletbc(u_right, right_dofs, P2),
    ]

    R = Ru
    K = ufl.derivative(Ru, u, ufl.TrialFunction(P2))

    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "superlu_dist",
        "snes_monitor": None,
        "ksp_monitor": None,
    }

    problem = dolfinx.fem.petsc.NonlinearProblem(
        F=R,
        J=K,
        u=u,
        bcs=bcs,
        petsc_options_prefix="nls_",
        petsc_options=petsc_options,
    )

    t0 = time.perf_counter()
    problem.solve()
    t1 = time.perf_counter()
    print(f"Time to solve: {t1 - t0:.3f} s")

    if save_solution:
        shutil.rmtree("compressible_emi.bp", ignore_errors=True)
        vtx = dolfinx.io.VTXWriter(mesh.comm, "compressible_emi.bp", [u])
        vtx.write(0.0)

    Ta.value = 50.0

    t0 = time.perf_counter()
    problem.solve()
    t1 = time.perf_counter()

    print(f"Time to solve: {t1 - t0:.3f} s")

    if plot_matrix:
        import matplotlib.pyplot as plt

        plt.spy(problem.A[:, :])
        plt.savefig("A_compressible_emi.png", dpi=300)

    if save_matrix:
        from petsc4py import PETSc

        viewer = PETSc.Viewer().createASCII(
            "compressible_emi.mtx", mode=PETSc.Viewer.Mode.WRITE
        )
        viewer.pushFormat(PETSc.Viewer.Format.ASCII_MATRIXMARKET)
        viewer.view(problem.A)
        viewer.destroy()

    if save_solution:
        vtx.write(1.0)


if __name__ == "__main__":
    main()
