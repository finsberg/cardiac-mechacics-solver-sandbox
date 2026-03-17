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


def incompressibility(C, p):
    J = pow(ufl.det(C), 1 / 2)
    return p * (J - 1)


def transverse_holzapfel_ogden(
    C,
    f0,
    a: float = 2.280,
    b: float = 9.726,
    a_f: float = 1.685,
    b_f: float = 15.779,
):
    I1 = ufl.tr(C)
    I4f = ufl.inner(C * f0, f0)

    return (a / (2.0 * b)) * (ufl.exp(b * (I1 - 3)) - 1.0) + (
        a_f / (2.0 * b_f)
    ) * heaviside(I4f - 1) * (ufl.exp(b_f * subplus(I4f - 1) ** 2) - 1.0)


def main():
    plot_matrix = False  # pip install matlotlib to enable plotting
    save_matrix = True
    save_solution = True

    logging.basicConfig(level=logging.INFO)
    dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_cube(comm, 3, 3, 3)

    P2 = dolfinx.fem.functionspace(mesh, ("Lagrange", 2, (3,)))
    P1 = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

    u = dolfinx.fem.Function(P2, name="u")
    p = dolfinx.fem.Function(P1, name="p")
    u_test = ufl.TestFunction(P2)
    p_test = ufl.TestFunction(P1)
    u_trial = ufl.TrialFunction(P2)
    p_trial = ufl.TrialFunction(P1)

    F = ufl.grad(u) + ufl.Identity(3)
    C = ufl.variable(F.T * F)
    J = pow(ufl.det(C), 1 / 2)

    # Active tension
    f0 = dolfinx.fem.Constant(mesh, [1.0, 0.0, 0.0])

    psi = transverse_holzapfel_ogden(C, f0=f0) + incompressibility(C, p)
    S_p = 2.0 * ufl.diff(psi, C)

    Ta = dolfinx.fem.Constant(mesh, 1.0)
    Sa = Ta * ufl.outer(f0, f0)

    S = S_p + Sa

    var_C = ufl.derivative(C, u, u_test)
    Ru = ufl.inner(S, 0.5 * var_C) * ufl.dx
    Rp = (J - 1) * p_test * ufl.dx

    def left(x):
        return np.isclose(x[0], 0)

    def right(x):
        return np.isclose(x[0], 1)

    left_marker = 1
    right_marker = 2

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

    u_bc = np.array((0,) * mesh.geometry.dim, dtype=dolfinx.default_scalar_type)
    left_dofs = dolfinx.fem.locate_dofs_topological(
        P2, facet_tag.dim, facet_tag.find(left_marker)
    )

    bcs = [dolfinx.fem.dirichletbc(u_bc, left_dofs, P2)]

    ds = ufl.ds(domain=mesh, subdomain_data=facet_tag)

    traction = dolfinx.fem.Constant(mesh, -0.5)
    N = ufl.FacetNormal(mesh)
    n = traction * ufl.det(F) * ufl.inv(F).T * N
    Ru += ufl.inner(u_test, n) * ds(right_marker)

    R = [Ru, Rp]
    K = [
        [
            ufl.derivative(Ru, u, u_trial),
            ufl.derivative(Ru, p, p_trial),
        ],
        [
            ufl.derivative(Rp, u, u_trial),
            ufl.derivative(Rp, p, p_trial),
        ],
    ]
    states = [u, p]

    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "superlu_dist",
    }

    problem = dolfinx.fem.petsc.NonlinearProblem(
        F=R,
        J=K,
        u=states,
        bcs=bcs,
        petsc_options_prefix="nls_",
        petsc_options=petsc_options,
    )

    t0 = time.perf_counter()
    problem.solve()
    t1 = time.perf_counter()
    print(f"Time to solve: {t1 - t0:.3f} s")

    if plot_matrix:
        import matplotlib.pyplot as plt

        A = problem.A[:, :]

        plt.spy(A)
        plt.savefig("A_incompressible.png", dpi=300)

    if save_matrix:
        from petsc4py import PETSc

        viewer = PETSc.Viewer().createBinary(
            "incompressible.dat", mode=PETSc.Viewer.Mode.WRITE
        )
        viewer.view(problem.A)
        viewer.destroy()

    if save_solution:
        shutil.rmtree("incompressible.bp", ignore_errors=True)
        with dolfinx.io.VTXWriter(mesh.comm, "incompressible.bp", [u]) as writer:
            writer.write(0.0)


if __name__ == "__main__":
    main()
