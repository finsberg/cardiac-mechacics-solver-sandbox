import time
import numpy as np
from mpi4py import MPI
import dolfinx
import dolfinx.fem.petsc
import ufl


def subplus(x):
    return ufl.conditional(ufl.ge(x, 0.0), x, 0.0)


def heaviside(x):
    return ufl.conditional(ufl.ge(x, 0.0), 1.0, 0.0)


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


def compressibility(C, kappa=1e3):
    J = pow(ufl.det(C), 1 / 2)
    return kappa * (J * ufl.ln(J) - J + 1)


def run_simulation(
    petsc_options,
    N=5,
    degree=2,
):
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_cube(comm, N, N, N)

    # Function space for the displacement
    P1 = dolfinx.fem.functionspace(mesh, ("Lagrange", degree, (3,)))
    quadrature_metadata = {"quadrature_degree": 4}
    dx = ufl.dx(metadata=quadrature_metadata)
    # The displacement
    u = dolfinx.fem.Function(P1)
    u_test = ufl.TestFunction(P1)

    # Compute the deformation gradient
    F = ufl.grad(u) + ufl.Identity(3)
    C = ufl.variable(F.T * F)
    J = ufl.det(F)
    Cdev = J ** (-2.0 / 3.0) * C

    # Active tension
    # Set fiber direction to be constant in the x-direction
    f0 = dolfinx.fem.Constant(mesh, [1.0, 0.0, 0.0])

    psi = transverse_holzapfel_ogden(Cdev, f0=f0) + compressibility(C)
    S_p = 2.0 * ufl.diff(psi, C)

    Ta = dolfinx.fem.Constant(mesh, 10.0)
    Sa = Ta * ufl.outer(f0, f0)

    S = S_p + Sa

    var_C = ufl.derivative(C, u, u_test)
    Ru = ufl.inner(S, 0.5 * var_C) * dx

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
        P1, facet_tag.dim, facet_tag.find(left_marker)
    )
    bcs = [dolfinx.fem.dirichletbc(u_bc, left_dofs, P1)]

    ds = ufl.ds(domain=mesh, subdomain_data=facet_tag, metadata=quadrature_metadata)

    # We can also apply a force on the right boundary using a Neumann boundary condition
    traction = dolfinx.fem.Constant(mesh, -0.5)
    N = ufl.FacetNormal(mesh)
    n = traction * ufl.det(F) * ufl.inv(F).T * N
    Ru += ufl.inner(u_test, n) * ds(right_marker)

    R = Ru
    K = ufl.derivative(Ru, u, ufl.TrialFunction(P1))

    problem = dolfinx.fem.petsc.NonlinearProblem(
        F=R,
        J=K,
        u=u,
        bcs=bcs,
        petsc_options_prefix="nls",
        petsc_options=petsc_options,
    )

    t0 = time.perf_counter()
    problem.solve()
    t1 = time.perf_counter()
    print(f"Time to solve: {t1 - t0:.3f} s")


def main():
    # logging.basicConfig(level=logging.INFO)
    # dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
    for N in [3, 5, 10]:
        print(f"Running simulation with N={N} subdivisions per side...")

        petsc_options = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "superlu_dist",
            # "snes_monitor": None,
            # "ksp_monitor": None,
        }
        print("Running with SuperLU direct solver...")
        run_simulation(petsc_options, N=N)

        petsc_options = {
            "ksp_type": "cg",  # Conjugate Gradient (great for symmetric matrices)
            "pc_type": "gamg",  # Algebraic Multigrid
            "pc_gamg_type": "agg",  # Aggregation AMG
            # "snes_monitor": None,  # Print Newton iterations
            # "ksp_monitor": None,  # Print Krylov iterations
        }
        print("Running with GAMG preconditioner and CG solver...")
        run_simulation(petsc_options, N=N)


if __name__ == "__main__":
    main()
