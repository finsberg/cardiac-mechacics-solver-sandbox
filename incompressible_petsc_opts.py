import time
import shutil
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
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


def run_simulation(
    petsc_options,
    N=5,
):
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_cube(comm, N, N, N)

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

    problem = dolfinx.fem.petsc.NonlinearProblem(
        F=R,
        J=K,
        u=states,
        bcs=bcs,
        petsc_options_prefix="nls_",
        petsc_options=petsc_options,
    )

    if petsc_options["pc_type"] == "fieldsplit":
        # # 1. Extract the PETSc SNES solver and PC
        snes = problem.solver
        snes.setFromOptions()  # Apply options so the PC becomes fieldsplit
        ksp = snes.getKSP()
        pc = ksp.getPC()

        # 2. Calculate local sizes of each field from the DofMaps
        size_u = P2.dofmap.index_map.size_local * P2.dofmap.index_map_bs
        size_p = P1.dofmap.index_map.size_local * P1.dofmap.index_map_bs

        # 3. Get global ownership offset for this MPI rank
        ofs_start, ofs_end = problem.x.getOwnershipRange()

        # 4. Create PETSc Index Sets (IS) based on FEniCSx block stacking
        is_u = PETSc.IS().createStride(size_u, first=ofs_start, step=1, comm=comm)
        is_p = PETSc.IS().createStride(
            size_p, first=ofs_start + size_u, step=1, comm=comm
        )

        # 5. Assign the Index Sets to the FieldSplit preconditioner
        pc.setFieldSplitIS(("u", is_u), ("p", is_p))

    t0 = time.perf_counter()
    problem.solve()
    t1 = time.perf_counter()
    print(f"Time to solve: {t1 - t0:.3f} s")


def main():
    for N in [5, 10, 20]:
        print(f"Running simulation with N={N} subdivisions per side...")
        # SuperLU:
        petsc_options = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "superlu_dist",
            # "snes_monitor": None,
            # "ksp_monitor": None,
            # 1. Parallelize the symbolic factorization (Crucial for memory/speed on 3D meshes)
            "mat_superlu_dist_parsymbfact": "true",
            # 2. Use ParMETIS for column permutation (Parallel) instead of serial METIS
            # "mat_superlu_dist_colperm": "PARMETIS",
            # "mat_superlu_dist_colperm": "METIS_AT_PLUS_A",
            # "mat_superlu_dist_colperm": "MMD_AT_PLUS_A",
            # 3. Row permutation (Paper authors found serial MC64 was faster than parallel AWPM here)
            "mat_superlu_dist_rowperm": "LargeDiag_MC64",
            # 4. Prevent zero-pivots on the near-zero pressure diagonal block
            "mat_superlu_dist_replacetinypivots": "true",
        }
        print("Running with SuperLU_DIST direct solver...")
        run_simulation(petsc_options, N=N)

        # MUMPS:
        petsc_options = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            # 1. Increase workspace allocation by 200% (MUMPS often underestimates fill-in for saddle point problems)
            "mat_mumps_icntl_14": 200,
            # 2. Enable null pivot detection (Essential for the zero/near-zero diagonals in the pressure block)
            "mat_mumps_icntl_24": 1,
            # 3. Use parallel analysis (distributes the symbolic factorization workload)
            "mat_mumps_icntl_28": 2,
            # 4. Use auto/default parallel ordering to avoid specific ParMETIS segfaults
            "mat_mumps_icntl_29": 0,
            # "snes_monitor": None,
            # "ksp_monitor": None,
        }
        print("Running with MUMPS direct solver...")
        run_simulation(petsc_options, N=N)

        # FieldSplit Schur complement preconditioner:
        petsc_options = {
            "ksp_type": "fgmres",
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "schur",
            "pc_fieldsplit_schur_fact_type": "full",
            "fieldsplit_u_ksp_type": "cg",
            "fieldsplit_u_pc_type": "gamg",
            "fieldsplit_p_ksp_type": "preonly",
            "fieldsplit_p_pc_type": "jacobi",
            # "snes_monitor": None,
            # "ksp_monitor": None,
        }
        print("Running with FieldSplit Schur complement preconditioner...")
        run_simulation(petsc_options, N=N)


if __name__ == "__main__":
    main()
