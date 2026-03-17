# Cardiac mechanics solver sandbox

This repository contains two examples from cardiac mechanics using a unit cube geometry. The intent is to have a sandbox for testing different solver options and preconditioners for the mechanics problem in a self-contained way.

You will find two examples, one for the incompressible case and one for the compressible case. 

To run the code you need FEniCSx installled. The easiest way to get it is using Docker. You can first pull the image and start a container with the current directory mounted as a volume:
```bash
docker run --name cardiac -w /home/shared -v $PWD:/home/shared -it ghcr.io/fenics/dolfinx/dolfinx:v0.10.0
```
Then you can run the examples using
```bash
python3 incompressible.py
```
or (for the compressible case)
```bash
python3 compressible.py
```

## Mathmatical background


### Physics & Kinematics Overview

In all models the material we are is using a transverse Holzapfel-Ogden strain energy density function $\Psi$, which is commonly used to model fibrous biological tissues (such as myocardium). An active tension component is also superimposed along the fiber direction using an active stress approach (another way to do this is to use an active strain approach).

The fundamental kinematic quantities driving the non-linearity in both models are:

Deformation Gradient: $\mathbf{F}(\mathbf{u}) = \mathbf{I} + \nabla \mathbf{u}$

Right Cauchy-Green Tensor: $\mathbf{C}(\mathbf{u}) = \mathbf{F}^T \mathbf{F}$

Volume Ratio (Jacobian): $J(\mathbf{u}) = \det(\mathbf{F}) = \sqrt{\det(\mathbf{C})}$

These non-linear formulations use the Newton-Raphson method to solve the equilibrium equations, generating a sequence of linearized algebraic systems:

$$\mathcal{K} \Delta \mathbf{x} = - \mathbf{R}$$

where $\mathcal{K}$ is the Jacobian matrix, $\mathbf{R}$ is the non-linear residual, and $\mathbf{x}$ contains the unknown degrees of freedom.

## Part I: Tissue-Level (Homogenized) Models

These models assume that cellular and extracellular structures scale uniformly through the tissue. They are subjected to an active tension component and a non-linear boundary traction (follower load).



### Example 1: Compressible Formulation

In the compressible formulation, the problem is solved strictly for the displacement field $\mathbf{u} \in \mathcal{V}$ using $P_1$ (Lagrange) elements.

#### Continuous Formulation

The strain energy $\Psi(\mathbf{C})$ is split into an isochoric (deviatoric) part and a volumetric part. Compressibility is penalized using a bulk modulus $\kappa$:

$$\Psi_{\text{vol}} = \kappa (J \ln J - J + 1)$$

The system is governed by the principle of virtual work. We seek $\mathbf{u}$ such that the non-linear residual $R_{\mathbf{u}}$ vanishes for all test functions $\delta\mathbf{u} \in \mathcal{V}$:

$$R_{\mathbf{u}}(\mathbf{u}; \delta\mathbf{u}) = \int_\Omega \mathbf{S}(\mathbf{u}) : \frac{1}{2}\delta\mathbf{C} \,\text{d}x - \int_{\Gamma_{\text{right}}} \mathbf{t} \cdot \delta\mathbf{u} \,\text{d}s = 0$$

where $\mathbf{S} = 2 \frac{\partial \Psi}{\partial \mathbf{C}} + \mathbf{S}_{\text{active}}$ is the highly non-linear Second Piola-Kirchhoff stress tensor, and $\mathbf{t}$ is an applied pressure boundary traction.

#### Linear Algebra Profile

Because this is a single-field formulation, the resulting Jacobian matrix $\mathcal{K} = \frac{\partial \mathbf{R}_{\mathbf{u}}}{\partial \mathbf{u}}$ is a standard stiffness matrix. While it is sparse, it can become highly ill-conditioned under large deformations due to the exponential nature of the Holzapfel-Ogden model.

Matrix Output: Saved as compressible.dat (PETSc binary) and A_compressible.png (sparsity pattern).

### Example 2: Incompressible (Mixed) Formulation

Biological tissues are largely composed of water and are therefore practically incompressible ($J \approx 1$). Enforcing strict incompressibility using only displacements leads to severe volumetric locking.

To overcome this, the incompressible example uses a mixed formulation solving for both the displacement field ($\mathbf{u} \in \mathcal{V}$) and the hydrostatic pressure field ($p \in \mathcal{Q}$) using Taylor-Hood ($P_2 / P_1$) elements.

#### Continuous Formulation

The strain energy potential is modified to $\Psi_{\text{iso}}(\mathbf{C}) + p(J-1)$. The corresponding variational equations dictate that the residuals $R_{\mathbf{u}}$ and $R_p$ must vanish for all test functions $\delta\mathbf{u} \in \mathcal{V}$ and $\delta p \in \mathcal{Q}$:

$$\begin{align*}
    R_{\mathbf{u}}(\mathbf{u}, p; \delta\mathbf{u}) &= \int_\Omega \mathbf{S}(\mathbf{u}, p) : \frac{1}{2}\delta\mathbf{C} \,\text{d}x - \int_{\Gamma_{\text{right}}} \mathbf{t} \cdot \delta\mathbf{u} \,\text{d}s = 0 \\
    R_p(\mathbf{u}, p; \delta p) &= \int_\Omega (J(\mathbf{u}) - 1) \delta p \,\text{d}x = 0
\end{align*}$$


#### Linear Algebra Profile

The mixed formulation fundamentally changes the structure of the discrete Jacobian matrix into a saddle-point problem:

$$
\begin{bmatrix}
    \mathcal{K}_{\mathbf{u}\mathbf{u}} & \mathcal{K}_{\mathbf{u}p} \\
    \mathcal{K}_{p\mathbf{u}} & \mathcal{K}_{pp}
\end{bmatrix}
\begin{bmatrix}
    \Delta \mathbf{u} \\
    \Delta p
\end{bmatrix} = - \begin{bmatrix}
    \mathbf{R}_{\mathbf{u}} \\
    \mathbf{R}_p
\end{bmatrix}
$$


## Part II: Microscale (EMI) Models

Cardiomyocytes are the functional building blocks of the heart. The EMI (Extracellular-Membrane-Intracellular) framework replaces homogenization with explicit 3D geometry. The domain $\Omega$ is partitioned into the intracellular space ($\Omega_i$) and the extracellular matrix ($\Omega_e$).

The scripts implement a pure passive stretch experiment (15% Dirichlet boundary displacement on the right face) without boundary tractions, mimicking in-vitro tensile testing.

See https://doi.org/10.1007/s10237-022-01660-8 for more details on the mathematical formulation and numerical methods.

### Example 3: Compressible EMI Formulation (compressible_emi.py)

The weak form is explicitly partitioned across the two subdomains, applying different constitutive laws:

Extracellular ($\Omega_e$): Isotropic hyperelasticity (parameters $a_e, b_e$).

Intracellular ($\Omega_i$): Transversely isotropic Holzapfel-Ogden (parameters $a_i, b_i, a_{if}, b_{if}$) modeling the sarcomere structure.

$$R_{\mathbf{u}}(\mathbf{u}; \delta\mathbf{u}) = \int_{\Omega_e} \mathbf{S}_e : \frac{1}{2}\delta\mathbf{C} \,\text{d}x + \int_{\Omega_i} \mathbf{S}_i : \frac{1}{2}\delta\mathbf{C} \,\text{d}x = 0$$

Linear Algebra Profile

Unlike the homogenized tissue model, this problem lacks the asymmetric follower load, making the underlying stiffness matrix naturally symmetric.
However, the sharp discontinuity in material parameters (stiffness) across the membrane boundary $\Gamma$ introduces heterogeneous block scaling. Matrix entries corresponding to degrees of freedom sitting exactly on the membrane interface couple drastically different stiffness regimes, which can severely degrade the performance of simple preconditioners (like Jacobi or standard AMG).

### Example 4: Incompressible (Mixed) EMI Formulation (incompressible_emi.py)

This combines the partitioned geometric approach of the EMI model with the mixed $P_2 / P_1$ saddle-point formulation.

$$\begin{align*}
    R_{\mathbf{u}} &= \int_{\Omega_e} \mathbf{S}_e(\mathbf{u}, p) : \frac{1}{2}\delta\mathbf{C} \,\text{d}x + \int_{\Omega_i} \mathbf{S}_i(\mathbf{u}, p) : \frac{1}{2}\delta\mathbf{C} \,\text{d}x = 0 \\
    R_p &= \int_{\Omega} (J - 1) \delta p \,\text{d}x = 0
\end{align*}$$




## License
MIT
