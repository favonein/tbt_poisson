import numpy as np
import scipy
from pyamg import ruge_stuben_solver
from pyamg.gallery import poisson
import poisson_geometry

# Consts
k = 8.617333e-5  # eV/K
T = 300  # Kelvin


def fermi(E, Ef):
    return 1 / (1 + np.exp((E - Ef) / (k * T)))


def _setup_poisson(nx, nz, nx_metal):
    """Sets up poisson matrix with B.C. for +/- x and z"""
    A = poisson((nx, nz), format="csr")

    # Convert to dense for easier manipulation (can stay sparse for efficiency)
    A = A.toarray()

    # Prepare masking arrays for boundary nodes
    metal_nodes = np.zeros((nx, nz), dtype=bool)
    metal_nodes[:nx_metal, :] = True

    semiconductor_far_nodes = np.zeros((nx, nz), dtype=bool)
    semiconductor_far_nodes[-1, :] = True  # Far right boundary

    # Create vacuum BC for top and bottom rows if needed
    top_nodes = np.zeros((nx, nz), dtype=bool)
    top_nodes[:, -1] = True

    bottom_nodes = np.zeros((nx, nz), dtype=bool)
    bottom_nodes[:, 0] = True

    # Flatten all BC masks to match matrix indexing
    metal_nodes_flat = metal_nodes.flatten()
    semi_nodes_flat = semiconductor_far_nodes.flatten()
    top_nodes_flat = top_nodes.flatten()
    bottom_nodes_flat = bottom_nodes.flatten()

    # Modify coefficient matrix A for Dirichlet BCs
    # For each BC node, replace row with identity entry
    for idx in np.where(metal_nodes_flat)[0]:
        A[idx, :] = 0  # Zero out row
        A[idx, idx] = 1  # Set diagonal to 1

    for idx in np.where(semi_nodes_flat)[0]:
        A[idx, :] = 0
        A[idx, idx] = 1

    # For Neumann BCs (vacuum interfaces), modify to enforce phi[i] = phi[i±1]
    for idx in np.where(top_nodes_flat)[0]:
        if idx >= nx:  # Skip if on bottom row
            A[idx, :] = 0
            A[idx, idx] = 1
            A[idx, idx - nx] = -1  # phi_top = phi_below_top

    for idx in np.where(bottom_nodes_flat)[0]:
        if idx + nx < nx * nz:  # Skip if on top row
            A[idx, :] = 0
            A[idx, idx] = 1
            A[idx, idx + nx] = -1  # phi_bottom = phi_above_bottom

    A_csr = scipy.sparse.csr_matrix(A)
    return ruge_stuben_solver(A_csr)


class SolverConfig:
    def __init__(self, sys, fn, fp, V=0, tol=1e-6, q_tol=1e-6, max_iter=1000, alpha=0.15):
        # Physical Parameters
        self.poisson_eq = _setup_poisson(sys.Nx, sys.Nz, sys.Nx_metal)
        self.V = V
        self.fn = fn
        self.fp = fp

        # initial phi_e guess is ramp from V to 0 in Junction/SC region
        phi_ramp = np.linspace(V, 0, sys.Nx - sys.Nx_metal)
        self.phi = np.concatenate([V * np.ones(sys.Nx_metal), phi_ramp])

    def calc_charge(self, sys):
        """Vectorized charge calculation."""
        # Create meshgrid for vectorized calculation
        Nx, Nz = self.phi.shape
        E_mesh, X_mesh, Z_mesh = np.meshgrid(sys.e_ax, np.arange(Nx), np.arange(Nz), indexing='ij')

        # Shift energies by potential
        E_shifted = E_mesh - self.phi[X_mesh[0, :, :], Z_mesh[0, :, :]]

        # Interpolate DOS at shifted energies
        D_shifted = np.zeros_like(E_shifted)
        valid_idx = (E_shifted >= sys.energy_axis.min()) & (E_shifted <= sys.energy_axis.max())

        # Use vectorized interpolation for valid indices
        D_shifted[valid_idx] = np.interp(
            E_shifted[valid_idx],
            sys.energy_axis,
            sys.D.reshape(-1, len(sys.energy_axis))[
                X_mesh[valid_idx] * Nz + Z_mesh[valid_idx],
                np.zeros_like(X_mesh[valid_idx], dtype=int)
            ]
        )

        # Apply masks for n and p regions
        mask_n = E_mesh >= (sys.E_c + self.phi[X_mesh[0, :, :], Z_mesh[0, :, :]])
        mask_p = E_mesh <= (sys.E_v + self.phi[X_mesh[0, :, :], Z_mesh[0, :, :]])

        # Integrate
        n = np.trapz(D_shifted * fermi(E_mesh, self.fn) * mask_n, x=sys.e_ax, axis=0)
        p = np.trapz(D_shifted * (1 - fermi(E_mesh, self.fp)) * mask_p, x=sys.e_ax, axis=0)

        return n, p

    def density(self, sys, max_iter=1000, q_tol=1e-6, alpha=0.15):
        """Compute carrier densities according to charge neutrality and Efn-Efp = V"""
        for i in range(max_iter):
            if abs(total_charge) < q_tol:
                print(f'Calculated new charge density')
                print(f'E_fn: {self.fn:.3f}, F_fp: {self.fp:.3f}')
                break

            n_new, p_new = self.calc_charge(sys)
            rho_new = self.c + p_new - n_new  # Charge density

            # Compute total charge imbalance
            total_charge = np.sum(rho_new) * sys.dx * sys.dz

            # Update Fermi levels based on charge imbalance -- Check
            fn_new = self.fn + alpha * (total_charge / np.sum(self.n * sys.dx * sys.dz))
            fp_new = self.fp + alpha * (total_charge / np.sum(self.p * sys.dx * sys.dz))

            # Update fermi levels, enforcing E_fn - E_fp = V
            diff = fn_new - fp_new
            self.fn = fn_new - (diff - self.V) / 2
            self.fp = fp_new + (diff - self.V) / 2

        return n_new, p_new, self.fn, self.fp

    def poisson_pyamg(self, dx, dz):
        """Solve Poisson's equation with pre-computed solver."""
        b = -4 * np.pi * self.rho.flatten() * dx * dz
        phi_solve = self.poisson_eq.solve(b, tol=1e-8)
        phi = np.array(phi_solve).reshape(self.rho.shape)
        return phi

    def solve_iter(self, sys, c, max_iter=1000, tol=1e-6, alpha=0.15):
        # For diagnostic
        phi_history = np.zeros((max_iter, sys.Nx, sys.Nz))
        rho_history = np.zeros((max_iter, sys.Nx, sys.Nz))
        fn_history = np.zeros(max_iter)
        fp_history = np.zeros(max_iter)

        for iteration in range(max_iter):

            print(f'Iteration: {iteration}')

            # Solve for carrier densities
            n, p, E_Fn, E_Fp = self.density(sys)
            rho = c + p - n  # Charge density

            # Store for monitoring
            rho_history[iteration] = rho
            fn_history[iteration] = E_Fn
            fp_history[iteration] = E_Fp

            # Solve Poisson equation
            phi_new = self.poisson_pyamg(sys.dx, sys.dz)
            phi_history[iteration] = phi_new

            # Check convergence
            delta = np.linalg.norm(phi_new - self.phi) / np.linalg.norm(self.phi)
            if delta < tol:
                print(f'Relative phi delta: {delta:.5e}')
                print(f'Successfully converged potential')
                print(f'Converged in {iteration} iterations')
                break

            print(f'Relative phi delta: {delta:.5e}')

            self.phi = alpha * self.phi + (1 - alpha) * phi_new

        return phi_history, rho_history, fn_history, fp_history, iteration
