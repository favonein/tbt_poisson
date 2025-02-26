import matplotlib.pyplot as plt
import matplotlib
import numpy as np


def plot_potential(calc, sys):
    """Plot final potential"""
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(sys.x_full / 10, sys.x_axis / 10, calc.phi.T, shading='auto', cmap='inferno')
    plt.colorbar(label="Electrostatic Potential (V)")
    plt.xlabel("x (nm)")
    plt.ylabel("z (nm)")
    plt.title("Electrostatic Potential Distribution")

    return plt.gcf()


def plot_convergence(sys, phi_history, rho_history, fn_history, fp_history, iterations):
    """Plot convergence metrics."""
    plt.figure(figsize=(15, 10))

    # Potential convergence
    plt.subplot(2, 2, 1)
    for i in range(1, iterations):
        if i % 5 == 0 or i == iterations - 1:  # Plot every 5th iteration + final
            plt.plot(phi_history[i, sys.Nx // 2, :], label=f'Iter {i}')
    plt.xlabel('z coordinate')
    plt.ylabel('Potential (eV)')
    plt.title('Potential Convergence at x=L/2')
    plt.legend()

    # Charge density convergence
    plt.subplot(2, 2, 2)
    for i in range(1, iterations):
        if i % 5 == 0 or i == iterations - 1:
            plt.plot(rho_history[i, sys.Nx // 2, :], label=f'Iter {i}')
    plt.xlabel('z coordinate')
    plt.ylabel('Charge density')
    plt.title('Charge Density Convergence at x=L/2')
    plt.legend()

    # Fermi level convergence
    plt.subplot(2, 2, 3)
    plt.plot(range(iterations), fn_history[:iterations], label='E_Fn')
    plt.plot(range(iterations), fp_history[:iterations], label='E_Fp')
    plt.xlabel('Iteration')
    plt.ylabel('Energy (eV)')
    plt.title('Fermi Level Convergence')
    plt.legend()

    # Convergence rate
    plt.subplot(2, 2, 4)
    deltas = np.array([np.linalg.norm(phi_history[i] - phi_history[i - 1])
                       for i in range(1, iterations)])
    plt.semilogy(range(1, iterations), deltas)
    plt.xlabel('Iteration')
    plt.ylabel('||phi_i - phi_{i-1}||')
    plt.title('Convergence Rate')

    plt.tight_layout()

    return plt.gcf()


def plot_band_diagram(phi, x_full, z_index, E_c, E_v, E_Fn, E_Fp, Nx_metal, Nx_junction):
    """Plot band diagram along x at a specific z-index."""
    plt.figure(figsize=(10, 6))

    # Interface positions
    x_metal_interface = x_full[Nx_metal]
    x_junction_end = x_full[Nx_metal + Nx_junction]

    # Calculate band edges
    E_c_x = E_c - phi[:, z_index]
    E_v_x = E_v - phi[:, z_index]

    # Plot band edges
    plt.plot(x_full, E_c_x, 'b-', label='E_c')
    plt.plot(x_full, E_v_x, 'r-', label='E_v')

    # Plot Fermi levels (can be position dependent)
    if isinstance(E_Fn, np.ndarray):
        plt.plot(x_full, E_Fn[:, z_index], 'g--', label='E_Fn')
        plt.plot(x_full, E_Fp[:, z_index], 'm--', label='E_Fp')
    else:
        plt.axhline(y=E_Fn, color='g', linestyle='--', label='E_Fn')
        plt.axhline(y=E_Fp, color='m', linestyle='--', label='E_Fp')

    # Mark interfaces
    plt.axvline(x=x_metal_interface, color='k', linestyle=':', label='Metal-Junction')
    plt.axvline(x=x_junction_end, color='k', linestyle='-.', label='Junction-SC')

    plt.xlabel('Position (Å)')
    plt.ylabel('Energy (eV)')
    plt.title(f'Band Diagram at z-index = {z_index}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return plt.gcf()


def plot_electron_density(n, x_full, z_axis, Nx_metal, Nx_junction, vacuum_mask=None):
    """Plot electron density with interface markers."""
    plt.figure(figsize=(10, 8))

    # Create electron density plot
    im = plt.pcolormesh(x_full, z_axis, n.T, cmap='plasma',
                        norm=matplotlib.colors.LogNorm(vmin=max(n.min(), 1e-6), vmax=n.max()))

    # Add interface lines
    plt.axvline(x=x_full[Nx_metal], color='w', linestyle='--', label='Metal-Junction')
    plt.axvline(x=x_full[Nx_metal + Nx_junction], color='w', linestyle=':', label='Junction-SC')

    # If vacuum mask is provided, show boundary
    if vacuum_mask is not None:
        # Find boundary points between vacuum and non-vacuum
        boundary = np.zeros_like(vacuum_mask, dtype=bool)
        for i in range(1, vacuum_mask.shape[0] - 1):
            for j in range(1, vacuum_mask.shape[1] - 1):
                if not vacuum_mask[i, j] and vacuum_mask[i, j + 1]:  # Non-vacuum to vacuum (upper)
                    boundary[i, j] = True

        # Get x,z coordinates of boundary
        x_boundary, z_boundary = np.where(boundary)
        plt.scatter(x_full[x_boundary], z_axis[z_boundary],
                    c='cyan', s=1, alpha=0.5, label='Vacuum Boundary')

    plt.colorbar(im, label='Electron Density')
    plt.xlabel('Position (Å)')
    plt.ylabel('z (Å)')
    plt.title('Electron Density Distribution')
    plt.legend()

    return plt.gcf()


def calculate_barrier_height(phi, Nx_metal, Nx_junction, E_c, E_v):
    """Calculate effective Schottky barrier height."""
    # Find maximum band bending in the junction
    junction_slice = phi[Nx_metal:Nx_metal + Nx_junction, :]
    max_band_bending = np.max(junction_slice)

    # Calculate barriers for electrons and holes
    SBH_n = E_c - max_band_bending  # Barrier for electrons
    SBH_p = max_band_bending - E_v  # Barrier for holes

    return SBH_n, SBH_p


def extract_depletion_width(n, p, x_full, Nx_metal, Nx_junction, bulk_n, threshold=0.1):
    """Extract depletion width based on carrier concentration."""
    # Get average carrier density along x at the middle z-index
    z_mid = n.shape[1] // 2
    n_profile = n[:, z_mid]

    # Define bulk value (average in bulk region)
    bulk_index_start = Nx_metal + Nx_junction + 50  # 50 points into bulk SC
    if bulk_index_start < len(n_profile):
        bulk_value = np.mean(n_profile[bulk_index_start:])
    else:
        bulk_value = bulk_n  # Use provided bulk value

    # Find where carrier density reaches bulk_value * threshold
    junction_end = Nx_metal + Nx_junction
    x_depletion = x_full[junction_end:]
    n_depletion = n_profile[junction_end:]

    # Find first point where n >= threshold * bulk_value
    threshold_indices = np.where(n_depletion >= threshold * bulk_value)[0]

    if len(threshold_indices) > 0:
        depletion_point = threshold_indices[0]
        depletion_width = x_depletion[depletion_point] - x_full[junction_end]
    else:
        depletion_width = None  # Depletion region extends beyond simulation

    return depletion_width

