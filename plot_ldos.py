import numpy as np
import matplotlib.pyplot as plt
from sisl.io.siesta import fdfSileSiesta
from sisl.io.tbtrans import tbtncSileTBtrans



def plot_ldos(file_path, x_range=None, z_range=None, e_vals=None, average_z=True, ldos_tol=1e-6):
    """
    Plots the Local Density of States (LDOS) from a saved .npy file.

    Parameters:
    - file_path (str): Path to the .npy file containing the LDOS data.
    - x_range (float): max range of x-axis (transport direction).
    - z_range (float): max range of z-axis (out-of-plane direction)
    - energy_range (tuple): (min, max) in the .npy file.
    - average_z (bool): If True, averages LDOS over z-dimension, else averages over x
    """

    # Load LDOS data
    data = np.load(file_path)
    Nx, Nz, NE = data.shape
    print(f"Loaded LDOS grid with shape: {data.shape} (Nx={Nx}, Nz={Nz}, NE={NE})")

    # Scale to proper ranges
    x_vals = np.linspace(0, x_range, Nx)  # X-direction
    z_vals = np.linspace(0, z_range, Nz)  # Z-direction
    e_min, e_max = e_vals[0], e_vals[-1]

    dz = z_vals[1] - z_vals[0]
    dE = e_vals[1] - e_vals[0]

    # Extract scattering region from grid
    ldos_x = np.sum(data, axis=(0)) * dE * dz
    # Get valid indices where LDOS integral is above tolerance
    valid_idx = np.where(ldos_x >= ldos_tol)[0]
    ldos_scatt = data[valid_idx, :, :]  # Shape: (n_valid_x, nz, nE)

    print(f"LDOS data range is: (0,{x_range}),(0,{z_range}),({e_min},{e_max})")

    # Plot based on averaging preference
    if average_z:
        dz = z_vals[1] - z_vals[0]
        ldos_avg = np.sum(ldos_scatt, axis=1) * dz  # Integrate over Z
        plt.imshow(ldos_avg.T, aspect='auto', origin='lower',
                   extent=[0, x_range, e_min, e_max],
                   cmap='inferno')
        plt.colorbar(label="LDOS (1/eV*Ang)")
        plt.xlabel("x (Ang)")
        plt.ylabel("Energy (eV)")
    else:
        dx = x_vals[1] - x_vals[0]
        ldos_avg = np.sum(ldos_scatt, axis=0) * dx  # Integrate over x
        plt.imshow(ldos_avg.T, aspect='auto', origin='lower',
                   extent=[0, z_range, e_min, e_max],
                   cmap='inferno')
        plt.colorbar(label="LDOS (1/eV*Ang)")
        plt.xlabel("z (Ang)")
        plt.ylabel("Energy (eV)")

    return plt

# Get Lattice information

fdf = fdfSileSiesta("Device_long.fdf")
tbt = tbtncSileTBtrans("Device.TBT.nc")

lat = fdf.read_lattice(True)
a1, a2, a3 = lat.cell  # Extract lattice vectors
energy_axis = tbt.E  # eV

x_len = np.linalg.norm(a1)  # Transport direction (a1)
z_len = np.linalg.norm(a3)  # Out-of-plane direction (a3)


# Process ldos and plot
plt = plot_ldos("ldos_arr_par.npy", x_range = x_len, z_range = z_len, e_vals = energy_axis, average_z=True)

plt.savefig('ldos_e.png')
