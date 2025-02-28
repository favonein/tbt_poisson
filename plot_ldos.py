import numpy as np
import matplotlib.pyplot as plt
from sisl.io.siesta import fdfSileSiesta
from sisl.io.tbtrans import tbtncSileTBtrans
import os
import argparse

dir_map = {'x': 0, 'a': 0, 'y': 1, 'b': 1, 'z': 2, 'c': 2}

# Read some system input for files and geometry orientation
parser = argparse.ArgumentParser(description="Compute LDOS from TBTrans Calculation")
parser.add_argument("data_file", type=str,
                    help=".npy file to load")
parser.add_argument("syslabel", type=str,
                    help="System Label before file endings (e.g. LABEL for LABEL.fdf, LABEL.TBT.nc)")
parser.add_argument("transport_direction", type=str, choices=["x", "y", "z", "a", "b", "c"],
                    help="Transport direction (xyz or abc)")
parser.add_argument("vacuum_direction", type=str, choices=["x", "y", "z", "a", "b", "c"],
                    help="Vacuum direction (xyz or abc)")

# Parse arguments
args = parser.parse_args()
ldos = args.data_file
syslabel = args.syslabel
transport = dir_map[args.transport_direction]
vacuum = dir_map[args.vacuum_direction]

print(f"Transport axis: {transport}")
print(f"Vacuum axis:    {vacuum}")

if transport == vacuum:
    print("Transport and Vacuum direction cannot be same")
    raise ValueError("Transport and Vacuum direction cannot be the same")

periodic = ({0, 1, 2} - {transport, vacuum}).pop()   # Direction that is not transport or vacuum

def plot_ldos(file_path, x_range=None, z_range=None, e_vals=None, ldos_tol=1e-6, cutoff=5):
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

    dx = x_vals[1] - x_vals[0]
    dz = z_vals[1] - z_vals[0]
    dE = e_vals[1] - e_vals[0]
    print(f"Grid Density: dx={dx:.6f}, dz={dz:.6f}, dE={dE:.4f}")

    cut = int(cutoff / dx) # Max radius, so we get solidly into region with existant LDOS
    # Extract scattering region from grid
    ldos_x = np.sum(data, axis=(1,2)) * dz * dE
    # Get valid indices where LDOS integral is above tolerance
    valid_idx = np.where(ldos_x > ldos_tol)[0]
    # Remove one more angstrom from either side to get comfortably in the Scatt region
    valid_idx = valid_idx[cut:-cut]
    ldos_scatt = data[valid_idx, :, :]  # Shape: (n_valid_x, nz, nE)
    x_vals = x_vals[valid_idx]

    print(f"LDOS scattering grid shape: {ldos_scatt.shape}")
    print(f"LDOS plot range is: ({x_vals[0]:.3f},{x_vals[-1]:.3f}), ({e_min:.3f},{e_max:.3f})")

    # Plot based on averaging preference
    ldos_avg = np.sum(ldos_scatt, axis=1)  # Integrate over vacuum direction
    plt.imshow(ldos_avg.T, aspect='auto', origin='lower',
                extent=[x_vals[0], x_vals[-1], e_min, e_max],
                cmap='inferno', vmin=0, vmax=max(ldos_avg / 10))
    plt.colorbar(label="LDOS (1/eV*Ang^2)")
    plt.xlabel("x (Ang)")
    plt.ylabel("Energy (eV)")

    return plt


# Get Lattice information
fdf = fdfSileSiesta("Device.fdf")
tbt = tbtncSileTBtrans("Device.TBT.nc")

cutoff = fdf.read_geometry().maxR()
lat = fdf.read_lattice(True)
energy_axis = tbt.E  # eV

x_len = np.linalg.norm(lat.cell[transport])  # Transport direction (a1)
z_len = np.linalg.norm(lat.cell[vacuum])  # Out-of-plane direction (a3)

print(f"Loaded Device Geometry: \n{lat.cell}")
# Process ldos and plot
plt = plot_ldos(ldos, x_range=x_len, z_range=z_len, e_vals=energy_axis, cutoff=cutoff)
plt.savefig('ldos_e.png')
